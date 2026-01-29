"""
Train the learnable (soft) prompt tokens for CLIP on the dataset

CORE IDEA:
    * CLIP image encoder is used as-is (pretrained)
    * CLIP text encoder is also frozen (no fine-tuning of CLIP weights)
    * We learn only a small set of prompt tokens (context tokens)
      -> parameter-efficient + fast training

This script:
1) Loads config (configs/default.yaml)
2) Loads pretrained CLIP + its preprocessing transform
3) Builds Food101 subset train/test dataloaders
4) Builds PromptTunedCLIP (CLIP frozen + trainable soft prompts)
5) Trains only the prompt tokens with cross-entropy loss
6) Evaluates after each epoch and saves the best prompt tokens
7) Saves a training summary json
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import yaml
import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import make_loaders
from src.eval import evaluate
from src.model import PromptTunedCLIP
from src.utils import set_seed, get_device


def _to_float(x: Any, default: float) -> float:
    """
    Safely convert config values that might be strings into floats.

    Why needed?
    YAML config values sometimes come as strings (e.g., "1e-4") depending on how config is written.
    This helper ensures lr / weight_decay become proper floats, otherwise falls back to default.
    """
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def main(cfg_path: str = "configs/default.yaml"):
    # ----------------------------
    # 1) Load config file
    # ----------------------------
    # cfg is a nested dict that contains model/data/train settings.
    with open(cfg_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # ----------------------------
    # 2) Reproducibility
    # ----------------------------
    # Fix random seeds so results are more repeatable.
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # Decide whether to run on CPU or GPU (depending on cfg and availability).
    device = get_device(cfg.get("device", "cpu"))

    # ----------------------------
    # 3) Load pretrained CLIP + tokenizer + preprocess
    # ----------------------------
    model_name = cfg["model"]["clip_name"]
    pretrained = cfg["model"]["pretrained"]

    # open_clip returns:
    # - clip_model: the pretrained model
    # - preprocess: transforms that produce the expected input format for that CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        # force_quick_gelu=True,  # uncomment if you want to silence mismatch warnings for some weights
    )

    # Tokenizer converts text prompts into token IDs for the CLIP text encoder
    tokenizer = open_clip.get_tokenizer(model_name)

    # Move CLIP model to selected device (CUDA/CPU)
    clip_model = clip_model.to(device)

    # ----------------------------
    # 4) Build dataset loaders (Food101 subset)
    # ----------------------------
    subset_classes = cfg["data"]["subset_classes"]
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))

    # make_loaders builds:
    # - a training loader (shuffled)
    # - a test loader (not shuffled)
    # It also filters Food101 to only subset_classes and remaps labels to [0..K-1].
    train_loader, test_loader = make_loaders(
        root=cfg["data"]["root"],
        subset_classes=subset_classes,
        train_transform=preprocess,
        eval_transform=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ----------------------------
    # 5) Create prompt-tuned model
    # ----------------------------
    # PromptTunedCLIP:
    # - freezes the CLIP model weights
    # - learns only soft prompt context vectors (n_ctx tokens)
    n_ctx = int(cfg["model"]["n_ctx"])
    model = PromptTunedCLIP(clip_model, tokenizer, subset_classes, n_ctx=n_ctx).to(device)

    # ----------------------------
    # 6) Sanity checks (optional but helpful)
    # ----------------------------
    # Before training, we verify that the prompt module produces valid text embeddings:
    # - correct shape
    # - no NaNs
    # - reasonable norms
    # - similarity matrix isn't degenerate
    with torch.no_grad():
        # Some implementations expose encode_text(), others just use forward()
        if hasattr(model.prompt, "encode_text"):
            tf = model.prompt.encode_text()
        else:
            tf = model.prompt()  # calls forward()

        print("Sanity check: text_features shape:", tuple(tf.shape))
        print("Sanity check: any NaN:", torch.isnan(tf).any().item())
        print("Sanity check: mean norm:", tf.norm(dim=1).mean().item())

        # Mean similarity between all class embeddings (rough diagnostic)
        sim = (tf @ tf.t()).mean().item()
        print("Sanity check: mean cosine(similarity matrix):", sim)

    # ----------------------------
    # 7) Optimizer setup (ONLY prompt parameters)
    # ----------------------------
    # We train only the soft prompt parameters (ctx tokens), not the whole CLIP model.
    lr = _to_float(cfg["train"].get("lr", 1e-4), default=1e-4)
    weight_decay = _to_float(cfg["train"].get("weight_decay", 0.0), default=0.0)

    optimizer = torch.optim.AdamW(
        model.prompt.parameters(),   # IMPORTANT: only prompt learner parameters are optimized
        lr=lr,
        weight_decay=weight_decay,
    )

    # Cross-entropy loss for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # ----------------------------
    # 8) Training loop
    # ----------------------------
    epochs = int(cfg["train"]["epochs"])
    best_top1 = -1.0  # track best Top-1 accuracy to save best checkpoint

    # Create output folders if missing
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    best_path = "checkpoints/best.pt"

    for epoch in range(epochs):
        # Enable training mode for prompt module (CLIP remains frozen, but this is still standard practice)
        model.train()

        # Progress bar for training batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        for step, (images, targets) in enumerate(pbar, start=1):
            # Move batch to device; non_blocking can speed transfers when using pinned memory
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Clear previous gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass:
            # model(images) returns logits [B, K] via image/text similarity
            logits = model(images)

            # Compute classification loss
            loss = criterion(logits, targets)

            # Backprop only affects prompt parameters
            loss.backward()

            # Update prompt parameters
            optimizer.step()

            # Track average training loss for display
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / step)

        # ----------------------------
        # 9) Evaluation after each epoch
        # ----------------------------
        # evaluate() computes average Top-1 and Top-5 over the test set
        metrics = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: top1={metrics['top1']:.4f}, top5={metrics['top5']:.4f}")

        # ----------------------------
        # 10) Save best checkpoint (only ctx tokens)
        # ----------------------------
        # We save ONLY the learned context vectors to keep the checkpoint very small.
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]

            # model.prompt.ctx is the trainable parameter we care about
            torch.save({"ctx": model.prompt.ctx.detach().cpu()}, best_path)

    # ----------------------------
    # 11) Save training summary
    # ----------------------------
    # This summary is useful for reporting in papers and for reproducibility.
    out = {
        "clip": f"{model_name}/{pretrained}",
        "subset_k": len(subset_classes),
        "n_ctx": n_ctx,
        "seed": seed,
        "lr": lr,
        "weight_decay": weight_decay,
        "best_top1": best_top1,
    }
    with open("results/tuned_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", best_path, "and results/tuned_summary.json")


if __name__ == "__main__":
    # Allows running as a script:
    # python train.py
    main()
