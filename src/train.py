"""
Train the learnable (soft) prompt tokens for CLIP on the dataset

CORE IDEA:
    * CLIP image encoder
    * Text encoder remain FROZEN (no fine-tuning of CLIP weights)
    * Learning only a small set of prompt tokens (context tokens) because it is parameter-efficient and fast





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

from utils import set_seed, get_device
from data import make_loaders
#Todo:define a evaluate in the eval


def _to_float(x: Any, default: float) -> float:
    """Safely convert config values that might be strings into floats."""
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def main(cfg_path: str = "configs/default.yaml"):
    # --- Load config ---
    with open(cfg_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # --- Reproducibility ---
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = get_device(cfg.get("device", "cpu"))

    # --- CLIP setup ---
    model_name = cfg["model"]["clip_name"]
    pretrained = cfg["model"]["pretrained"]

    # You can optionally set force_quick_gelu=True to match OpenAI weights more strictly.
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        # force_quick_gelu=True,  # uncomment if you want to silence the mismatch warning
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    clip_model = clip_model.to(device)

    # --- Data ---
    subset_classes = cfg["data"]["subset_classes"]
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))

    train_loader, test_loader = make_loaders(
        root=cfg["data"]["root"],
        subset_classes=subset_classes,
        train_transform=preprocess,
        eval_transform=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # --- Prompt-tuned model (CLIP frozen, prompt tokens trainable) ---
    n_ctx = int(cfg["model"]["n_ctx"])
    model = PromptTunedCLIP(clip_model, tokenizer, subset_classes, n_ctx=n_ctx).to(device)

    # --- Quick sanity check for text features ---
    # Your SoftPromptLearner might implement forward() (callable) instead of encode_text().
    with torch.no_grad():
        if hasattr(model.prompt, "encode_text"):
            tf = model.prompt.encode_text()
        else:
            tf = model.prompt()  # calls forward()
        print("Sanity check: text_features shape:", tuple(tf.shape))
        print("Sanity check: any NaN:", torch.isnan(tf).any().item())
        print("Sanity check: mean norm:", tf.norm(dim=1).mean().item())
        sim = (tf @ tf.t()).mean().item()
        print("Sanity check: mean cosine(similarity matrix):", sim)

    # --- Optimizer (only prompt params) ---
    lr = _to_float(cfg["train"].get("lr", 1e-4), default=1e-4)
    weight_decay = _to_float(cfg["train"].get("weight_decay", 0.0), default=0.0)

    optimizer = torch.optim.AdamW(
        model.prompt.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    epochs = int(cfg["train"]["epochs"])
    best_top1 = -1.0

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    best_path = "checkpoints/best.pt"

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        for step, (images, targets) in enumerate(pbar, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / step)

        # --- Eval ---
        metrics = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: top1={metrics['top1']:.4f}, top5={metrics['top5']:.4f}")

        # --- Save best prompt ctx ---
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            # Save only the learned context tokens for small checkpoint size
            torch.save({"ctx": model.prompt.ctx.detach().cpu()}, best_path)

    # --- Save summary ---
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
    main()