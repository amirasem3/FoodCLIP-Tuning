"""
Zero-shot CLIP baseline evaluation on a Food-101 **subset** using handcrafted prompt templates.

What this script does (high-level):
1) Load experiment configuration from configs/default.yaml
2) Load a pretrained CLIP model (frozen; no training)
3) Build a test dataloader for a subset of Food-101 classes
4) Build ONE text embedding per class by averaging multiple prompt templates
5) Encode each test image with CLIP image encoder
6) Compute image-text similarity scores (logits)
7) Measure Top-1 / Top-5 classification accuracy
8) Save summary results to results/baseline.json

This is the baseline that prompt-tuning methods should beat.
"""

from __future__ import annotations

import json
import yaml
import open_clip

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import set_seed, get_device
from src.data import make_loaders
from src.prompts import TEMPLATES
from src.eval import accuracy_topk


def build_zeroshot_text_features(clip_model, tokenizer, classnames, device):
    """
    Build zero-shot text embeddings for each class using multiple prompt templates.

    Idea:
    - For each class name (e.g., "ice_cream"):
        - Create multiple text prompts using TEMPLATES:
          e.g., "a photo of ice cream", "this is ice cream", etc.
        - Encode each prompt with CLIP's text encoder
        - Normalize each prompt embedding (unit-length)
        - Average all prompt embeddings to get a single class embedding
        - Normalize the averaged vector again

    Returns:
        Tensor of shape [K, D]
        K = number of classes
        D = text embedding dimension of CLIP model
    """
    all_class_features = []

    for name in classnames:
        # Convert class name to a more natural phrase for prompts
        # e.g., "ice_cream" -> "ice cream"
        class_text = name.replace("_", " ")

        # Build multiple prompts for the same class
        texts = [t.format(class_text) for t in TEMPLATES]

        # Tokenize prompts into CLIP's expected token format
        tokenized = tokenizer(texts).to(device)

        # Encode prompts -> [num_templates, D]
        with torch.no_grad():
            text_feats = clip_model.encode_text(tokenized)

        # Normalize each prompt embedding to unit length
        text_feats = F.normalize(text_feats, dim=-1)

        # Average across templates -> [D]
        class_feat = text_feats.mean(dim=0)

        # Normalize averaged class embedding to unit length
        class_feat = F.normalize(class_feat, dim=-1)

        all_class_features.append(class_feat)

    # Stack class embeddings -> [K, D]
    return torch.stack(all_class_features, dim=0)


def main(cfg_path="configs/default.yaml"):
    """
    Main evaluation entry point.

    Steps:
    - Load YAML config
    - Set random seeds for reproducibility
    - Select device (cpu/cuda)
    - Load pretrained CLIP model and preprocessing transforms
    - Create test dataloader for the subset of classes
    - Build averaged text embeddings per class
    - For each test batch:
        - Encode images
        - Compute similarities to class text embeddings
        - Compute top-1/top-5 accuracy
    - Save summary metrics to results/baseline.json
    """
    # ----------------------------
    # 1) Load experiment config
    # ----------------------------
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ----------------------------
    # 2) Reproducibility and device
    # ----------------------------
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    # ----------------------------
    # 3) Load pretrained CLIP model
    # ----------------------------
    model_name = cfg["model"]["clip_name"]
    pretrained = cfg["model"]["pretrained"]

    # open_clip provides model + preprocessing transforms aligned with that model
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    # Put model on device and set eval mode (no dropout, etc.)
    clip_model = clip_model.to(device).eval()

    # We never train in this baseline; gradients aren't needed
    # (We'll still wrap encoding blocks with torch.no_grad() below for clarity.)

    # ----------------------------
    # 4) Create Food-101 subset loaders
    # ----------------------------
    subset_classes = cfg["data"]["subset_classes"]

    # make_loaders is expected to:
    # - filter dataset to only the subset classes
    # - create train/test loaders
    # Note: We only need the test loader for evaluation.
    _, test_loader = make_loaders(
        root=cfg["data"]["root"],
        subset_classes=subset_classes,
        train_transform=preprocess,
        eval_transform=preprocess,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    # ----------------------------
    # 5) Build text features for each class
    # ----------------------------
    # text_features: [K, D] where K = number of subset classes
    text_features = build_zeroshot_text_features(
        clip_model, tokenizer, subset_classes, device
    )

    # CLIP uses a learned logit scale (temperature). We exponentiate it per CLIP design.
    # This controls how "peaky" the similarity distribution is.
    logit_scale = clip_model.logit_scale.exp()

    # ----------------------------
    # 6) Evaluate on the test set
    # ----------------------------
    top1s, top5s = [], []

    for images, targets in tqdm(test_loader, desc="Baseline eval"):
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            # Encode images -> [B, D]
            image_features = clip_model.encode_image(images)

            # Normalize to unit length so dot product becomes cosine similarity
            image_features = F.normalize(image_features, dim=-1)

            # Similarity logits: [B, K]
            # (cosine similarity * logit_scale)
            logits = logit_scale * (image_features @ text_features.t())

        # Compute Top-1 / Top-5 accuracy
        acc = accuracy_topk(logits, targets, topk=(1, 5))
        top1s.append(acc["top1"])
        top5s.append(acc["top5"])

    # ----------------------------
    # 7) Summarize + save results
    # ----------------------------
    out = {
        "clip": f"{model_name}/{pretrained}",
        "subset_k": len(subset_classes),
        "num_templates": len(TEMPLATES),
        # Averaging over batches (assumes accuracy_topk returns per-batch scalar accuracy)
        "top1": float(sum(top1s) / len(top1s)),
        "top5": float(sum(top5s) / len(top5s)),
    }

    # Save metrics to JSON for easy comparison with tuned methods
    with open("results/baseline.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Saved results/baseline.json:", out)


if __name__ == "__main__":
    # Allows running this file directly:
    # python path/to/this_script.py
    main()
