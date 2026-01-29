"""
Generating report and comparison results for project reports

This script compares:
- Zero-shot CLIP baseline performance (handcrafted prompt templates)
vs
- Prompt-tuned CLIP performance (soft prompt tokens learned during training)

It produces:
- Overall Top-1 / Top-5 results in results/main_results.csv
- Confusion matrices saved as .npy files
- Per-class accuracy CSV + plots for:
  - baseline accuracy
  - tuned accuracy
  - improvement (tuned - baseline)
- A compact JSON run summary for reproducibility (results/run_summary.json)
"""

from __future__ import annotations

import os, json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open_clip
from tqdm import tqdm

from src.utils import set_seed, get_device
from src.data import make_loaders
from src.prompts import TEMPLATES
from src.model import PromptTunedCLIP


@torch.no_grad()
def build_zeroshot_text_features(clip_model, tokenizer, classnames, device):
    """
    Build one text embedding per class using handcrafted prompt templates (zero-shot CLIP style).

    For each class:
    - Create multiple prompts using TEMPLATES (e.g., "a photo of {}")
    - Encode each prompt with CLIP text encoder
    - Normalize each prompt embedding
    - Average them to get one class embedding
    - Normalize the averaged embedding

    Returns:
        Tensor [K, D] where:
        K = number of classes
        D = CLIP embedding dimension
    """
    all_class_features = []
    for name in classnames:
        # Replace underscores so class names become natural language (e.g., "ice_cream" -> "ice cream")
        texts = [t.format(name.replace("_", " ")) for t in TEMPLATES]

        # Tokenize prompts for CLIP text encoder
        tokenized = tokenizer(texts).to(device)

        # Encode all prompt variants for this class -> [num_templates, D]
        text_feats = clip_model.encode_text(tokenized)

        # Normalize each prompt feature (unit-length)
        text_feats = F.normalize(text_feats, dim=-1)

        # Average prompt embeddings -> [D], then normalize again
        class_feat = F.normalize(text_feats.mean(dim=0), dim=-1)

        all_class_features.append(class_feat)

    # Stack all class features -> [K, D]
    return torch.stack(all_class_features, dim=0)  # [K, D]


@torch.no_grad()
def eval_baseline(clip_model, tokenizer, loader, classnames, device):
    """
    Evaluate the zero-shot CLIP baseline on a dataset loader.

    What is computed:
    - Top-1 accuracy (best prediction matches target)
    - Top-5 accuracy (target appears in top 5 predictions)
    - Confusion matrix [K, K] (rows=true, cols=predicted)
    - Per-class accuracy derived from confusion matrix

    Returns:
        dict with keys: top1, top5, confusion, per_class_acc
    """
    # Ensure model behaves in inference mode (no dropout, etc.)
    clip_model.eval()

    # Build text features for all classes once (faster than rebuilding per batch)
    text_features = build_zeroshot_text_features(clip_model, tokenizer, classnames, device)

    # CLIP uses a learned logit scale (temperature); exp ensures positive scaling
    logit_scale = clip_model.logit_scale.exp()

    # Number of classes in this subset
    K = len(classnames)

    # Confusion matrix counts: confusion[true_class, predicted_class]
    confusion = torch.zeros((K, K), dtype=torch.int64)

    # Running counters for accuracy
    correct1 = 0
    correct5 = 0
    total = 0

    # Iterate over evaluation batches
    for images, targets in tqdm(loader, desc="Evaluating baseline"):
        images = images.to(device)
        targets = targets.to(device)

        # Encode images with CLIP image encoder -> [B, D]
        image_features = clip_model.encode_image(images)

        # Normalize so dot product becomes cosine similarity
        image_features = F.normalize(image_features, dim=-1)

        # Compute similarity logits between image features and text features -> [B, K]
        logits = logit_scale * image_features @ text_features.t()

        # ----------------------------
        # Compute Top-1 / Top-5 stats
        # ----------------------------
        # Top-1 prediction: highest-scoring class index
        top1 = logits.argmax(dim=1)

        # Top-5 predictions: indices of 5 highest-scoring classes
        top5 = logits.topk(5, dim=1).indices

        # Count how many are correct
        correct1 += (top1 == targets).sum().item()
        correct5 += (top5 == targets.view(-1, 1)).any(dim=1).sum().item()
        total += targets.size(0)

        # ----------------------------
        # Update confusion matrix
        # ----------------------------
        # For each sample, increment confusion[true_label, predicted_label]
        for t, p in zip(targets, top1):
            confusion[t, p] += 1

    # Convert counts to overall accuracies
    top1_acc = correct1 / total
    top5_acc = correct5 / total

    # Per-class accuracy:
    # diag(confusion) = correct predictions per class
    # sum(confusion, dim=1) = total samples per true class
    per_class_acc = (
        confusion.diag().float()
        / confusion.sum(dim=1).clamp(min=1).float()
    ).cpu().numpy()

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "confusion": confusion.cpu().numpy(),
        "per_class_acc": per_class_acc
    }


@torch.no_grad()
def eval_tuned(model, loader, device):
    """
    Evaluate the prompt-tuned CLIP model (PromptTunedCLIP) on a dataset loader.

    This is similar to eval_baseline, but:
    - logits are produced directly by the tuned model(images)
    - the tuned model internally uses learned soft prompt tokens for text features

    Returns:
        dict with keys: top1, top5, confusion, per_class_acc
    """
    model.eval()

    # Number of classes (K) inferred from how many prompts were created
    K = model.prompt.tokenized_prompts.size(0)

    # Confusion matrix counts: confusion[true_class, predicted_class]
    confusion = torch.zeros((K, K), dtype=torch.int64)

    correct1 = 0
    correct5 = 0
    total = 0

    for images, targets in tqdm(loader, desc="Evaluating tuned"):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass produces logits [B, K] using image encoder + learned prompt text embeddings
        logits = model(images)

        # Top-1 prediction (argmax)
        top1 = logits.argmax(dim=1)

        # Top-5 predictions
        top5 = logits.topk(5, dim=1).indices

        # Update accuracy counters
        correct1 += (top1 == targets).sum().item()
        correct5 += (top5 == targets.view(-1, 1)).any(dim=1).sum().item()
        total += targets.size(0)

        # Update confusion matrix using top-1 predictions
        for t, p in zip(targets, top1):
            confusion[t, p] += 1

    top1_acc = correct1 / total
    top5_acc = correct5 / total

    # Per-class accuracy from confusion matrix
    per_class_acc = (
        confusion.diag().float()
        / confusion.sum(dim=1).clamp(min=1).float()
    ).cpu().numpy()

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "confusion": confusion.cpu().numpy(),
        "per_class_acc": per_class_acc
    }


def plot_per_class_improvement(classnames, base_acc, tuned_acc, outpath):
    """
    Create a bar plot of per-class accuracy improvement:
        improvement = tuned_acc - base_acc

    Also returns a DataFrame containing per-class baseline/tuned/improvement values,
    sorted by improvement (largest first).

    Saves:
        outpath (PNG figure)
    """
    # Build a table for per-class results
    df = pd.DataFrame({
        "class": classnames,
        "baseline": base_acc,
        "tuned": tuned_acc,
        "improvement": tuned_acc - base_acc
    }).sort_values("improvement", ascending=False)

    # Plot improvement bars (positive means tuned improved, negative means worse)
    plt.figure()
    plt.bar(df["class"], df["improvement"])
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy Improvement (Tuned - Baseline)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return df


def plot_per_class_accuracy(classnames, acc, title, outpath):
    """
    Create a bar plot of per-class accuracy (baseline or tuned).

    Produces a DataFrame sorted by accuracy (highest first) and saves a figure.

    Saves:
        outpath (PNG figure)
    """
    df = pd.DataFrame({"class": classnames, "acc": acc}).sort_values("acc", ascending=False)

    plt.figure()
    plt.bar(df["class"], df["acc"])
    plt.xticks(rotation=90)
    plt.ylim(0, 1.0)  # accuracies are in [0, 1]
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return df


def main(cfg_path="configs/default.yaml", tuned_ckpt="checkpoints/best.pt"):
    """
    Main entry point to generate evaluation reports and plots.

    Workflow:
    1) Create results/ folder
    2) Load config, set seed, pick device
    3) Load CLIP model + preprocessing transform
    4) Build Food101-subset test loader
    5) Evaluate baseline (zero-shot templates)
    6) Load tuned prompt vectors from checkpoint and evaluate tuned model
    7) Save CSVs, figures, confusion matrices, and JSON run summary
    """
    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)

    # Load experiment configuration
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Reproducibility + device selection (cpu/cuda)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    # ----------------------------
    # Load CLIP model and preprocess
    # ----------------------------
    model_name = cfg["model"]["clip_name"]
    pretrained = cfg["model"]["pretrained"]

    # open_clip provides both the model and the correct preprocessing transforms
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    # Move CLIP model to device (GPU/CPU)
    clip_model = clip_model.to(device)

    # Subset classes used in this project
    classnames = cfg["data"]["subset_classes"]

    # ----------------------------
    # Build data loaders
    # ----------------------------
    # We only need test_loader for evaluation
    _, test_loader = make_loaders(
        root=cfg["data"]["root"],
        subset_classes=classnames,
        train_transform=preprocess,
        eval_transform=preprocess,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    # ----------------------------
    # Baseline evaluation (zero-shot)
    # ----------------------------
    base = eval_baseline(clip_model, tokenizer, test_loader, classnames, device)

    # ----------------------------
    # Tuned evaluation (load learned ctx tokens)
    # ----------------------------
    # Build the tuned model wrapper (CLIP frozen + SoftPromptLearner)
    tuned_model = PromptTunedCLIP(
        clip_model,
        tokenizer,
        classnames,
        n_ctx=cfg["model"]["n_ctx"]
    ).to(device)

    # Load checkpoint containing learned context vectors (ctx)
    ckpt = torch.load(tuned_ckpt, map_location="cpu")

    # Copy learned context vectors into the prompt learner parameters
    # (This assumes ckpt["ctx"] has shape [n_ctx, D])
    tuned_model.prompt.ctx.data.copy_(ckpt["ctx"])

    # Evaluate tuned model
    tuned = eval_tuned(tuned_model, test_loader, device)

    # ----------------------------
    # Save overall results as a CSV
    # ----------------------------
    main_df = pd.DataFrame([{
        "setting": "baseline",
        "top1": base["top1"],
        "top5": base["top5"]
    }, {
        "setting": "tuned",
        "top1": tuned["top1"],
        "top5": tuned["top5"]
    }])

    # Add improvement columns relative to baseline (delta)
    main_df["delta_top1_vs_baseline"] = main_df["top1"] - base["top1"]
    main_df["delta_top5_vs_baseline"] = main_df["top5"] - base["top5"]

    main_df.to_csv("results/main_results.csv", index=False)

    # ----------------------------
    # Save confusion matrices
    # ----------------------------
    # Stored as numpy arrays for easy plotting later (e.g., in a notebook)
    np.save("results/confusion_baseline.npy", base["confusion"])
    np.save("results/confusion_tuned.npy", tuned["confusion"])

    # ----------------------------
    # Save per-class results + plots
    # ----------------------------
    # Improvement plot (tuned - baseline) and DataFrame
    per_class_df = plot_per_class_improvement(
        classnames,
        base["per_class_acc"],
        tuned["per_class_acc"],
        outpath="results/fig_perclass_improvement.png"
    )

    # Baseline per-class accuracy plot (before tuning)
    plot_per_class_accuracy(
        classnames,
        base["per_class_acc"],
        title="Per-class Accuracy (Baseline / Before Optimization)",
        outpath="results/fig_perclass_baseline.png",
    )

    # Tuned per-class accuracy plot (after tuning)
    plot_per_class_accuracy(
        classnames,
        tuned["per_class_acc"],
        title="Per-class Accuracy (Prompt-Tuned / After Optimization)",
        outpath="results/fig_perclass_tuned.png",
    )

    # Save per-class table as CSV
    per_class_df.to_csv("results/per_class.csv", index=False)

    # ----------------------------
    # Save a compact run summary (useful for paper + reproducibility)
    # ----------------------------
    summary = {
        "clip": f"{model_name}/{pretrained}",
        "subset_k": len(classnames),
        "n_ctx": cfg["model"]["n_ctx"],
        "templates_used": len(TEMPLATES),
        "seed": cfg["seed"],
        "baseline_top1": base["top1"],
        "baseline_top5": base["top5"],
        "tuned_top1": tuned["top1"],
        "tuned_top5": tuned["top5"],
        "delta_top1": tuned["top1"] - base["top1"],
        "delta_top5": tuned["top5"] - base["top5"],
    }

    with open("results/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print what we produced so the user can quickly find outputs
    print("Saved results to results/:")
    print(" - main_results.csv")
    print(" - per_class.csv")
    print(" - fig_perclass_improvement.png")
    print(" - run_summary.json")
    print(" - confusion_baseline.npy / confusion_tuned.npy")


if __name__ == "__main__":
    # Allows this script to be run directly:
    # python src/report.py   (or wherever this file lives)
    main()
