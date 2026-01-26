"""
Generating report and comparison results for project reports





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
    all_class_features = []
    for name in classnames:
        texts = [t.format(name.replace("_", " ")) for t in TEMPLATES]
        tokenized = tokenizer(texts).to(device)
        text_feats = clip_model.encode_text(tokenized)
        text_feats = F.normalize(text_feats, dim=-1)
        class_feat = F.normalize(text_feats.mean(dim=0), dim=-1)
        all_class_features.append(class_feat)
    return torch.stack(all_class_features, dim=0)  # [K, D]


@torch.no_grad()
def eval_baseline(clip_model, tokenizer, loader, classnames, device):
    clip_model.eval()
    text_features = build_zeroshot_text_features(clip_model, tokenizer, classnames, device)
    logit_scale = clip_model.logit_scale.exp()

    K = len(classnames)
    confusion = torch.zeros((K, K), dtype=torch.int64)

    correct1 = 0
    correct5 = 0
    total = 0

    for images, targets in tqdm(loader, desc="Evaluating baseline"):
        images = images.to(device)
        targets = targets.to(device)

        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = logit_scale * image_features @ text_features.t()

        # top-1 / top-5
        top1 = logits.argmax(dim=1)
        top5 = logits.topk(5, dim=1).indices

        correct1 += (top1 == targets).sum().item()
        correct5 += (top5 == targets.view(-1, 1)).any(dim=1).sum().item()
        total += targets.size(0)

        for t, p in zip(targets, top1):
            confusion[t, p] += 1

    top1_acc = correct1 / total
    top5_acc = correct5 / total

    per_class_acc = (confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()).cpu().numpy()

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "confusion": confusion.cpu().numpy(),
        "per_class_acc": per_class_acc
    }


@torch.no_grad()
def eval_tuned(model, loader, device):
    model.eval()
    K = model.prompt.tokenized_prompts.size(0)
    confusion = torch.zeros((K, K), dtype=torch.int64)

    correct1 = 0
    correct5 = 0
    total = 0

    for images, targets in tqdm(loader, desc="Evaluating tuned"):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)

        top1 = logits.argmax(dim=1)
        top5 = logits.topk(5, dim=1).indices

        correct1 += (top1 == targets).sum().item()
        correct5 += (top5 == targets.view(-1, 1)).any(dim=1).sum().item()
        total += targets.size(0)

        for t, p in zip(targets, top1):
            confusion[t, p] += 1

    top1_acc = correct1 / total
    top5_acc = correct5 / total

    per_class_acc = (confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()).cpu().numpy()

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "confusion": confusion.cpu().numpy(),
        "per_class_acc": per_class_acc
    }


def plot_per_class_improvement(classnames, base_acc, tuned_acc, outpath):
    df = pd.DataFrame({
        "class": classnames,
        "baseline": base_acc,
        "tuned": tuned_acc,
        "improvement": tuned_acc - base_acc
    }).sort_values("improvement", ascending=False)

    plt.figure()
    plt.bar(df["class"], df["improvement"])
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy Improvement (Tuned - Baseline)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return df

def plot_per_class_accuracy(classnames, acc, title, outpath):
    df = pd.DataFrame({"class": classnames, "acc": acc}).sort_values("acc", ascending=False)

    plt.figure()
    plt.bar(df["class"], df["acc"])
    plt.xticks(rotation=90)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return df
def main(cfg_path="configs/default.yaml", tuned_ckpt="checkpoints/best.pt"):
    os.makedirs("results", exist_ok=True)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    # Load CLIP + preprocess
    model_name = cfg["model"]["clip_name"]
    pretrained = cfg["model"]["pretrained"]

    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    clip_model = clip_model.to(device)

    classnames = cfg["data"]["subset_classes"]

    # Data loaders
    _, test_loader = make_loaders(
        root=cfg["data"]["root"],
        subset_classes=classnames,
        train_transform=preprocess,
        eval_transform=preprocess,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    # Baseline evaluation
    base = eval_baseline(clip_model, tokenizer, test_loader, classnames, device)

    # Load tuned model and checkpoint (ctx only)
    tuned_model = PromptTunedCLIP(clip_model, tokenizer, classnames, n_ctx=cfg["model"]["n_ctx"]).to(device)

    ckpt = torch.load(tuned_ckpt, map_location="cpu")
    tuned_model.prompt.ctx.data.copy_(ckpt["ctx"])

    tuned = eval_tuned(tuned_model, test_loader, device)

    # Save main results
    main_df = pd.DataFrame([{
        "setting": "baseline",
        "top1": base["top1"],
        "top5": base["top5"]
    }, {
        "setting": "tuned",
        "top1": tuned["top1"],
        "top5": tuned["top5"]
    }])
    main_df["delta_top1_vs_baseline"] = main_df["top1"] - base["top1"]
    main_df["delta_top5_vs_baseline"] = main_df["top5"] - base["top5"]
    main_df.to_csv("results/main_results.csv", index=False)

    # Save confusion matrices
    np.save("results/confusion_baseline.npy", base["confusion"])
    np.save("results/confusion_tuned.npy", tuned["confusion"])

    # Save per-class results + plot
    per_class_df = plot_per_class_improvement(
        classnames,
        base["per_class_acc"],
        tuned["per_class_acc"],
        outpath="results/fig_perclass_improvement.png"
    )

    # BEFORE (baseline) plot
    plot_per_class_accuracy(
        classnames,
        base["per_class_acc"],
        title="Per-class Accuracy (Baseline / Before Optimization)",
        outpath="results/fig_perclass_baseline.png",
    )

    # AFTER (tuned) plot
    plot_per_class_accuracy(
        classnames,
        tuned["per_class_acc"],
        title="Per-class Accuracy (Prompt-Tuned / After Optimization)",
        outpath="results/fig_perclass_tuned.png",
    )



    per_class_df.to_csv("results/per_class.csv", index=False)

    # Save a run summary for your paper/reproducibility
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

    print("Saved results to results/:")
    print(" - main_results.csv")
    print(" - per_class.csv")
    print(" - fig_perclass_improvement.png")
    print(" - run_summary.json")
    print(" - confusion_baseline.npy / confusion_tuned.npy")


if __name__ == "__main__":
    main()


