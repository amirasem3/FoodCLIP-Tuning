"""
This file will handle the Zero-shot CLIP baseline evaluation on the Food101 **subset** using handcrafted prompt template
in below steps:
  1- Loading project configuration from configs/default.yaml
  2- Loading a pretrained CLIP model (frozen; no training)
  3- Leading the subset of Food-101 subset test data loader
  4- Using averaging multiple prompt templates for building one text embedding per class
  5- Encoding each test image
  6- Computing similarity to class text embeddings
  7- Measuring Top-1/Top-5 accuracy
  8- Save summary results in location of results/baseline.json
In summary, this file will provide the **baseline** which prompt-tuning improvements.
"""

from __future__ import annotations
import json
import yaml
import open_clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import set_seed, get_device
#TODO: define make_loaders in data
from src.prompts import TEMPLATES
#TODO: define the accuracy_topk in eval

def build_zeroshot_text_features(clip_model, tokenizer, classnames, device):
    # average text features across templates per class
    all_class_features = []
    for name in classnames:
        texts = [t.format(name.replace("_"," ")) for t in TEMPLATES]
        tokenized = tokenizer(texts).to(device)
        text_feats = clip_model.encode_text(tokenized)
        text_feats = F.normalize(text_feats, dim=-1)
        class_feat = text_feats.mean(dim=0)
        class_feat = F.normalize(class_feat, dim=-1)
        all_class_features.append(class_feat)
    return torch.stack(all_class_features, dim=0)  # [K, D]

def main(cfg_path="configs/default.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)


    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    model_name = cfg["model"]["clip_name"]
    pretrained = cfg["model"]["pretrained"]
    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    clip_model = clip_model.to(device).eval()

    subset_classes = cfg["data"]["subset_classes"]
    _, test_loader = make_loaders(
        root=cfg["data"]["root"],
        subset_classes=subset_classes,
        train_transform=preprocess,
        eval_transform=preprocess,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    text_features = build_zeroshot_text_features(clip_model, tokenizer, subset_classes, device)  # [K, D]
    logit_scale = clip_model.logit_scale.exp()

    top1s, top5s = [], []
    for images, targets in tqdm(test_loader, desc="Baseline eval"):
        images = images.to(device)
        targets = targets.to(device)

        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = logit_scale * image_features @ text_features.t()

        acc = accuracy_topk(logits, targets, topk=(1,5))
        top1s.append(acc["top1"])
        top5s.append(acc["top5"])

    out = {
        "clip": f"{model_name}/{pretrained}",
        "subset_k": len(subset_classes),
        "num_templates": len(TEMPLATES),
        "top1": float(sum(top1s)/len(top1s)),
        "top5": float(sum(top5s)/len(top5s)),
    }

    with open("results/baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results/baseline.json:", out)

if __name__ == "__main__":
    main()