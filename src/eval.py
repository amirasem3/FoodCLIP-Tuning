"""
Evaluating utilities for classification.

"""


from __future__ import annotations
import torch

@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:, :k].any(dim=1).float().mean().item()
        res[f"top{k}"] = correct_k
    return res

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_top1 = []
    all_top5 = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        acc = accuracy_topk(logits, targets, topk=(1, 5))
        all_top1.append(acc["top1"])
        all_top5.append(acc["top5"])

    return {
        "top1": float(sum(all_top1) / len(all_top1)),
        "top5": float(sum(all_top5) / len(all_top5)),
    }
