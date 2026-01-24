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