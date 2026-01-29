"""
Evaluating utilities for classification.

This module provides:
- accuracy_topk: compute Top-K accuracies from model logits and true labels
- evaluate: run a model over a dataloader and report average Top-1 / Top-5 accuracy
"""

from __future__ import annotations
import torch


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    """
    Compute Top-K accuracy for one batch.

    Args:
        logits: Tensor of shape [B, C]
            - B = batch size
            - C = number of classes
            - values are raw model scores (higher means more confident)
        targets: Tensor of shape [B]
            - true class index for each example (0..C-1)
        topk: tuple of K values (e.g., (1, 5)) to compute Top-K accuracy for

    Returns:
        dict like {"top1": 0.83, "top5": 0.97} (values are batch-average accuracies)
    """
    # We need the largest K because we will compute top-1, top-5, etc.
    maxk = max(topk)

    # Get the indices of the top `maxk` predicted classes for each sample.
    # pred has shape [B, maxk], containing class indices sorted by score (highest first).
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)

    # Compare predictions with the true labels.
    # targets.view(-1, 1) makes targets shape [B, 1] so it can be compared to [B, maxk].
    # correct has shape [B, maxk] with True where prediction matches target.
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))

    res = {}
    for k in topk:
        # For Top-k:
        # - Look at the first k predicted classes (correct[:, :k])
        # - Check if ANY of them equals the target (any(dim=1))
        # - Convert to float (True->1.0, False->0.0)
        # - Average across the batch (mean)
        correct_k = correct[:, :k].any(dim=1).float().mean().item()
        res[f"top{k}"] = correct_k

    return res


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate a classification model over an entire dataset (provided by `loader`).

    Args:
        model: a callable model that maps images -> logits of shape [B, C]
        loader: DataLoader yielding (images, targets) batches
        device: torch device (e.g., "cuda" or "cpu")

    Returns:
        dict like {"top1": ..., "top5": ...}
        Note: This averages per-batch accuracies. (So each batch contributes equally.)
    """
    # Put model in evaluation mode (disables dropout, uses running stats for batchnorm, etc.)
    model.eval()

    # Store per-batch accuracies so we can average them at the end
    all_top1 = []
    all_top5 = []

    # Iterate over batches from the dataloader
    for images, targets in loader:
        # Move data to the correct device (GPU/CPU)
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass: produce logits (raw scores) for each class
        logits = model(images)

        # Compute Top-1 and Top-5 accuracy for this batch
        acc = accuracy_topk(logits, targets, topk=(1, 5))

        # Save batch results
        all_top1.append(acc["top1"])
        all_top5.append(acc["top5"])

    # Average across batches to get final metrics
    return {
        "top1": float(sum(all_top1) / len(all_top1)),
        "top5": float(sum(all_top5) / len(all_top5)),
    }
