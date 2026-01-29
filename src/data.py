"""
Manages the data utilities of the project.

This module provides:
- A dataset wrapper (SubsetFood101) that filters Food101 to a user-defined subset of classes
- A helper (make_loaders) to build train/test DataLoaders for that subset

Key idea:
Food101 labels are originally in the range [0..100]. If we select only K classes, we
remap their labels to [0..K-1] so training/evaluation code can treat it as a normal K-class task.
"""

from __future__ import annotations

from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Food101


class SubsetFood101(Dataset):
    """
    Wraps torchvision.datasets.Food101 and filters it to a subset of classes.

    What it does:
    - Loads the Food101 dataset (train or test split)
    - Keeps only examples whose label is in `subset_classes`
    - Remaps the original Food101 class indices into a compact range [0..K-1]

    Example:
    If subset_classes = ["pizza", "sushi", "ice_cream"]
    then labels become:
        pizza -> 0
        sushi -> 1
        ice_cream -> 2
    """

    def __init__(
            self,
            root: str,
            split: str,
            subset_classes: List[str],
            transform=None,
            download: bool = True,
    ):
        super().__init__()

        # Load the full Food101 dataset split ("train" or "test")
        # transform is applied inside Food101 when retrieving samples
        self.base = Food101(
            root=root,
            split=split,
            transform=transform,
            download=download,
        )

        # Food101 exposes:
        #   - self.base.classes: list of class names (length 101)
        #   - self.base.class_to_idx: mapping {class_name: class_index}
        # We'll create a class_to_idx mapping explicitly for readability.
        class_to_idx = {c: i for i, c in enumerate(self.base.classes)}

        # Convert subset class names into their original Food101 integer labels
        # (these are the labels the base dataset returns)
        wanted_base_idxs = [class_to_idx[c] for c in subset_classes]

        # Build a list of indices in the base dataset that belong to the wanted classes.
        # After this, our dataset length becomes len(self.keep_indices), not len(self.base).
        #
        # Note: This loop calls self.base[i], so it will apply transforms and load images.
        # It's correct, but potentially slow. (If this becomes a bottleneck, we can speed it up.)
        self.keep_indices: List[int] = []
        for i in range(len(self.base)):
            _, y = self.base[i]  # y is the original Food101 label (0..100)
            if y in wanted_base_idxs:
                self.keep_indices.append(i)

        # Store the subset class order exactly as provided (important for consistent mapping)
        self.subset_classes = subset_classes

        # Build a mapping from original Food101 labels -> new compact labels [0..K-1]
        # Example: base label 54 ("pizza") -> new label 0
        self.baseidx_to_new = {class_to_idx[c]: j for j, c in enumerate(subset_classes)}

    def __len__(self) -> int:
        # Dataset length is the number of kept examples (not the full Food101 size)
        return len(self.keep_indices)

    def __getitem__(self, idx: int):
        """
        Returns:
            x: transformed image tensor (or PIL image depending on transform)
            y: remapped label in [0..K-1]
        """
        # Map the requested idx (0..len-1) to an index in the original Food101 dataset
        base_i = self.keep_indices[idx]

        # Retrieve the sample from the base dataset (this applies `transform`)
        x, y_base = self.base[base_i]

        # Remap original Food101 label (0..100) to subset label (0..K-1)
        y = self.baseidx_to_new[int(y_base)]

        return x, y


def make_loaders(
        root: str,
        subset_classes: List[str],
        train_transform,
        eval_transform,
        batch_size: int,
        num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to create train/test DataLoaders for a Food101 subset.

    Args:
        root: dataset root directory
        subset_classes: list of class names to keep (defines label mapping order)
        train_transform: preprocessing/augmentation for training samples
        eval_transform: preprocessing for evaluation samples
        batch_size: batch size for both loaders
        num_workers: DataLoader worker processes

    Returns:
        train_loader, test_loader
    """
    # Build subset datasets for each split
    train_ds = SubsetFood101(
        root=root,
        split="train",
        subset_classes=subset_classes,
        transform=train_transform,
        download=True,
    )
    test_ds = SubsetFood101(
        root=root,
        split="test",
        subset_classes=subset_classes,
        transform=eval_transform,
        download=True,
    )

    # Create loaders
    # - shuffle=True for training
    # - shuffle=False for evaluation
    # - pin_memory=True helps speed up host->GPU transfers when using CUDA
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
