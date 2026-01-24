"""
Manages the data utilities of the project

"""



from __future__ import annotations
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Food101

class SubsetFood101(Dataset):
    """
    Wraps Food101 and filters to a subset of classes.
    Remaps labels to [0..K-1].
    """
    def __init__(self, root: str, split: str, subset_classes: List[str], transform=None, download=True):
        super().__init__()
        self.base = Food101(root=root, split=split, transform=transform, download=download)

        # Food101 has classes list and class_to_idx
        class_to_idx = {c: i for i, c in enumerate(self.base.classes)}
        wanted_base_idxs = [class_to_idx[c] for c in subset_classes]

        # build index list of examples that belong to wanted classes
        self.keep_indices = []
        for i in range(len(self.base)):
            _, y = self.base[i]
            if y in wanted_base_idxs:
                self.keep_indices.append(i)

        # remap labels
        self.subset_classes = subset_classes
        self.baseidx_to_new = {class_to_idx[c]: j for j, c in enumerate(subset_classes)}

    def __len__(self):
        return len(self.keep_indices)

    def __getitem__(self, idx):
        base_i = self.keep_indices[idx]
        x, y_base = self.base[base_i]
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
    train_ds = SubsetFood101(root=root, split="train", subset_classes=subset_classes, transform=train_transform, download=True)
    test_ds  = SubsetFood101(root=root, split="test",  subset_classes=subset_classes, transform=eval_transform,  download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader