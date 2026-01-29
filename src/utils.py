"""
Utility functions shared across the project.

This file includes small helper functions that are used in multiple places:
- set_seed: makes runs more reproducible by fixing random seeds
- get_device: chooses CPU or GPU device based on user preference + availability
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seeds for common libraries to improve reproducibility.

    Why this matters:
    Many operations in ML use randomness (data shuffling, weight init, etc.).
    By fixing seeds, you can reduce run-to-run variation and make experiments easier to compare.

    What gets seeded:
    - Python's built-in random
    - NumPy random generator
    - PyTorch CPU RNG
    - PyTorch CUDA RNG (all GPUs)

    Note:
    This improves reproducibility but does not guarantee perfect determinism on all GPUs/ops.
    """
    random.seed(seed)                # Python random module
    np.random.seed(seed)             # NumPy random module
    torch.manual_seed(seed)          # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed) # PyTorch CUDA RNG for all available GPUs


def get_device(device_str: str) -> torch.device:
    """
    Choose the torch.device to run on (CPU or CUDA GPU).

    Args:
        device_str: Usually "cuda" or "cpu" (comes from config).
            - If "cuda" is requested AND a CUDA GPU is available -> returns torch.device("cuda")
            - Otherwise -> returns torch.device("cpu")

    Returns:
        torch.device object to be used with .to(device)
    """
    # Use GPU only when requested AND available
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    # Fallback to CPU
    return torch.device("cpu")
