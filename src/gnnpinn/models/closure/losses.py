"""Regularization losses for sparse closure discovery."""

from __future__ import annotations

from typing import Any


def l1_sparsity(coefficients: Any, weight: float = 1.0) -> Any:
    """Return weighted L1 sparsity penalty."""

    import torch

    if weight < 0:
        raise ValueError("weight must be non-negative")
    return torch.as_tensor(weight, dtype=coefficients.dtype, device=coefficients.device) * coefficients.abs().sum()

