"""Explicit class selection without assuming a segmentation layout."""

import torch
from torch import Tensor


def select_class_uncertainty(std: Tensor, classes: Tensor, *, class_dim: int = 1) -> Tensor:
    """Gather class-wise standard deviations, removing only the class axis.

    ``classes`` must be int64 indices with the shape of ``std`` minus class_dim.
    Labels may come from a deterministic prediction or the MC mean. For a
    single-channel binary probability, use its standard deviation directly.
    """
    if not std.is_floating_point() or std.ndim == 0:
        raise ValueError("std must be a floating-point Tensor with a class axis")
    if not -std.ndim <= class_dim < std.ndim:
        raise ValueError("class_dim is out of range")
    class_dim %= std.ndim
    expected = std.shape[:class_dim] + std.shape[class_dim + 1 :]
    if classes.shape != expected or classes.dtype != torch.int64:
        raise ValueError("classes must be int64 with the prediction shape minus the class axis")
    if classes.device != std.device:
        raise ValueError("classes and std must be on the same device")
    if ((classes < 0) | (classes >= std.shape[class_dim])).any():
        raise ValueError("Class indices are out of range")
    return std.gather(class_dim, classes.unsqueeze(class_dim)).squeeze(class_dim)
