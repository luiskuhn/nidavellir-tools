"""Repeated prediction with numerically stable, streaming output statistics."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .dropout import mc_dropout_mode


@dataclass(frozen=True)
class MCPrediction:
    """Output-shaped statistics; optional samples have a leading MC dimension."""

    mean: Tensor
    variance: Tensor
    std: Tensor
    num_samples: int
    correction: int
    samples: Tensor | None = None


def mc_dropout_predict(
    model: nn.Module,
    inputs: Any,
    *,
    num_samples: int = 30,
    output_transform: Callable[[Any], Tensor] | None = None,
    predict_fn: Callable[[nn.Module, Any], Any] | None = None,
    correction: int = 1,
    return_samples: bool = False,
) -> MCPrediction:
    """Sample a model with dropout active and all other modules in eval mode.

    By default call ``model(inputs)``. A callback can adapt multiple inputs or
    reconstruct a whole tiled prediction per pass. Transform outputs explicitly
    (e.g. softmax/sigmoid for probabilities); no activation is inferred.

    Statistics preserve output shape/device, accumulating half precision in
    float32 and retaining float64. Samples, when requested, use that same dtype.
    No gradients are recorded. RNG state advances naturally; seed once outside
    this function for repeatability, never once per pass. Memory is independent
    of num_samples unless return_samples=True. Module flags are restored even
    when a callback raises. Callbacks must not change model modes or weights.
    """
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 2:
        raise ValueError("num_samples must be an integer >= 2")
    if isinstance(correction, bool) or not isinstance(correction, int) or correction not in (0, 1):
        raise ValueError("correction must be 0 (population) or 1 (sample)")
    stored = [] if return_samples else None
    mean = m2 = None
    signature = None
    with mc_dropout_mode(model), torch.inference_mode():
        for count in range(1, num_samples + 1):
            output = model(inputs) if predict_fn is None else predict_fn(model, inputs)
            value = output if output_transform is None else output_transform(output)
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError("Each transformed prediction must be a floating-point Tensor")
            current = (value.shape, value.device, value.dtype)
            if signature is not None and current != signature:
                raise ValueError("Prediction shape, device and dtype must remain constant")
            signature = current
            if not torch.isfinite(value).all():
                raise ValueError("Predictions must contain only finite values")
            dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
            value = value.detach().to(dtype)
            if stored is not None:
                stored.append(value.clone())
            if mean is None:
                mean = value.clone()
                m2 = torch.zeros_like(mean)
            else:
                delta = value - mean
                mean = mean + delta / count
                m2 = m2 + delta * (value - mean)
        variance = (m2 / (num_samples - correction)).clamp_min(0)
        return MCPrediction(
            mean=mean,
            variance=variance,
            std=variance.sqrt(),
            num_samples=num_samples,
            correction=correction,
            samples=torch.stack(stored) if stored is not None else None,
        )
