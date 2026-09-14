"""Temporary dropout-only evaluation mode."""

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn

_DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)


@contextmanager
def mc_dropout_mode(model: nn.Module) -> Iterator[nn.Module]:
    """Enable standard dropout only, restoring every module flag on exit.

    Rates and weights are unchanged. Functional dropout and custom stochastic
    layers are not activated. Do not share this model across concurrent calls.
    This context controls module modes only, not gradient recording.
    """
    modules = list(model.modules())
    dropouts = [module for module in modules if isinstance(module, _DROPOUT_TYPES)]
    if not any(0 < module.p < 1 for module in dropouts):
        raise ValueError("MC dropout requires a supported dropout layer with 0 < p < 1")
    states = [module.training for module in modules]
    try:
        model.eval()
        for module in dropouts:
            module.train()
        yield model
    finally:
        # Recursive train() calls would overwrite deliberately mixed child modes.
        for module, training in zip(modules, states, strict=True):
            module.training = training
