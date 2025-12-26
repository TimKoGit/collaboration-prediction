"""Utilities for PyTorch compatibility and serialization."""

import contextlib
import functools
import logging
from typing import Generator

import torch

logger = logging.getLogger(__name__)

# Store original torch.load to avoid global patching
_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False.

    This function wraps the original torch.load and sets weights_only=False
    if not explicitly provided. This is necessary because PyTorch 2.6+ changed
    the default to weights_only=True, but some datasets (like OGB) contain
    objects that aren't in the safe globals list.

    Args:
        *args: Positional arguments passed to torch.load
        **kwargs: Keyword arguments passed to torch.load

    Returns:
        Loaded object from torch.load
    """
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


@contextlib.contextmanager
def patch_torch_load() -> Generator[None, None, None]:
    """Context manager to temporarily patch torch.load.

    This is needed because PyTorch 2.6+ changed the default to weights_only=True,
    but some trusted datasets contain objects that aren't in the safe globals list.

    Yields:
        None: Context manager that patches torch.load during execution
    """
    torch.load = _patched_torch_load
    try:
        yield
    finally:
        torch.load = _original_torch_load
