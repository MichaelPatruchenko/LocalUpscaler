"""Load model architectures from weight files using spandrel."""

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def load_model_from_file(path: Path, device: str = "cpu"):
    """Load a model from a weight file, auto-detecting architecture.

    Uses spandrel to infer the correct architecture from the state dict.
    Returns the model moved to the specified device, in eval mode.
    """
    import spandrel

    log.info(f"Loading model from {path} onto {device}")
    model_descriptor = spandrel.ModelLoader(device=torch.device(device)).load_from_file(path)
    model = model_descriptor.model
    model.eval()
    log.info(f"Loaded {model_descriptor.architecture} (scale={getattr(model_descriptor, 'scale', '?')})")
    return model
