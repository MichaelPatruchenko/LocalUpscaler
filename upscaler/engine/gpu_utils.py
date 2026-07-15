"""GPU memory safety utilities: auto tile sizing, OOM retry, CPU fallback."""

import gc
import logging
import math
from typing import Callable, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

# Overhead factor accounts for intermediate tensors, attention maps, etc.
_OVERHEAD_FACTOR = 4.0
# Minimum / maximum tile dimensions
_MIN_TILE = 64
_MAX_TILE = 1024
# Round tile size down to nearest multiple of this (alignment helps GPU perf)
_TILE_ALIGN = 64


def estimate_tile_size(
    device: str,
    model_memory_mb: int,
    scale: int = 1,
    channels: int = 3,
) -> int:
    """Estimate the largest safe tile size for the current GPU state.

    Returns a tile dimension (square) that should fit in available VRAM,
    accounting for model weights already loaded and processing overhead.
    Falls back to a conservative default if VRAM cannot be queried.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        # CPU has no VRAM limit — use a generous default
        return _MAX_TILE

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        log.warning("Could not query CUDA memory, using conservative tile size")
        return 256

    # Available memory for tile processing (subtract model weight estimate)
    model_bytes = model_memory_mb * 1024 * 1024
    available = max(free_bytes - model_bytes, 0)

    # Each pixel in a tile needs:
    #   input: channels × 4 bytes (float32)
    #   output: channels × scale² × 4 bytes
    #   intermediate: overhead factor covers attention maps, gradients buffer, etc.
    # With FP16 the actual usage is ~half, but we budget for FP32 as safety margin.
    bytes_per_pixel = channels * 4 * (1 + scale * scale) * _OVERHEAD_FACTOR

    if bytes_per_pixel <= 0:
        return 256

    max_pixels = available / bytes_per_pixel
    tile = int(math.sqrt(max_pixels))

    # Align and clamp
    tile = (tile // _TILE_ALIGN) * _TILE_ALIGN
    tile = max(_MIN_TILE, min(tile, _MAX_TILE))

    log.info(
        f"VRAM estimate: {free_bytes / 1e9:.2f} GB free, "
        f"model ~{model_memory_mb} MB, "
        f"estimated safe tile: {tile}"
    )
    return tile


class _FP16CorruptionError(Exception):
    """Raised when FP16 inference produces NaN/inf or garbage values."""
    pass


# Tile output validation thresholds.
# Image model outputs should be roughly in [0, 1]. FP16 corruption can produce
# values far outside this range, or chaotic noise with very high variance.
_VALID_MIN = -1.0
_VALID_MAX = 2.0


def _validate_tile_output(result: np.ndarray) -> bool:
    """Check tile output for FP16 corruption.

    Returns True if the output looks valid, False if it's likely corrupt.
    Detects: NaN, inf, values far outside image range, chaotic noise.
    """
    if not np.isfinite(result).all():
        return False
    if result.min() < _VALID_MIN or result.max() > _VALID_MAX:
        return False
    return True


def _clear_gpu_cache():
    """Clear PyTorch CUDA cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_tile_process(
    image: np.ndarray,
    scale: int,
    device: str,
    model: torch.nn.Module,
    tile_size_hint: int = 512,
    overlap: int = 32,
    model_memory_mb: int = 0,
    max_retries: int = 4,
    forward_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Process image with tiles, auto-sizing, FP16, OOM retry, and CPU fallback.

    Builds its own forward function with FP16 autocast on CUDA.
    If ``forward_fn`` is provided it is used instead (no autocast wrapping).

    Args:
        image: Input image as float32 numpy array.
        scale: Upscale factor (1 for denoisers).
        device: Current device string ('cuda', 'cuda:0', 'cpu', etc.).
        model: The torch model (needed for CPU fallback — will be moved to CPU).
        tile_size_hint: User-requested tile size (will be reduced if VRAM is tight).
        overlap: Tile overlap in pixels.
        model_memory_mb: Approximate model weight size in MB.
        max_retries: Maximum number of OOM retries before CPU fallback.
        forward_fn: Optional custom forward function (overrides built-in).

    Returns:
        Processed image as float32 numpy array.
    """
    from upscaler.plugins.upscalers.real_esrgan import tile_process

    is_cuda = device.startswith("cuda") if isinstance(device, str) else False

    # Pick initial tile size: minimum of user hint and VRAM-based estimate
    if is_cuda:
        estimated = estimate_tile_size(device, model_memory_mb, scale)
        current_tile = min(tile_size_hint, estimated)
        # Ensure alignment
        current_tile = max((current_tile // _TILE_ALIGN) * _TILE_ALIGN, _MIN_TILE)
    else:
        current_tile = tile_size_hint

    # Track whether FP16 produced corrupt output so we skip it on retry
    use_fp16 = is_cuda

    def _make_forward(dev: str, fp16: bool = False):
        """Create a forward function, optionally with FP16 autocast for CUDA."""
        def _forward(tile: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(dev)
                if fp16 and dev.startswith("cuda"):
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = model(t)
                    out = out.float()  # always convert back to FP32
                else:
                    out = model(t)
                result = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
                # NaN/inf is always genuine corruption (FP16 attention overflow
                # or a real failure) — retry in FP32 / fall back to CPU.
                if not np.isfinite(result).all():
                    raise _FP16CorruptionError()
                # Finite-but-out-of-range under FP16 may be overflow garbage, so
                # retry in FP32. Under FP32/CPU the model's output is trustworthy:
                # an image model that overshoots [0,1] is normal — clamp to the
                # image range (as spandrel's descriptor does) instead of treating
                # it as a fatal error.
                if (fp16 and dev.startswith("cuda")
                        and (result.min() < _VALID_MIN or result.max() > _VALID_MAX)):
                    raise _FP16CorruptionError()
                return np.clip(result, 0.0, 1.0)
        return _forward

    # Try on current device with shrinking tiles
    for attempt in range(max_retries):
        forward = _make_forward(device, fp16=use_fp16)
        try:
            log.info(
                f"Tile process attempt {attempt + 1}: device={device}, "
                f"tile={current_tile}, fp16={use_fp16}"
            )
            result = tile_process(image, forward, scale, current_tile, overlap)
            # Final validation of the blended result
            if not _validate_tile_output(result):
                raise _FP16CorruptionError()
            return result
        except _FP16CorruptionError:
            log.warning(
                "FP16 produced corrupt output (NaN/inf from attention overflow), "
                "retrying in FP32"
            )
            use_fp16 = False
            _clear_gpu_cache()
            # Don't reduce tile size — FP32 uses more memory but produces
            # correct results. If it then OOMs, the next iteration handles it.
            continue
        except torch.cuda.OutOfMemoryError:
            _clear_gpu_cache()
            current_tile = max(current_tile // 2, _MIN_TILE)
            current_tile = (current_tile // _TILE_ALIGN) * _TILE_ALIGN
            current_tile = max(current_tile, _MIN_TILE)
            log.warning(f"OOM on attempt {attempt + 1}, reducing tile to {current_tile}")
            if current_tile <= _MIN_TILE and attempt >= 1:
                # Already at minimum tile — break to CPU fallback
                break

    # CPU fallback
    log.warning("GPU retries exhausted, falling back to CPU")
    _clear_gpu_cache()
    model.cpu()
    forward_cpu = _make_forward("cpu", fp16=False)
    # On CPU, use larger tiles (no VRAM constraint)
    result = tile_process(image, forward_cpu, scale, tile_size_hint, overlap)
    # Move model back to original device for potential next use
    if is_cuda:
        try:
            model.to(device)
        except Exception:
            pass  # If we can't move back, leave on CPU
    return result


def safe_forward(
    tensor: torch.Tensor,
    model: torch.nn.Module,
    device: str,
) -> torch.Tensor:
    """Run model forward pass with FP16 autocast, OOM catch, and CPU fallback.

    Intended for colorizers and other plugins that process the whole image
    at a fixed resolution (no tiling needed).

    If FP16 produces corrupt output (NaN/inf from attention overflow),
    automatically retries in FP32 before falling back to CPU.

    Args:
        tensor: Input tensor already on the target device, shape (N, C, H, W).
        model: The torch model.
        device: Current device string.

    Returns:
        Output tensor (on CPU, float32).
    """
    is_cuda = device.startswith("cuda") if isinstance(device, str) else False

    # Try on current device with FP16
    if is_cuda:
        # Attempt 1: FP16 autocast
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(tensor)
            output = output.float().cpu()  # always convert back to FP32
            if not torch.isfinite(output).all() or output.min() < _VALID_MIN or output.max() > _VALID_MAX:
                log.warning(
                    "FP16 produced corrupt output, retrying in FP32"
                )
                _clear_gpu_cache()
            else:
                return output
        except torch.cuda.OutOfMemoryError:
            log.warning("OOM during FP16 forward pass, trying FP32")
            _clear_gpu_cache()

        # Attempt 2: FP32 on GPU
        try:
            with torch.no_grad():
                output = model(tensor)
            return output.cpu()
        except torch.cuda.OutOfMemoryError:
            log.warning("OOM during FP32 forward pass, falling back to CPU")
            _clear_gpu_cache()
    else:
        with torch.no_grad():
            output = model(tensor)
        return output.cpu()

    # CPU fallback
    model.cpu()
    tensor_cpu = tensor.cpu()
    with torch.no_grad():
        output = model(tensor_cpu)
    # Try to move model back
    if is_cuda:
        try:
            model.to(device)
        except Exception:
            pass
    return output
