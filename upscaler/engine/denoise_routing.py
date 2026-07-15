"""Routing helpers: substitute the CPU BM3D denoiser with GPU SCUNet."""
import logging

log = logging.getLogger(__name__)


def cuda_available(device: str) -> bool:
    """True when the resolved device would use CUDA.

    Mirrors ModelManager.get_device: "cpu" -> False; "auto"/"cuda*" -> depends
    on torch.cuda.is_available(). torch is optional, so any import error -> False.
    """
    if not isinstance(device, str) or device == "cpu":
        return False
    if device != "auto" and not device.startswith("cuda"):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _bm3d_sigma_to_strength(sigma) -> float:
    """Map BM3D sigma (1-75) to SCUNet strength (0.1-1.0)."""
    try:
        s = float(sigma)
    except (TypeError, ValueError):
        s = 25.0
    return round(min(max(s / 50.0, 0.1), 1.0), 2)


def substitute_gpu_denoise(denoise_cfg: dict, gpu_available: bool,
                           enabled: bool) -> dict:
    """Return a NEW denoise config with BM3D routed to GPU SCUNet when allowed.

    When ``enabled`` AND ``gpu_available`` AND "BM3D" is present: drop BM3D and
    add a SCUNet entry (strength mapped from BM3D sigma); if SCUNet is already
    present, keep it and just drop BM3D. Otherwise return the config unchanged.
    Never mutates the input.
    """
    if not isinstance(denoise_cfg, dict):
        return denoise_cfg
    new_cfg = dict(denoise_cfg)
    if not (enabled and gpu_available and "BM3D" in new_cfg):
        return new_cfg
    bm3d_params = new_cfg.pop("BM3D")
    if "SCUNet" not in new_cfg:
        sigma = bm3d_params.get("sigma", 25) if isinstance(bm3d_params, dict) else 25
        new_cfg["SCUNet"] = {"strength": _bm3d_sigma_to_strength(sigma)}
    log.info("BM3D -> SCUNet (GPU denoise)")
    return new_cfg
