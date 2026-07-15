import numpy as np
import pytest

torch = pytest.importorskip("torch")
from upscaler.engine.gpu_utils import safe_tile_process, _FP16CorruptionError


class _OvershootModel(torch.nn.Module):
    """Finite output far outside [0,1]/[-1,2] — a model that legitimately overshoots."""
    def forward(self, x):
        return x * 5.0 + 3.0


class _NaNModel(torch.nn.Module):
    def forward(self, x):
        return x * float("nan")


def test_finite_out_of_range_is_clamped_not_errored_on_cpu():
    # Regression: NAFNet's finite out-of-range output was treated as fatal
    # "FP16 corruption" on the CPU/FP32 path, erroring the whole job. Finite
    # output must be clamped to the image range and accepted, not rejected.
    img = np.random.default_rng(0).random((48, 48, 3)).astype(np.float32)
    out = safe_tile_process(img, scale=1, device="cpu", model=_OvershootModel(),
                            tile_size_hint=48, overlap=8, model_memory_mb=0)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_nan_output_still_raises_on_cpu():
    # NaN/inf is genuine corruption and must still be detected.
    img = np.random.default_rng(0).random((48, 48, 3)).astype(np.float32)
    with pytest.raises(_FP16CorruptionError):
        safe_tile_process(img, scale=1, device="cpu", model=_NaNModel(),
                          tile_size_hint=48, overlap=8, model_memory_mb=0)
