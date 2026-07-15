import numpy as np
import pytest
from upscaler.plugins.deblur.kernels import (
    focus_kernel, motion_kernel, gaussian_kernel,
)


def test_gaussian_kernel_normalized_and_symmetric():
    k = gaussian_kernel(3.0)
    assert k.dtype == np.float32
    assert abs(k.sum() - 1.0) < 1e-5
    assert np.allclose(k, k[::-1, :], atol=1e-6)
    assert np.allclose(k, k[:, ::-1], atol=1e-6)
    cy, cx = k.shape[0] // 2, k.shape[1] // 2
    assert k[cy, cx] == k.max()


def test_focus_kernel_normalized_disk():
    k = focus_kernel(5.0, edge_feather=10.0, correction_strength=0.0)
    assert abs(k.sum() - 1.0) < 1e-5
    cy, cx = k.shape[0] // 2, k.shape[1] // 2
    assert k[cy, cx] > 0
    assert k[0, 0] == 0


def test_focus_kernel_with_edge_correction():
    # Exercises the gaussian edge-ring correction branch.
    k = focus_kernel(6.0, edge_feather=30.0, correction_strength=50.0)
    assert abs(k.sum() - 1.0) < 1e-5
    assert np.isfinite(k).all()
    assert (k >= 0).all()
    # out-of-range correction must be clamped, not produce NaN/garbage
    k2 = focus_kernel(6.0, edge_feather=30.0, correction_strength=500.0)
    assert np.isfinite(k2).all()
    assert abs(k2.sum() - 1.0) < 1e-5


def test_motion_kernel_normalized_and_directional():
    horizontal = motion_kernel(6.0, angle=0.0)
    assert abs(horizontal.sum() - 1.0) < 1e-5
    cy = horizontal.shape[0] // 2
    row_energy = horizontal[cy, :].sum()
    assert row_energy > 0.5


def test_motion_kernel_angle_changes_orientation():
    h = motion_kernel(6.0, angle=0.0)
    v = motion_kernel(6.0, angle=90.0)
    cy, cx = h.shape[0] // 2, h.shape[1] // 2
    assert h[cy, :].sum() > h[:, cx].sum()
    assert v[:, cx].sum() > v[cy, :].sum()


def test_zero_protection():
    k = gaussian_kernel(0.1)
    assert np.isfinite(k).all()
    assert abs(k.sum() - 1.0) < 1e-5
