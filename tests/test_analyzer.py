"""Tests for SourceAnalyzer."""
import cv2
import numpy as np
import pytest


class TestSourceAnalyzer:
    def test_analyze_returns_dict(self, sample_rgb_uint8):
        from upscaler.engine.analyzer import SourceAnalyzer
        a = SourceAnalyzer()
        result = a.analyze(sample_rgb_uint8, {"format": ".png", "bit_depth": 8, "icc_profile": None})
        assert isinstance(result, dict)
        assert "noise_level" in result
        assert "color_space" in result
        assert "bit_depth" in result
        assert "resolution" in result

    def test_noise_estimate_low_for_clean(self):
        from upscaler.engine.analyzer import SourceAnalyzer
        clean = np.full((64, 64, 3), 128, dtype=np.uint8)
        a = SourceAnalyzer()
        result = a.analyze(clean, {"format": ".png", "bit_depth": 8, "icc_profile": None})
        assert result["noise_level"] < 5

    def test_noise_estimate_high_for_noisy(self):
        from upscaler.engine.analyzer import SourceAnalyzer
        rng = np.random.default_rng(42)
        noisy = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        a = SourceAnalyzer()
        result = a.analyze(noisy, {"format": ".png", "bit_depth": 8, "icc_profile": None})
        assert result["noise_level"] > 10


def test_analyze_includes_blur_assessment():
    from upscaler.engine.analyzer import SourceAnalyzer
    rng = np.random.default_rng(11)
    sharp = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4.0)
    a = SourceAnalyzer().analyze(blurred, {"bit_depth": 8}, detect_faces=False)
    assert "blur_assessment" in a
    assert a["blur_assessment"]["needs_deblur"] is True


# --- Этап 1: метрики качества лиц -----------------------------------------
from upscaler.plugins.face.facedet import Face


def _face_like_crop(sharp: bool, size: int = 96) -> np.ndarray:
    """Синтетическая «лицевая» текстура: резкая или размытая."""
    rng = np.random.default_rng(3)
    img = (rng.random((size, size, 3)) * 120 + 80).astype(np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, (200, 170, 150), -1)
    cv2.rectangle(img, (20, 20), (40, 40), (30, 30, 30), -1)
    if not sharp:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=4.0)
    return img


def _scene_with_face(sharp: bool):
    scene = np.full((200, 200, 3), 128, dtype=np.uint8)
    scene[50:146, 50:146] = _face_like_crop(sharp)
    face = Face(bbox=(50, 50, 96, 96),
                landmarks=np.zeros((5, 2), np.float32), score=0.9)
    return scene, [face]


def test_face_quality_metrics_keys_and_ranges():
    from upscaler.engine.analyzer import SourceAnalyzer
    scene, faces = _scene_with_face(sharp=True)
    out = SourceAnalyzer._face_quality_metrics(scene, faces, inv_scale=1.0)
    assert set(out) == {"face_sharpness", "face_noise", "face_min_px"}
    assert 0.0 <= out["face_sharpness"] <= 1.0
    assert out["face_noise"] >= 0.0
    assert out["face_min_px"] == 96


def test_face_quality_sharp_scores_higher_than_blurred():
    from upscaler.engine.analyzer import SourceAnalyzer
    sharp_scene, faces = _scene_with_face(sharp=True)
    blur_scene, _ = _scene_with_face(sharp=False)
    s = SourceAnalyzer._face_quality_metrics(sharp_scene, faces, 1.0)
    b = SourceAnalyzer._face_quality_metrics(blur_scene, faces, 1.0)
    assert s["face_sharpness"] > b["face_sharpness"]


def test_face_quality_min_px_uses_inv_scale():
    from upscaler.engine.analyzer import SourceAnalyzer
    scene, faces = _scene_with_face(sharp=True)
    out = SourceAnalyzer._face_quality_metrics(scene, faces, inv_scale=2.0)
    assert out["face_min_px"] == 192  # 96 px на детекционной копии, x2 в оригинале


def test_face_quality_empty_faces_gives_defaults():
    from upscaler.engine.analyzer import SourceAnalyzer
    scene, _ = _scene_with_face(sharp=True)
    out = SourceAnalyzer._face_quality_metrics(scene, [], 1.0)
    assert out == {}


def test_face_quality_bbox_clamped_to_image():
    from upscaler.engine.analyzer import SourceAnalyzer
    scene, _ = _scene_with_face(sharp=True)
    face = Face(bbox=(180, 180, 60, 60),  # выходит за край 200x200
                landmarks=np.zeros((5, 2), np.float32), score=0.9)
    out = SourceAnalyzer._face_quality_metrics(scene, [face], 1.0)
    assert out["face_min_px"] == 20  # кламп: 200-180


def test_analysis_contains_effective_downscale_factor():
    import numpy as np
    from upscaler.engine.analyzer import SourceAnalyzer
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    out = SourceAnalyzer().analyze(img, {}, detect_faces=False)
    assert 0.0 < out["effective_downscale_factor"] <= 1.0


# --- Этап B: метрики дефектов фото ------------------------------------------


def _flaws(img):
    from upscaler.engine.analyzer import SourceAnalyzer
    return SourceAnalyzer().analyze(img, {}, detect_faces=False)


def test_haze_level_high_on_hazy_image():
    import numpy as np
    rng = np.random.default_rng(5)
    base = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    hazy = np.clip(base.astype(np.float64) * 0.4 + 255 * 0.5,
                   0, 255).astype(np.uint8)
    assert _flaws(hazy)["haze_level"] > _flaws(base)["haze_level"] + 0.2


def test_clip_fractions():
    import numpy as np
    img = np.full((100, 100, 3), 128, np.uint8)
    img[:10] = 0      # 10% клиппинг теней
    img[-5:] = 255    # 5% клиппинг светов
    out = _flaws(img)
    assert 0.05 <= out["shadow_clip"] <= 0.15
    assert 0.02 <= out["highlight_clip"] <= 0.08
    assert out["shadow_mass"] >= out["shadow_clip"]


def test_vignette_strength_detects_dark_corners():
    import numpy as np
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float64)
    r2 = ((xx - 64) / 64) ** 2 + ((yy - 64) / 64) ** 2
    img = np.clip(200 * (1.0 - 0.4 * r2 / 2.0), 0, 255).astype(np.uint8)
    img3 = np.stack([img] * 3, axis=2)
    flat = np.full((128, 128, 3), 200, np.uint8)
    assert _flaws(img3)["vignette_strength"] > \
        _flaws(flat)["vignette_strength"] + 0.1


def test_flaw_metrics_present_and_bounded():
    import numpy as np
    rng = np.random.default_rng(6)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    out = _flaws(img)
    for key in ("haze_level", "shadow_clip", "highlight_clip",
                "shadow_mass", "vignette_strength"):
        assert 0.0 <= out[key] <= 1.0, key
