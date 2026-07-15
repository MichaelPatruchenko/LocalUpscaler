"""Skin Smooth: frequency separation в маске кожи."""
import numpy as np
from upscaler.plugins.face.facedet import Face


def _plugin(faces):
    from upscaler.plugins.adjusters.skin_smooth import SkinSmoothPlugin
    p = SkinSmoothPlugin()
    p.initialize("cpu")
    p._detect_faces = lambda rgb: faces
    return p


def _skin_scene():
    """Кадр с «кожаной» текстурной областью и синим шумным фоном."""
    rng = np.random.default_rng(7)
    img = np.zeros((96, 96, 3), np.uint8)
    img[..., 2] = rng.integers(100, 200, (96, 96))          # синий фон
    skin = np.stack([
        rng.integers(180, 220, (48, 48)),
        rng.integers(120, 160, (48, 48)),
        rng.integers(90, 130, (48, 48)),
    ], axis=2).astype(np.uint8)
    img[24:72, 24:72] = skin
    return img


def _face():
    return Face(bbox=(24, 24, 48, 48),
                landmarks=np.zeros((5, 2), np.float32), score=0.9)


def test_no_faces_noop():
    img = _skin_scene()
    out = _plugin([]).process(img, {"strength": 1.0})
    assert np.array_equal(out, img)


def test_smooths_inside_face_region():
    img = _skin_scene()
    out = _plugin([_face()]).process(img, {"strength": 1.0, "radius": 8})
    var_in = img[32:64, 32:64].astype(float).var()
    var_out = out[32:64, 32:64].astype(float).var()
    assert var_out < var_in * 0.9


def test_background_untouched():
    img = _skin_scene()
    out = _plugin([_face()]).process(img, {"strength": 1.0, "radius": 8})
    assert np.abs(out[:12, :12].astype(int) - img[:12, :12].astype(int)).max() <= 1


def test_zero_strength_noop():
    img = _skin_scene()
    out = _plugin([_face()]).process(img, {"strength": 0.0})
    assert np.array_equal(out, img)


def test_detection_error_graceful():
    from upscaler.plugins.adjusters.skin_smooth import SkinSmoothPlugin
    p = SkinSmoothPlugin()
    p.initialize("cpu")
    p._detect_faces = lambda rgb: (_ for _ in ()).throw(RuntimeError("no model"))
    img = _skin_scene()
    out = p.process(img, {"strength": 1.0})
    assert np.array_equal(out, img)
