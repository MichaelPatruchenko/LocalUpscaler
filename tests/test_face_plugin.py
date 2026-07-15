# tests/test_face_plugin.py
import numpy as np
import pytest
from upscaler.plugins.face.codeformer import CodeFormerPlugin
from upscaler.plugins.face import facedet
from upscaler.plugins.face.facedet import Face


def _img(size=400):
    return (np.random.default_rng(1).random((size, size, 3)) * 255).astype(np.uint8)


class _FakeSession:
    """Returns a constant bright face crop in CodeFormer's [-1,1] NCHW layout."""
    def get_inputs(self):
        class _I:
            name = "input"
        return [_I()]

    def run(self, _out, feeds):
        x = next(iter(feeds.values()))
        out = np.full_like(x, 0.6, dtype=np.float32)  # ~204 after denorm
        return [out]


def test_metadata():
    assert CodeFormerPlugin.name == "CodeFormer"
    assert CodeFormerPlugin.category == "face"


def test_no_faces_returns_input_unchanged(monkeypatch):
    monkeypatch.setattr(facedet, "detect_faces", lambda *a, **k: [])
    p = CodeFormerPlugin()
    p.initialize("cpu")
    monkeypatch.setattr(p, "_yunet_path", lambda: "fake")
    img = _img()
    out = p.process(img, {"enabled": True})
    assert np.array_equal(out, img)


def test_restores_detected_face(monkeypatch):
    lm = np.array([[160, 180], [240, 180], [200, 230],
                   [170, 280], [230, 280]], dtype=np.float32)
    monkeypatch.setattr(facedet, "detect_faces",
                        lambda *a, **k: [Face((140, 150, 120, 160), lm, 0.99)])
    p = CodeFormerPlugin()
    p.initialize("cpu")
    p._session = _FakeSession()           # bypass real ONNX load
    p._session_path = "fake"
    monkeypatch.setattr(p, "_yunet_path", lambda: "fake")
    img = np.full((400, 400, 3), 30, np.uint8)
    out = p.process(img, {"enabled": True, "fidelity": 0.7})
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out.mean() > img.mean()        # face region brightened


def test_disabled_returns_input(monkeypatch):
    p = CodeFormerPlugin()
    p.initialize("cpu")
    img = _img()
    assert np.array_equal(p.process(img, {"enabled": False}), img)


def test_plugin_is_discovered():
    from upscaler.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    reg.discover_builtin()
    names = [p.name for p in reg.list_plugins("face")]
    assert "CodeFormer" in names
    assert reg.get("CodeFormer") is not None


def test_alpha_channel_preserved(monkeypatch):
    lm = np.array([[160, 180], [240, 180], [200, 230],
                   [170, 280], [230, 280]], dtype=np.float32)
    monkeypatch.setattr(facedet, "detect_faces",
                        lambda *a, **k: [Face((140, 150, 120, 160), lm, 0.99)])
    p = CodeFormerPlugin()
    p.initialize("cpu")
    p._session = _FakeSession()
    p._session_path = "fake"
    monkeypatch.setattr(p, "_yunet_path", lambda: "fake")
    img = np.dstack([np.full((400, 400, 3), 30, np.uint8),
                     np.full((400, 400), 255, np.uint8)])
    out = p.process(img, {"enabled": True, "fidelity": 0.7})
    assert out.shape == img.shape  # 4 channels preserved


def test_load_session_respects_device(monkeypatch):
    """load_session uses CUDA provider only when device != cpu and it's available."""
    import sys
    import types

    captured = {}
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.get_available_providers = lambda: [
        "CUDAExecutionProvider", "CPUExecutionProvider"]

    class _Sess:
        def __init__(self, path, providers=None):
            captured["providers"] = providers

    fake_ort.InferenceSession = _Sess
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    from upscaler.plugins.face.pipeline_loader import load_session

    load_session("m.onnx", "cpu")
    assert captured["providers"] == ["CPUExecutionProvider"]

    load_session("m.onnx", "cuda")
    assert captured["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]


# --- Этап 3: ручные зоны заменяют автодетекцию ------------------------------
def _img(w=200, h=200):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_regions_skip_global_detection(monkeypatch):
    plugin = CodeFormerPlugin()
    monkeypatch.setattr(plugin, "_yunet_path", lambda: "unused.onnx")
    calls = {"global": 0, "crops": []}

    def fake_detect(rgb, path, score_threshold=0.6):
        # Глобальный вызов получил бы полный кадр 200x200
        if rgb.shape[:2] == (200, 200):
            calls["global"] += 1
        else:
            calls["crops"].append(rgb.shape[:2])
        return []

    monkeypatch.setattr(
        "upscaler.plugins.face.codeformer.facedet.detect_faces", fake_detect)
    # Сессия не должна понадобиться до детекции; заглушим на всякий случай
    monkeypatch.setattr(plugin, "_get_session",
                        lambda: (_ for _ in ()).throw(RuntimeError("no session")))
    out = plugin.process(_img(), {"regions": [[0.25, 0.25, 0.5, 0.5]],
                                  "min_face_px": 16})
    assert calls["global"] == 0          # глобальная детекция не вызывалась
    assert len(calls["crops"]) == 1      # YuNet искал только внутри зоны
    assert out.shape == (200, 200, 3)


def test_faces_from_regions_synthetic_fallback(monkeypatch):
    plugin = CodeFormerPlugin()
    monkeypatch.setattr(plugin, "_yunet_path", lambda: "unused.onnx")
    monkeypatch.setattr(
        "upscaler.plugins.face.codeformer.facedet.detect_faces",
        lambda *a, **k: [])
    faces = plugin._faces_from_regions(_img(), [[0.25, 0.25, 0.5, 0.5]])
    assert len(faces) == 1
    assert faces[0].bbox == (50, 50, 100, 100)
    assert faces[0].landmarks.shape == (5, 2)  # синтетические лендмарки


def test_faces_from_regions_uses_detected_landmarks(monkeypatch):
    plugin = CodeFormerPlugin()
    monkeypatch.setattr(plugin, "_yunet_path", lambda: "unused.onnx")
    lms = np.ones((5, 2), np.float32) * 10.0

    def fake_detect(rgb, path, score_threshold=0.6):
        return [facedet.Face(bbox=(0, 0, 10, 10), landmarks=lms, score=0.9)]

    monkeypatch.setattr(
        "upscaler.plugins.face.codeformer.facedet.detect_faces", fake_detect)
    faces = plugin._faces_from_regions(_img(), [[0.25, 0.25, 0.5, 0.5]])
    assert len(faces) == 1
    # Зона 50..150, запас 20% -> кроп с (30, 30): лендмарки смещены на origin кропа
    assert np.allclose(faces[0].landmarks, lms + np.array([30, 30], np.float32))


def test_faces_from_regions_skips_degenerate():
    plugin = CodeFormerPlugin()
    faces = plugin._faces_from_regions(_img(), [[0.5, 0.5, 0.001, 0.001]])
    assert faces == []


def test_regions_respect_min_face_px(monkeypatch):
    plugin = CodeFormerPlugin()
    monkeypatch.setattr(plugin, "_yunet_path", lambda: "unused.onnx")
    monkeypatch.setattr(
        "upscaler.plugins.face.codeformer.facedet.detect_faces",
        lambda *a, **k: [])
    infer_calls = []
    monkeypatch.setattr(plugin, "_get_session", lambda: object())
    monkeypatch.setattr(plugin, "_infer",
                        lambda s, c, f: infer_calls.append(1) or c)
    # Зона 20x20 px < min_face_px=64 -> теперь обрабатывается (ручная зона)
    plugin.process(_img(), {"regions": [[0.4, 0.4, 0.1, 0.1]],
                            "min_face_px": 64})
    assert infer_calls == [1]


def test_manual_region_ignores_min_face_px(monkeypatch):
    """Ручная зона обрабатывается даже если меньше min_face_px."""
    import numpy as np
    from upscaler.plugins.face.codeformer import CodeFormerPlugin
    import upscaler.plugins.face.codeformer as cf
    plugin = CodeFormerPlugin()
    monkeypatch.setattr(plugin, "_yunet_path", lambda: "unused.onnx")
    monkeypatch.setattr(cf.facedet, "detect_faces", lambda *a, **k: [])
    infer = []
    monkeypatch.setattr(plugin, "_get_session", lambda: object())
    monkeypatch.setattr(plugin, "_infer",
                        lambda s, c, f: infer.append(1) or c)
    img = np.full((200, 200, 3), 128, np.uint8)
    # Зона 40x40, min_face_px=64 -> раньше пропускалась; теперь обрабатывается
    plugin.process(img, {"regions": [[0.4, 0.4, 0.2, 0.2]],
                         "min_face_px": 64})
    assert infer == [1]


def test_auto_detect_still_respects_min_face_px(monkeypatch):
    """Без ручных зон (авто-детекция) min_face_px по-прежнему фильтрует."""
    import numpy as np
    from upscaler.plugins.face.codeformer import CodeFormerPlugin
    from upscaler.plugins.face.facedet import Face
    import upscaler.plugins.face.codeformer as cf
    plugin = CodeFormerPlugin()
    monkeypatch.setattr(plugin, "_yunet_path", lambda: "unused.onnx")
    small = Face(bbox=(80, 80, 40, 40),
                 landmarks=np.zeros((5, 2), np.float32), score=0.9)
    monkeypatch.setattr(cf.facedet, "detect_faces", lambda *a, **k: [small])
    infer = []
    monkeypatch.setattr(plugin, "_get_session", lambda: object())
    monkeypatch.setattr(plugin, "_infer",
                        lambda s, c, f: infer.append(1) or c)
    img = np.full((200, 200, 3), 128, np.uint8)
    plugin.process(img, {"min_face_px": 64})  # нет regions
    assert infer == []  # мелкое авто-лицо отфильтровано


def test_faces_from_regions_passes_angle(monkeypatch):
    import numpy as np
    from upscaler.plugins.face.codeformer import CodeFormerPlugin
    import upscaler.plugins.face.codeformer as cf
    from upscaler.plugins.face.regions import synthetic_landmarks
    plugin = CodeFormerPlugin()
    monkeypatch.setattr(plugin, "_yunet_path", lambda: "unused.onnx")
    monkeypatch.setattr(cf.facedet, "detect_faces", lambda *a, **k: [])
    img = np.full((200, 200, 3), 128, np.uint8)
    faces = plugin._faces_from_regions(img, [[0.25, 0.25, 0.5, 0.5, 30.0]])
    assert len(faces) == 1
    # angle применён -> лендмарки не совпадают с осевыми
    axis = synthetic_landmarks(50, 50, 100, 100, angle=0.0)
    assert not np.allclose(faces[0].landmarks, axis)
