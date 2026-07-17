"""Надёжная детекция лиц: нормализация разрешения, обратное масштабирование
координат, фильтр правдоподобности (спек 2026-07-17, Часть 1)."""
import numpy as np
import pytest

from upscaler.plugins.face.facedet import (
    Face, detect_faces, plausible_face,
)


def _valid_landmarks(x, y, w, h):
    """Правдоподобная геометрия лица внутри bbox (x, y, w, h)."""
    return np.array([
        [x + 0.30 * w, y + 0.35 * h],   # правый глаз
        [x + 0.70 * w, y + 0.35 * h],   # левый глаз
        [x + 0.50 * w, y + 0.55 * h],   # нос
        [x + 0.35 * w, y + 0.75 * h],   # правый угол рта
        [x + 0.65 * w, y + 0.75 * h],   # левый угол рта
    ], dtype=np.float32)


def _face(x=100, y=100, w=100, h=120, lms=None, score=0.9):
    if lms is None:
        lms = _valid_landmarks(x, y, w, h)
    return Face(bbox=(x, y, w, h), landmarks=lms, score=score)


class _FakeDetector:
    """Подменяет cv2.FaceDetectorYN: фиксирует размер входа и возвращает
    заранее заданные строки-детекции в координатах детекционной копии."""
    last_input_sizes: list = []

    def __init__(self, rows):
        self._rows = rows

    def setInputSize(self, size):
        _FakeDetector.last_input_sizes.append(tuple(size))

    def detect(self, bgr):
        if not self._rows:
            return 0, None
        return 1, np.array(self._rows, dtype=np.float32)


def _row(x, y, w, h, score=0.9):
    """Строка YuNet: bbox + 5 правдоподобных лендмарков + score."""
    lms = _valid_landmarks(x, y, w, h)
    return [x, y, w, h] + lms.ravel().tolist() + [score]


@pytest.fixture
def fake_yunet(monkeypatch):
    """Заглушка cv2.FaceDetectorYN.create; возвращает setter строк детекции."""
    import cv2
    holder = {"rows": []}
    _FakeDetector.last_input_sizes = []

    class _FakeFactory:
        @staticmethod
        def create(*args, **kwargs):
            return _FakeDetector(holder["rows"])

    monkeypatch.setattr(cv2, "FaceDetectorYN", _FakeFactory)

    def set_rows(rows):
        holder["rows"] = rows

    return set_rows


# ─── Обратное масштабирование ───

def test_large_image_coords_scaled_back_to_original(fake_yunet):
    # 2560x2560 -> детекция на 640 (scale=0.25); bbox в координатах копии
    # должен вернуться умноженным на 4.
    fake_yunet([_row(40, 50, 80, 90)])
    img = np.zeros((2560, 2560, 3), np.uint8)
    faces = detect_faces(img, "model.onnx")
    assert len(faces) == 1
    assert faces[0].bbox == (160, 200, 320, 360)
    np.testing.assert_allclose(
        faces[0].landmarks, _valid_landmarks(40, 50, 80, 90) * 4.0, atol=1e-3)


def test_detection_runs_on_normalized_size(fake_yunet):
    fake_yunet([])
    img = np.zeros((2560, 1280, 3), np.uint8)
    detect_faces(img, "model.onnx")
    assert _FakeDetector.last_input_sizes[-1] == (320, 640)


def test_small_image_not_resized_coords_unchanged(fake_yunet):
    fake_yunet([_row(10, 12, 60, 70)])
    img = np.zeros((200, 300, 3), np.uint8)
    faces = detect_faces(img, "model.onnx")
    assert _FakeDetector.last_input_sizes[-1] == (300, 200)
    assert len(faces) == 1
    assert faces[0].bbox == (10, 12, 60, 70)
    np.testing.assert_allclose(
        faces[0].landmarks, _valid_landmarks(10, 12, 60, 70), atol=1e-3)


def test_sorted_by_descending_score(fake_yunet):
    fake_yunet([_row(10, 10, 50, 60, score=0.7),
                _row(120, 10, 50, 60, score=0.95)])
    img = np.zeros((300, 300, 3), np.uint8)
    faces = detect_faces(img, "model.onnx")
    assert [round(f.score, 2) for f in faces] == [0.95, 0.7]


def test_no_detections_returns_empty(fake_yunet):
    fake_yunet([])
    assert detect_faces(np.zeros((100, 100, 3), np.uint8), "model.onnx") == []


def test_failure_degrades_to_empty(monkeypatch):
    import cv2

    class _Boom:
        @staticmethod
        def create(*a, **k):
            raise RuntimeError("no model")

    monkeypatch.setattr(cv2, "FaceDetectorYN", _Boom)
    assert detect_faces(np.zeros((100, 100, 3), np.uint8), "missing.onnx") == []


# ─── Фильтр правдоподобности ───

def test_plausible_valid_geometry_passes():
    assert plausible_face(_face())


def test_implausible_eyes_below_mouth_rejected():
    x, y, w, h = 100, 100, 100, 120
    lms = _valid_landmarks(x, y, w, h)
    lms[:, 1] = (y + h) - (lms[:, 1] - y)  # вертикальное отражение
    assert not plausible_face(_face(lms=lms))


def test_implausible_degenerate_eyes_rejected():
    lms = _valid_landmarks(100, 100, 100, 120)
    lms[1] = lms[0]  # совпавшие глаза -> межзрачковое = 0
    assert not plausible_face(_face(lms=lms))


def test_implausible_landmarks_outside_bbox_rejected():
    lms = _valid_landmarks(100, 100, 100, 120)
    lms[2] = [400.0, 160.0]  # нос далеко за bbox (даже с запасом 30%)
    assert not plausible_face(_face(lms=lms))


def test_implausible_aspect_ratio_rejected():
    assert not plausible_face(_face(w=300, h=60))
    assert not plausible_face(_face(w=40, h=200))


def test_implausible_nan_rejected():
    lms = _valid_landmarks(100, 100, 100, 120)
    lms[0, 0] = np.nan
    assert not plausible_face(_face(lms=lms))


def test_detect_faces_filters_implausible(fake_yunet):
    bad = _row(10, 10, 60, 70)
    # глаза ниже рта: отражаем y-координаты лендмарков внутри bbox
    lms = _valid_landmarks(10, 10, 60, 70)
    lms[:, 1] = (10 + 70) - (lms[:, 1] - 10)
    bad[4:14] = lms.ravel().tolist()
    good = _row(150, 10, 60, 70)
    fake_yunet([bad, good])
    faces = detect_faces(np.zeros((300, 300, 3), np.uint8), "model.onnx")
    assert len(faces) == 1
    assert faces[0].bbox[0] == 150
