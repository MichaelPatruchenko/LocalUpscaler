import numpy as np
from upscaler.plugins.face.align import (
    FFHQ_512_TEMPLATE, align_face, paste_back,
)
from upscaler.plugins.face.facedet import detect_faces


def test_template_shape():
    assert FFHQ_512_TEMPLATE.shape == (5, 2)


def test_align_returns_512_crop_and_affine():
    img = (np.random.default_rng(0).random((400, 400, 3)) * 255).astype(np.uint8)
    # landmarks roughly centered
    lm = np.array([[160, 180], [240, 180], [200, 230],
                   [170, 280], [230, 280]], dtype=np.float32)
    crop, M = align_face(img, lm)
    assert crop.shape == (512, 512, 3)
    assert crop.dtype == np.uint8
    assert M.shape == (2, 3)


def test_paste_back_preserves_shape_dtype_and_changes_region():
    img = np.full((400, 400, 3), 30, dtype=np.uint8)
    lm = np.array([[160, 180], [240, 180], [200, 230],
                   [170, 280], [230, 280]], dtype=np.float32)
    crop, M = align_face(img, lm)
    restored = np.full((512, 512, 3), 200, dtype=np.uint8)  # bright face
    out = paste_back(img, restored, M)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    # The face region should now be brighter than the untouched background.
    assert out.mean() > img.mean()


def test_detect_faces_bad_model_path_returns_empty():
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    # nonexistent model path must not raise -> []
    assert detect_faces(img, "does_not_exist.onnx") == []


def test_detect_faces_missing_yunet_returns_empty(monkeypatch):
    import cv2
    monkeypatch.delattr(cv2, "FaceDetectorYN", raising=False)
    img = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype(np.uint8)
    assert detect_faces(img, "whatever.onnx") == []
