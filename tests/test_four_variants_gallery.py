"""FourVariantsGalleryDialog: сетка кандидатов, доступность выбора."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QPushButton

from upscaler.ui.four_variants_gallery import FourVariantsGalleryDialog

_app = QApplication.instance() or QApplication([])


def _cand(vid, status="done", img=True):
    return {
        "id": vid, "name_key": f"variants.{vid}",
        "style_directive": "", "config": {},
        "result_image": np.zeros((16, 16, 3), np.uint8) if img else None,
        "metrics": {"brisque": 33.3, "niqe": 4.2}, "status": status,
    }


def test_gallery_choose_buttons_follow_status():
    cands = [_cand("natural"), _cand("sharp", status="failed", img=False),
             _cand("clean", status="satisfied", img=False), _cand("vivid")]
    dlg = FourVariantsGalleryDialog(cands, iteration=2, max_iter=3)
    buttons = dlg.findChildren(QPushButton)
    assert [b.isEnabled() for b in buttons] == [True, False, False, True]
    assert dlg.selected_index is None


def test_gallery_click_sets_selected_index_and_accepts():
    cands = [_cand("natural"), _cand("sharp"),
             _cand("clean"), _cand("vivid")]
    dlg = FourVariantsGalleryDialog(cands)
    buttons = dlg.findChildren(QPushButton)
    buttons[2].click()
    assert dlg.selected_index == 2
    assert dlg.result() == FourVariantsGalleryDialog.DialogCode.Accepted


def test_gallery_close_without_choice_keeps_none():
    dlg = FourVariantsGalleryDialog([_cand("natural")])
    dlg.reject()
    assert dlg.selected_index is None
