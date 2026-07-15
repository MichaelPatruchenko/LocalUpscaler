"""BlendPanel: чисто-UI логика вкладки «Смешивание» (список вариантов).

Блок G, задача 4: base/layer-комбобоксы и список слоёв заменены списком
вариантов (VariantPanel на VersionListPanel) с двойным выбором
(первичный/вторичный, как в HistoryPanel) + режим наложения/прозрачность +
кнопки «Смешать выбранные»/«Авто-подбор»/«Предпросмотр»/«Применить».
"""
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from upscaler.engine.blend import BLEND_MODES
from upscaler.ui.blend_panel import BlendPanel

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolate_i18n(tmp_path, monkeypatch):
    """Изолировать настройки языка и синглтон переводчика для каждого теста."""
    from upscaler import config
    from upscaler.ui import i18n
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(i18n, "_translator", None)
    yield
    i18n.set_language("ru")
    monkeypatch.setattr(i18n, "_translator", None)


def _panel(n: int = 0) -> BlendPanel:
    """Панель с *n* вариантами-заглушками (id 1..n), последний — primary."""
    p = BlendPanel()
    for i in range(1, n + 1):
        p.add_variant(i, np.zeros((4, 4, 3), np.uint8), f"step{i}")
    return p


# --- Контракт (из брифа задачи 4) --------------------------------------------

def test_blend_panel_has_variant_list_and_controls():
    p = BlendPanel()
    assert hasattr(p, "variant_list")
    assert hasattr(p, "mode_combo") and hasattr(p, "opacity_slider")
    assert hasattr(p, "blend_selected_btn")


def test_blend_selected_emits_recipe():
    p = _panel()
    for i in (1, 2):
        p.add_variant(i, np.zeros((4, 4, 3), np.uint8), f"step{i}")
    p.variant_list.list_widget.primary_clicked.emit(1)
    p.variant_list.list_widget.secondary_clicked.emit(2)
    idx = p.mode_combo.findData("soft_light")
    p.mode_combo.setCurrentIndex(idx)
    p.opacity_slider.setValue(60)
    got = []
    p.blend_selected_requested.connect(lambda r: got.append(r))
    p.blend_selected_btn.click()
    assert got and got[0]["primary"] == 1 and got[0]["secondary"] == 2
    assert got[0]["mode"] == "soft_light"
    assert abs(got[0]["opacity"] - 0.6) < 1e-6


def test_blend_selected_needs_two(qtbot=None):
    p = BlendPanel()
    p.add_variant(1, np.zeros((4, 4, 3), np.uint8), "s1")
    p.variant_list.list_widget.primary_clicked.emit(1)
    got = []
    p.blend_selected_requested.connect(lambda r: got.append(r))
    p.blend_selected_btn.click()
    assert got == []  # нужен второй вариант


# --- Дополнительное покрытие новой модели ------------------------------------

def test_mode_combo_covers_all_modes():
    p = _panel()
    modes = {p.mode_combo.itemData(i) for i in range(p.mode_combo.count())}
    assert modes == set(BLEND_MODES)


def test_add_variant_delegates_to_variant_list():
    p = BlendPanel()
    p.add_variant(7, np.zeros((4, 4, 3), np.uint8), "v7")
    assert p.variant_list.list_widget.count() == 1
    assert p.variant_list.selected() == (7, None)


def test_set_variants_populates_list_and_replaces_previous():
    p = BlendPanel()
    p.add_variant(1, np.zeros((4, 4, 3), np.uint8), "stale")
    p.set_variants([
        (10, np.zeros((4, 4, 3), np.uint8), "a"),
        (11, np.zeros((4, 4, 3), np.uint8), "b"),
    ])
    assert p.variant_list.list_widget.count() == 2
    ids = set()
    for i in range(p.variant_list.list_widget.count()):
        item = p.variant_list.list_widget.item(i)
        from PySide6.QtCore import Qt
        ids.add(int(item.data(Qt.ItemDataRole.UserRole)))
    assert ids == {10, 11}


def test_selection_changed_forwarded_from_variant_list():
    p = _panel(2)
    got = []
    p.selection_changed.connect(lambda a, b: got.append((a, b)))
    p.variant_list.list_widget.primary_clicked.emit(1)
    p.variant_list.list_widget.secondary_clicked.emit(2)
    assert got[-1] == (1, 2)


def test_preview_and_apply_need_two_variants():
    p = _panel(1)
    p.variant_list.list_widget.primary_clicked.emit(1)
    got = []
    p.preview_requested.connect(lambda r: got.append(("preview", r)))
    p.apply_requested.connect(lambda r: got.append(("apply", r)))
    p.preview_btn.click()
    p.apply_btn.click()
    assert got == []


def test_preview_and_apply_emit_recipe_with_two_variants():
    p = _panel(2)
    p.variant_list.list_widget.primary_clicked.emit(1)
    p.variant_list.list_widget.secondary_clicked.emit(2)
    idx = p.mode_combo.findData("screen")
    p.mode_combo.setCurrentIndex(idx)
    p.opacity_slider.setValue(25)
    got = {}
    p.preview_requested.connect(lambda r: got.setdefault("preview", r))
    p.apply_requested.connect(lambda r: got.setdefault("apply", r))
    p.preview_btn.click()
    p.apply_btn.click()
    for key in ("preview", "apply"):
        assert got[key]["primary"] == 1
        assert got[key]["secondary"] == 2
        assert got[key]["mode"] == "screen"
        assert abs(got[key]["opacity"] - 0.25) < 1e-6


def test_auto_requested_emitted_without_selection():
    p = BlendPanel()
    got = []
    p.auto_requested.connect(lambda: got.append(True))
    p.auto_btn.click()
    assert got == [True]


# --- Этап D: ретрансляция панели ---------------------------------------------

def test_blend_panel_retranslate():
    from upscaler.ui import i18n
    p = _panel()
    i18n.set_language("en")
    p.retranslate()
    assert p.auto_btn.text() == "Auto-select"
    i18n.set_language("ru")
    p.retranslate()
    assert p.auto_btn.text() == "Авто-подбор"
