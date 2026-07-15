import pytest
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from upscaler.ui.face_panel import FacePanel

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


def test_get_face_config_defaults():
    panel = FacePanel()
    cfg = panel.get_face_config()
    assert cfg["enabled"] is True
    assert 0.0 <= cfg["fidelity"] <= 1.0
    assert isinstance(cfg["upscale_background"], bool)


def test_signal_emitted_on_change():
    panel = FacePanel()
    received = {}
    panel.face_params_changed.connect(lambda d: received.update(d))
    panel._emit_changed()
    assert "fidelity" in received


# --- Этап 3: зоны лиц в панели ----------------------------------------------

def test_face_config_includes_regions():
    panel = FacePanel()
    panel.set_zones([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.2, 0.2]])
    cfg = panel.get_face_config()
    assert cfg["regions"] == [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.2, 0.2]]
    assert panel.zones_list.count() == 2


def test_face_config_no_zones_empty_regions():
    panel = FacePanel()
    assert panel.get_face_config()["regions"] == []


def test_clear_zones_button_emits():
    panel = FacePanel()
    panel.set_zones([[0.1, 0.1, 0.3, 0.3]])
    got = []
    panel.zones_edited.connect(lambda z: got.append(z))
    panel.clear_zones_btn.click()
    assert got and got[-1] == []
    assert panel.get_face_config()["regions"] == []


def test_delete_selected_zone_button():
    panel = FacePanel()
    panel.set_zones([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.2, 0.2]])
    panel.zones_list.setCurrentRow(0)
    got = []
    panel.zones_edited.connect(lambda z: got.append(z))
    panel.delete_zone_btn.click()
    assert got[-1] == [[0.5, 0.5, 0.2, 0.2]]


def test_zone_edit_toggle_signal():
    panel = FacePanel()
    got = []
    panel.zone_edit_toggled.connect(lambda v: got.append(v))
    panel.zone_edit_btn.setChecked(True)
    assert got == [True]


def test_apply_config_with_regions():
    panel = FacePanel()
    panel.apply_config({"fidelity": 0.5, "regions": [[0.2, 0.2, 0.4, 0.4]]})
    assert panel.get_face_config()["regions"] == [[0.2, 0.2, 0.4, 0.4]]


def test_clear_zones_public_method():
    panel = FacePanel()
    panel.set_zones([[0.1, 0.1, 0.3, 0.3]])
    got = []
    panel.zones_edited.connect(lambda z: got.append(z))
    panel.clear_zones()
    assert got and got[-1] == []
    assert panel.get_face_config()["regions"] == []


# --- Этап D: ретрансляция панели ---------------------------------------------

def test_face_panel_retranslate():
    from upscaler.ui import i18n
    panel = FacePanel()
    i18n.set_language("en")
    panel.retranslate()
    assert panel.preview_btn.text() == "Preview"
    i18n.set_language("ru")
    panel.retranslate()
    assert panel.preview_btn.text() == "Предпросмотр"
