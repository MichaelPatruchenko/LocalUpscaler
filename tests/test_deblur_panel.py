import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from upscaler.ui.deblur_panel import DeblurPanel

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


def test_auto_mode_config():
    panel = DeblurPanel()
    panel.set_auto(True)
    cfg = panel.get_deblur_config()
    assert cfg["auto"] is True


def test_manual_mode_config_has_params():
    panel = DeblurPanel()
    panel.set_auto(False)
    panel.set_blur_type("motion")
    cfg = panel.get_deblur_config()
    assert cfg["auto"] is False
    assert cfg["blur_type"] == "motion"
    assert "radius" in cfg and "angle" in cfg and "method" in cfg


def test_apply_config_reflects_manual_params():
    panel = DeblurPanel()
    panel.apply_config({
        "auto": False, "blur_type": "motion", "radius": 6.0, "angle": 45.0,
        "smooth": 40, "method": "rl", "tv_iterations": 200, "edge_taper": False,
    })
    cfg = panel.get_deblur_config()
    assert cfg["auto"] is False
    assert cfg["blur_type"] == "motion"
    assert abs(cfg["radius"] - 6.0) < 0.2
    assert abs(cfg["angle"] - 45.0) < 1.0
    assert cfg["method"] == "rl"
    assert cfg["edge_taper"] is False


def test_apply_config_auto():
    panel = DeblurPanel()
    panel.apply_config({"auto": True, "method": "wiener"})
    assert panel.get_deblur_config()["auto"] is True


def test_signal_emitted_on_change():
    panel = DeblurPanel()
    received = {}
    panel.deblur_params_changed.connect(lambda d: received.update(d))
    panel.set_auto(False)
    panel._emit_changed()
    assert "auto" in received


# --- Этап D: ретрансляция панели ---------------------------------------------

def test_deblur_panel_retranslate():
    from upscaler.ui import i18n
    panel = DeblurPanel()
    i18n.set_language("en")
    panel.retranslate()
    assert panel.preview_btn.text() == "Preview"
    i18n.set_language("ru")
    panel.retranslate()
    assert panel.preview_btn.text() == "Предпросмотр"
