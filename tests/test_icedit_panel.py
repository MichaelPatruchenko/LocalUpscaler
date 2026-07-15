import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from upscaler.ui.icedit_panel import ICEditPanel

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


def test_config_has_expected_keys():
    panel = ICEditPanel()
    panel.set_instruction("make the sky blue")
    cfg = panel.get_icedit_config()
    assert cfg["instruction"] == "make the sky blue"
    assert cfg["variant"] in ("moe", "normal")
    assert "steps" in cfg and "guidance" in cfg and "offload" in cfg


def test_signal_emitted_on_change():
    panel = ICEditPanel()
    received = {}
    panel.icedit_params_changed.connect(lambda d: received.update(d))
    panel.set_instruction("add a hat")
    panel._emit_changed()
    assert received.get("instruction") == "add a hat"


def test_apply_config_reflects_instruction_and_variant():
    panel = ICEditPanel()
    panel.apply_config({"instruction": "remove logo", "variant": "normal",
                        "steps": 20})
    cfg = panel.get_icedit_config()
    assert cfg["instruction"] == "remove logo"
    assert cfg["variant"] == "normal"
    assert cfg["steps"] == 20


# --- Этап 5: официальные дефолты в панели -------------------------------------

def test_panel_defaults_normal_variant_guidance_50():
    panel = ICEditPanel()
    cfg = panel.get_icedit_config()
    assert cfg["variant"] == "normal"
    assert cfg["guidance"] == 50.0


# --- Этап D: ретрансляция панели ---------------------------------------------

def test_icedit_panel_retranslate():
    from upscaler.ui import i18n
    panel = ICEditPanel()
    i18n.set_language("en")
    panel.retranslate()
    assert panel.preview_btn.text() == "Preview"
    i18n.set_language("ru")
    panel.retranslate()
    assert panel.preview_btn.text() == "Предпросмотр"
