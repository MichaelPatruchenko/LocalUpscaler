import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from upscaler.ui.settings_dialog import SettingsDialog

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


def test_prefer_gpu_denoise_roundtrip():
    dlg = SettingsDialog({"prefer_gpu_denoise": True})
    assert dlg.get_settings()["prefer_gpu_denoise"] is True
    dlg.gpu_denoise_cb.setChecked(False)
    assert dlg.get_settings()["prefer_gpu_denoise"] is False


def test_prefer_gpu_denoise_defaults_true_when_absent():
    dlg = SettingsDialog({})
    assert dlg.gpu_denoise_cb.isChecked() is True


# --- Этап D: ретрансляция диалога ---------------------------------------------

def test_settings_dialog_retranslate():
    from upscaler.ui import i18n
    dlg = SettingsDialog({})
    i18n.set_language("en")
    dlg.retranslate()
    assert dlg.gpu_denoise_cb.text() == (
        "Replace BM3D with a GPU denoiser (SCUNet) when a GPU is available")
    i18n.set_language("ru")
    dlg.retranslate()
    assert dlg.gpu_denoise_cb.text() == (
        "Заменять BM3D на GPU-шумодав (SCUNet) при наличии видеокарты")
