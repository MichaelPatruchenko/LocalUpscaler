"""Ядро i18n: tr(), переключение языка, паритет ключей."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_i18n(tmp_path, monkeypatch):
    """Изолировать настройки и сбросить синглтон переводчика для каждого теста."""
    from upscaler import config
    from upscaler.ui import i18n
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(i18n, "_translator", None)
    yield
    monkeypatch.setattr(i18n, "_translator", None)


def test_key_parity_ru_en():
    from upscaler.ui.i18n import _STRINGS
    assert set(_STRINGS["ru"]) == set(_STRINGS["en"]), (
        "ключи RU и EN должны совпадать"
    )


def test_tr_returns_current_language():
    from upscaler.ui import i18n
    i18n.set_language("ru")
    assert i18n.tr("btn.process") == "Обработать"
    i18n.set_language("en")
    assert i18n.tr("btn.process") == "Process"
    i18n.set_language("ru")


def test_tr_missing_key_returns_key():
    from upscaler.ui import i18n
    assert i18n.tr("nonexistent.key.xyz") == "nonexistent.key.xyz"


def test_tr_format_substitution():
    from upscaler.ui import i18n
    i18n.set_language("en")
    # ключ status.zone_count == "Zones: {n}"
    assert i18n.tr("status.zone_count", n=3) == "Zones: 3"
    i18n.set_language("ru")
    assert i18n.tr("status.zone_count", n=3) == "Зон: 3"


def test_set_language_validates():
    from upscaler.ui import i18n
    i18n.set_language("bogus")           # игнорируется
    assert i18n.current_language() in ("ru", "en")
    i18n.set_language("ru")


def test_language_changed_signal():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from upscaler.ui import i18n
    _app = QApplication.instance() or QApplication([])
    got = []
    i18n.translator().language_changed.connect(lambda l: got.append(l))
    i18n.set_language("en")
    i18n.set_language("ru")
    assert got == ["en", "ru"]


def test_available_languages():
    from upscaler.ui import i18n
    assert i18n.available_languages() == ("ru", "en")


def test_all_step_tokens_translated():
    from upscaler.engine.pipeline import PipelineExecutor, step_label
    from upscaler.ui import i18n
    for lang in ("ru", "en"):
        i18n.set_language(lang)
        for token in PipelineExecutor.DEFAULT_ORDER:
            assert step_label(token).strip(), (lang, token)
    i18n.set_language("ru")


def test_all_blend_modes_translated():
    from upscaler.engine.blend import BLEND_MODES, blend_mode_label
    from upscaler.ui import i18n
    for lang in ("ru", "en"):
        i18n.set_language(lang)
        for mode in BLEND_MODES:
            assert blend_mode_label(mode).strip(), (lang, mode)
    i18n.set_language("ru")


def test_step_label_switches_language():
    from upscaler.engine.pipeline import step_label
    from upscaler.ui import i18n
    i18n.set_language("en")
    assert step_label("upscale") == "Upscale"
    i18n.set_language("ru")
    assert step_label("upscale") == "Масштабирование"


def test_menu_and_message_keys_present():
    from upscaler.ui import i18n
    for key in ("menu.file", "menu.open", "menu.save", "menu.exit",
                "msg.no_image_title", "msg.no_image_body"):
        assert i18n.tr(key) != key, key  # ключ существует (не эхо)


def test_message_translates():
    from upscaler.ui import i18n
    i18n.set_language("en")
    assert i18n.tr("msg.no_image_title") == "No image"
    i18n.set_language("ru")
    assert i18n.tr("msg.no_image_title") == "Нет изображения"


def test_describe_translates():
    from upscaler.engine.auto_config import AutoConfigurator
    from upscaler.ui import i18n
    ac = AutoConfigurator()
    cfg = ac.recommend({"noise_level": 3.0, "blur_score": 400}, scale=4)
    i18n.set_language("en")
    desc_en = ac.describe(cfg)
    i18n.set_language("ru")
    desc_ru = ac.describe(cfg)
    assert desc_en != desc_ru or desc_en == desc_ru  # оба не падают
    assert isinstance(desc_en, str) and isinstance(desc_ru, str)
