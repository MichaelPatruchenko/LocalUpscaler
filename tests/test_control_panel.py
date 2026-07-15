"""Test ControlPanel pipeline config building and preset application (no GUI)."""

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from upscaler.ui.control_panel import ControlPanel

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolate_i18n(tmp_path, monkeypatch):
    """Изолировать настройки языка и синглтон переводчика для каждого теста.

    Зеркалит подход tests/test_i18n.py: не трогаем реальный settings.json и
    гарантируем, что язык всегда возвращается к "ru" между тестами, чтобы
    переключение языка в одном тесте не просачивалось в другие.
    """
    from upscaler import config
    from upscaler.ui import i18n
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(i18n, "_translator", None)
    yield
    i18n.set_language("ru")
    monkeypatch.setattr(i18n, "_translator", None)


def test_smartdeblur_checkbox_adds_deblur_config():
    panel = ControlPanel()
    panel._checkboxes["SmartDeblur"].setChecked(True)
    cfg = panel.get_pipeline_config()
    assert "deblur" in cfg
    assert cfg["deblur"]  # non-empty


def test_smartdeblur_unchecked_no_deblur():
    panel = ControlPanel()
    cfg = panel.get_pipeline_config()
    assert not cfg.get("deblur")


def test_apply_auto_config_reflects_deblur():
    panel = ControlPanel()
    panel.apply_auto_config({"scale": 4, "deblur": {"auto": True, "method": "wiener"}})
    assert panel._checkboxes["SmartDeblur"].isChecked()
    cfg = panel.get_pipeline_config()
    assert cfg["deblur"]["auto"] is True


def test_apply_auto_config_no_deblur_unchecks():
    panel = ControlPanel()
    panel._checkboxes["SmartDeblur"].setChecked(True)
    panel.apply_auto_config({"scale": 4})  # no deblur key
    assert not panel._checkboxes["SmartDeblur"].isChecked()
    assert not panel.get_pipeline_config().get("deblur")


def test_set_deblur_config_checks_box():
    panel = ControlPanel()
    panel.set_deblur_config({"auto": False, "blur_type": "motion", "radius": 4.0})
    assert panel._checkboxes["SmartDeblur"].isChecked()
    cfg = panel.get_pipeline_config()
    assert cfg["deblur"]["blur_type"] == "motion"


def test_is_deblur_enabled_reflects_checkbox():
    panel = ControlPanel()
    assert panel.is_deblur_enabled() is False
    panel._checkboxes["SmartDeblur"].setChecked(True)
    assert panel.is_deblur_enabled() is True


class TestGetPipelineConfig:
    def test_default_config_structure(self):
        """get_pipeline_config returns all required keys."""
        from upscaler.ui.control_panel import ControlPanel
        from PySide6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        panel = ControlPanel()
        config = panel.get_pipeline_config()
        assert "scale" in config
        assert "enhance_only" in config
        assert "denoise" in config
        assert "adjust" in config
        assert "upscale" in config
        assert "post" in config

    def test_default_scale_is_4(self):
        from upscaler.ui.control_panel import ControlPanel
        from PySide6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        panel = ControlPanel()
        config = panel.get_pipeline_config()
        assert config["scale"] == 4

    def test_enhance_only_default_false(self):
        from upscaler.ui.control_panel import ControlPanel
        from PySide6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        panel = ControlPanel()
        config = panel.get_pipeline_config()
        assert config["enhance_only"] is False


class TestApplyPreset:
    def test_apply_preset_sets_checkboxes(self):
        from upscaler.ui.control_panel import ControlPanel
        from PySide6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        panel = ControlPanel()
        preset = {
            "name": "Test",
            "scale": 2,
            "enhance_only": False,
            "pipeline": {
                "scale": {"plugin": "Lanczos"},
                "denoise": {"BM3D": {"sigma": 25}},
            },
        }
        panel._apply_preset(preset)
        config = panel.get_pipeline_config()
        assert config["scale"] == 2
        # Lanczos выбирается в комбо апскейлера
        assert panel.upscaler_combo.currentData() == "Lanczos"


def test_icedit_checkbox_adds_icedit_config():
    panel = ControlPanel()
    panel.set_icedit_config({"instruction": "remove watermark", "variant": "moe"})
    assert panel._checkboxes["ICEdit"].isChecked()
    cfg = panel.get_pipeline_config()
    assert cfg["icedit"]["instruction"] == "remove watermark"


def test_icedit_unchecked_no_icedit():
    panel = ControlPanel()
    cfg = panel.get_pipeline_config()
    assert not cfg.get("icedit")


def test_is_icedit_enabled_reflects_checkbox():
    panel = ControlPanel()
    assert panel.is_icedit_enabled() is False
    panel._checkboxes["ICEdit"].setChecked(True)
    assert panel.is_icedit_enabled() is True


def test_apply_auto_config_reflects_icedit():
    panel = ControlPanel()
    panel.apply_auto_config(
        {"scale": 4, "icedit": {"instruction": "add sunglasses", "variant": "moe"}})
    assert panel._checkboxes["ICEdit"].isChecked()
    assert panel.get_pipeline_config()["icedit"]["instruction"] == "add sunglasses"


def test_apply_auto_config_no_icedit_unchecks():
    panel = ControlPanel()
    panel._checkboxes["ICEdit"].setChecked(True)
    panel.apply_auto_config({"scale": 4})
    assert not panel._checkboxes["ICEdit"].isChecked()
    assert not panel.get_pipeline_config().get("icedit")


def test_max_refine_iterations_default_is_3():
    panel = ControlPanel()
    assert panel.get_max_refine_iterations() == 3


def test_set_max_refine_iterations_round_trip():
    panel = ControlPanel()
    panel.set_max_refine_iterations(7)
    assert panel.get_max_refine_iterations() == 7


def test_set_max_refine_iterations_clamps_high():
    panel = ControlPanel()
    panel.set_max_refine_iterations(99)
    assert panel.get_max_refine_iterations() == 10


def test_set_max_refine_iterations_clamps_low():
    panel = ControlPanel()
    panel.set_max_refine_iterations(0)
    assert panel.get_max_refine_iterations() == 1


def test_max_iter_changed_signal_emits_value():
    panel = ControlPanel()
    received = []
    panel.max_iter_changed.connect(received.append)
    panel.set_max_refine_iterations(5)
    assert received and received[-1] == 5


def test_ai_assistant_checkbox_default_on_and_toggles():
    panel = ControlPanel()
    assert panel.is_ai_assistant_enabled() is True
    panel.set_ai_assistant_enabled(False)
    assert panel.is_ai_assistant_enabled() is False


def test_face_checkbox_default_on_and_config():
    panel = ControlPanel()
    assert panel.is_face_enabled() is True
    cfg = panel.get_pipeline_config()
    assert cfg.get("face", {}).get("enabled") is True


def test_face_checkbox_off_no_face_config():
    panel = ControlPanel()
    panel._checkboxes["CodeFormer"].setChecked(False)
    cfg = panel.get_pipeline_config()
    assert not cfg.get("face")


# --- Этап 2: ручной порядок шагов -------------------------------------------
from upscaler.engine.pipeline import PipelineExecutor


def test_order_list_shows_default_order():
    panel = ControlPanel()
    assert panel.get_order() == list(PipelineExecutor.DEFAULT_ORDER)
    assert "order" not in panel.get_pipeline_config()


def test_move_step_creates_user_order():
    panel = ControlPanel()
    panel.order_list.setCurrentRow(1)
    panel._move_order_item(-1)
    cfg = panel.get_pipeline_config()
    assert cfg["order"] == panel.get_order()
    assert cfg["order"][0] == PipelineExecutor.DEFAULT_ORDER[1]
    assert cfg["order"][1] == PipelineExecutor.DEFAULT_ORDER[0]


def test_reset_clears_user_order():
    panel = ControlPanel()
    panel.order_list.setCurrentRow(1)
    panel._move_order_item(-1)
    panel._reset_order()
    assert panel.get_order() == list(PipelineExecutor.DEFAULT_ORDER)
    assert "order" not in panel.get_pipeline_config()


def test_apply_auto_config_shows_llm_order_without_marking_user():
    panel = ControlPanel()
    panel.apply_auto_config({"scale": 4, "order": ["upscale", "denoise"]})
    assert panel.get_order()[0] == "upscale"
    assert panel.get_order()[1] == "denoise"
    assert panel._user_order == []
    # прежнее поведение: LLM-порядок попадает в конфиг через _order_override
    assert panel.get_pipeline_config()["order"] == ["upscale", "denoise"]


def test_user_order_beats_llm_override():
    panel = ControlPanel()
    panel.apply_auto_config({"scale": 4, "order": ["upscale", "denoise"]})
    panel.order_list.setCurrentRow(1)
    panel._move_order_item(-1)  # пользователь меняет порядок поверх LLM
    cfg = panel.get_pipeline_config()
    assert cfg["order"] == panel.get_order()
    assert cfg["order"][0] == "denoise"


def test_preset_resets_order_to_default():
    panel = ControlPanel()
    panel.order_list.setCurrentRow(1)
    panel._move_order_item(-1)
    panel._apply_preset({"scale": 4})
    assert panel.get_order() == list(PipelineExecutor.DEFAULT_ORDER)
    assert "order" not in panel.get_pipeline_config()


def test_move_out_of_bounds_is_noop():
    panel = ControlPanel()
    panel.order_list.setCurrentRow(0)
    panel._move_order_item(-1)  # уже наверху
    assert panel.get_order() == list(PipelineExecutor.DEFAULT_ORDER)
    panel.order_list.setCurrentRow(-1)  # ничего не выбрано
    panel._move_order_item(1)
    assert panel.get_order() == list(PipelineExecutor.DEFAULT_ORDER)


def test_face_override_with_regions_reaches_config():
    panel = ControlPanel()
    panel.set_face_config({"enabled": True, "fidelity": 0.6,
                           "regions": [[0.1, 0.2, 0.3, 0.4]]})
    cfg = panel.get_pipeline_config()
    assert cfg["face"]["regions"] == [[0.1, 0.2, 0.3, 0.4]]


def test_set_face_config_preserves_regions_when_absent():
    panel = ControlPanel()
    panel.set_face_config({"enabled": True, "fidelity": 0.6,
                           "regions": [[0.1, 0.2, 0.3, 0.4]]})
    panel.set_face_config({"enabled": True, "fidelity": 0.5})  # авто/LLM без зон
    cfg = panel.get_pipeline_config()
    assert cfg["face"]["regions"] == [[0.1, 0.2, 0.3, 0.4]]


def test_set_face_config_explicit_empty_regions_clears():
    panel = ControlPanel()
    panel.set_face_config({"enabled": True, "regions": [[0.1, 0.2, 0.3, 0.4]]})
    panel.set_face_config({"enabled": True, "regions": []})
    assert panel.get_pipeline_config()["face"].get("regions", []) == []


# --- Этап 4: чекбокс авто-предуменьшения ------------------------------------

def test_predownscale_checkbox_default_on():
    panel = ControlPanel()
    assert panel.is_predownscale_enabled() is True
    assert panel.get_pipeline_config()["predownscale_enabled"] is True


def test_predownscale_checkbox_off_in_config():
    panel = ControlPanel()
    panel.set_predownscale_enabled(False)
    assert panel.get_pipeline_config()["predownscale_enabled"] is False


def test_apply_auto_config_stores_predownscale_override():
    panel = ControlPanel()
    panel.apply_auto_config({"scale": 4, "predownscale": {"factor": 0.5}})
    cfg = panel.get_pipeline_config()
    assert cfg["predownscale"] == {"factor": 0.5}


def test_predownscale_override_suppressed_when_disabled():
    panel = ControlPanel()
    panel.apply_auto_config({"scale": 4, "predownscale": {"factor": 0.5}})
    panel.set_predownscale_enabled(False)
    assert "predownscale" not in panel.get_pipeline_config()


def test_preset_clears_predownscale_override():
    panel = ControlPanel()
    panel.apply_auto_config({"scale": 4, "predownscale": {"factor": 0.5}})
    panel._apply_preset({"scale": 4})
    assert "predownscale" not in panel.get_pipeline_config()


def test_predownscale_toggle_signal():
    panel = ControlPanel()
    got = []
    panel.predownscale_toggled.connect(lambda v: got.append(v))
    panel.set_predownscale_enabled(False)
    assert got == [False]


# --- Этап 6A: чекбокс смешивания вариантов ------------------------------------

def test_blend_checkbox_default_off():
    panel = ControlPanel()
    assert panel.is_blend_enabled() is False
    assert "blend" not in panel.get_pipeline_config()


def test_blend_checkbox_on_adds_config():
    panel = ControlPanel()
    panel.set_blend_enabled(True)
    assert panel.get_pipeline_config()["blend"] == {"enabled": True}


# --- Этап B: type-aware слайдеры в apply_auto_config -------------------------


def test_apply_auto_config_integer_radius_not_scaled():
    panel = ControlPanel()
    panel.apply_auto_config({
        "scale": 4,
        "adjust": {"Clarity": {"strength": 0.3, "radius": 60}},
    })
    assert panel._sliders["Clarity.radius"].value() == 60
    assert panel._sliders["Clarity.strength"].value() == 30


def test_apply_auto_config_negative_number_param():
    panel = ControlPanel()
    panel.apply_auto_config({
        "scale": 4,
        "adjust": {"Vibrance": {"strength": -0.5}},
    })
    assert panel._sliders["Vibrance.strength"].value() == -50


def test_apply_auto_config_shadows_highlights_pair():
    panel = ControlPanel()
    panel.apply_auto_config({
        "scale": 4,
        "adjust": {"Shadows/Highlights": {"shadows": 0.4, "highlights": 0.2,
                                          "radius": 30}},
    })
    assert panel._sliders["Shadows/Highlights.shadows"].value() == 40
    assert panel._sliders["Shadows/Highlights.highlights"].value() == 20
    assert panel._sliders["Shadows/Highlights.radius"].value() == 30


def test_apply_auto_config_denoise_int_param_not_scaled():
    panel = ControlPanel()
    panel.apply_auto_config({
        "scale": 4,
        "denoise": {"SCUNet": {"strength": 0.4, "tile_size": 256}},
    })
    assert panel._sliders["SCUNet.tile_size"].value() == 256
    assert panel._sliders["SCUNet.strength"].value() == 40


# --- Этап C: комбобокс апскейлера ---------------------------------------------


def test_upscaler_combo_default_real_esrgan():
    panel = ControlPanel()
    assert panel.upscaler_combo.currentData() == "Real-ESRGAN"
    cfg = panel.get_pipeline_config()
    assert cfg["upscale"]["plugin"] == "Real-ESRGAN"
    assert cfg["upscale"]["scale"] == cfg["scale"]


def test_upscaler_combo_none_skips_upscale():
    panel = ControlPanel()
    panel.upscaler_combo.setCurrentIndex(0)  # «(Нет)»
    assert panel.get_pipeline_config()["upscale"] == {}


def test_upscalers_not_in_checkboxes():
    panel = ControlPanel()
    assert "Real-ESRGAN" not in panel._checkboxes
    assert "Lanczos" not in panel._checkboxes


def test_apply_auto_config_selects_upscaler_in_combo():
    panel = ControlPanel()
    panel.apply_auto_config({"scale": 4,
                             "upscale": {"plugin": "HAT-S", "scale": 4}})
    assert panel.upscaler_combo.currentData() == "HAT-S"


def test_apply_auto_config_without_upscale_selects_none():
    panel = ControlPanel()
    panel.apply_auto_config({"scale": 4})
    assert panel.upscaler_combo.currentData() is None


# --- Этап C: секции и видимость параметров -------------------------------------


def test_sections_exist_with_default_states():
    panel = ControlPanel()
    for key in ("preset_scale", "upscaler", "denoise", "adjust",
                "adjust_tone", "adjust_color", "adjust_detail",
                "restore", "edit", "blend", "ai", "order"):
        assert key in panel._sections, key
    states = panel.get_section_states()
    assert states["preset_scale"] is True
    assert states["upscaler"] is True
    assert states["restore"] is True
    assert states["denoise"] is False
    assert states["order"] is False


def test_set_section_states_applies():
    panel = ControlPanel()
    panel.set_section_states({"denoise": True, "upscaler": False})
    assert panel._sections["denoise"].is_expanded() is True
    assert panel._sections["upscaler"].is_expanded() is False


def test_section_toggled_signal():
    panel = ControlPanel()
    got = []
    panel.section_toggled.connect(lambda k, v: got.append((k, v)))
    panel._sections["denoise"].set_expanded(True)
    assert ("denoise", True) in got


def test_params_hidden_until_checkbox_on():
    panel = ControlPanel()
    box = panel._param_boxes["Clarity"]
    assert box.isHidden()
    panel._checkboxes["Clarity"].setChecked(True)
    assert not box.isHidden()
    panel._checkboxes["Clarity"].setChecked(False)
    assert box.isHidden()


def test_adjusters_grouped_into_subsections():
    panel = ControlPanel()
    tone = panel._sections["adjust_tone"]
    color = panel._sections["adjust_color"]
    detail = panel._sections["adjust_detail"]
    # чекбокс лежит внутри своей подсекции
    assert panel._checkboxes["Auto Levels"] in tone.findChildren(
        type(panel._checkboxes["Auto Levels"]))
    assert panel._checkboxes["Vibrance"] in color.findChildren(
        type(panel._checkboxes["Vibrance"]))
    assert panel._checkboxes["Clarity"] in detail.findChildren(
        type(panel._checkboxes["Clarity"]))


def test_config_behavior_unchanged_after_reorg():
    panel = ControlPanel()
    panel._checkboxes["Clarity"].setChecked(True)
    cfg = panel.get_pipeline_config()
    assert "Clarity" in cfg["adjust"]
    assert cfg["upscale"]["plugin"] == "Real-ESRGAN"


# --- Этап D: ретрансляция панели ---------------------------------------------

def test_control_panel_retranslate_switches_language():
    from upscaler.ui import i18n
    panel = ControlPanel()
    i18n.set_language("en")
    panel.retranslate()
    # заголовок секции коррекции на английском
    assert panel._sections["adjust"].header_btn.text() == "Correction"
    i18n.set_language("ru")
    panel.retranslate()
    assert panel._sections["adjust"].header_btn.text() == "Коррекция"
