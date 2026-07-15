import pytest

from upscaler.engine.auto_config import AutoConfigurator


@pytest.fixture(autouse=True)
def _pin_russian(tmp_path, monkeypatch):
    """describe() зависит от языка; пиним RU и изолируем настройки,
    чтобы persisted language из реального settings.json не ломал ассерты."""
    from upscaler import config
    from upscaler.ui import i18n
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(i18n, "_translator", None)
    i18n.set_language("ru")
    yield
    monkeypatch.setattr(i18n, "_translator", None)


def test_blurry_image_enables_deblur():
    cfg = AutoConfigurator().recommend(
        {"blur_score": 40.0, "noise_level": 3.0, "blur_assessment": {
            "needs_deblur": True, "blur_type": "gaussian", "radius": 2.0,
            "angle": 0.0, "smooth": 25.0, "method": "wiener", "sharpness": 10.0
        }}, scale=4)
    assert "deblur" in cfg
    assert cfg["deblur"].get("auto") is False


def test_sharp_image_skips_deblur():
    cfg = AutoConfigurator().recommend(
        {"blur_score": 800.0, "noise_level": 3.0}, scale=4)
    assert not cfg.get("deblur")


def test_describe_mentions_deblur():
    ac = AutoConfigurator()
    cfg = ac.recommend({"blur_score": 40.0, "noise_level": 3.0, "blur_assessment": {
        "needs_deblur": True, "blur_type": "gaussian", "radius": 2.0,
        "angle": 0.0, "smooth": 25.0, "method": "wiener", "sharpness": 10.0
    }}, scale=4)
    assert "Деблюр" in ac.describe(cfg)


def test_classify_photo_vs_graphic():
    ac = AutoConfigurator()
    photo = {"gray_entropy": 7.4, "edge_density": 0.06, "saturation_mean": 60,
             "glcm_homogeneity": 0.35}
    graphic = {"gray_entropy": 3.0, "edge_density": 0.22, "saturation_mean": 10,
               "glcm_homogeneity": 0.85}
    assert ac._classify_content(photo) == "photo"
    assert ac._classify_content(graphic) in ("illustration", "graphic")


def test_low_noise_photo_keeps_detail_no_denoise():
    ac = AutoConfigurator()
    cfg = ac.recommend({"noise_level": 7.0, "blur_score": 400,
                        "gray_entropy": 7.2, "edge_density": 0.06,
                        "saturation_mean": 55, "glcm_homogeneity": 0.35}, scale=4)
    assert cfg["denoise"] == {}  # noise<=8 on a photo -> no softening


def test_photo_prefers_fidelity_upscaler():
    ac = AutoConfigurator()
    up = ac._select_upscaler(noise=4, detail=18, megapixels=1.5, scale=4,
                             freq_high=0.1, wavelet_detail_ratio=0.0,
                             fractal_dimension=1.4, content="photo")
    assert up == "Real-ESRGAN"


def test_photo_clean_high_detail_picks_hat_s():
    ac = AutoConfigurator()
    up = ac._select_upscaler(noise=4, detail=26, megapixels=1.5, scale=4,
                             freq_high=0.1, wavelet_detail_ratio=0.0,
                             fractal_dimension=1.4, content="photo")
    assert up == "HAT-S"


def test_illustration_routes_to_swinir():
    ac = AutoConfigurator()
    up = ac._select_upscaler(noise=4, detail=18, megapixels=1.5, scale=4,
                             freq_high=0.1, wavelet_detail_ratio=0.0,
                             fractal_dimension=1.4, content="illustration")
    assert up == "SwinIR"


def test_graphic_complex_texture_routes_to_dat():
    ac = AutoConfigurator()
    up = ac._select_upscaler(noise=4, detail=18, megapixels=1.5, scale=4,
                             freq_high=0.1, wavelet_detail_ratio=0.0,
                             fractal_dimension=1.9, content="graphic")
    assert up == "DAT"


def test_grayscale_no_saturation():
    ac = AutoConfigurator()
    cfg = ac.recommend({"is_grayscale": True, "saturation_mean": 5,
                        "noise_level": 2, "blur_score": 500}, scale=4)
    assert "Saturation" not in cfg["adjust"]
    assert "Auto Color" not in cfg["adjust"]


def test_faces_enable_face_restore():
    ac = AutoConfigurator()
    cfg = ac.recommend({"has_faces": True, "face_count": 2,
                        "noise_level": 3, "blur_score": 400}, scale=4)
    assert cfg.get("face", {}).get("enabled") is True
    assert 0.0 <= cfg["face"]["fidelity"] <= 1.0


def test_no_faces_no_face_restore():
    ac = AutoConfigurator()
    cfg = ac.recommend({"has_faces": False, "noise_level": 3,
                        "blur_score": 400}, scale=4)
    assert not cfg.get("face")


def _analysis(**over):
    a = {
        "noise_level": 5.0, "blur_score": 80.0, "brightness": 128.0,
        "contrast": 50.0, "detail_level": 20.0, "color_cast": (0, 0, 0),
        "dynamic_range": 200.0, "resolution": (800, 600),
    }
    a.update(over)
    return a


def test_autoconfig_enables_deblur_from_assessment():
    cfg = AutoConfigurator().recommend(_analysis(blur_assessment={
        "needs_deblur": True, "blur_type": "focus", "radius": 5.0,
        "angle": 0.0, "smooth": 35.0, "method": "tv",
    }), scale=2)
    assert cfg.get("deblur")
    assert cfg["deblur"]["method"] == "tv"
    assert cfg["deblur"]["blur_type"] == "focus"
    assert cfg["deblur"]["auto"] is False
    assert cfg["deblur"]["edge_taper"] is True


def test_autoconfig_skips_deblur_when_not_needed():
    cfg = AutoConfigurator().recommend(_analysis(blur_assessment={
        "needs_deblur": False, "blur_type": "gaussian", "radius": 1.0,
        "angle": 0.0, "smooth": 25.0, "method": "wiener",
    }), scale=2)
    assert not cfg.get("deblur")


def test_autoconfig_skips_deblur_without_assessment():
    cfg = AutoConfigurator().recommend(_analysis(), scale=2)
    assert not cfg.get("deblur")


# --- Этап 1: алгоритмический fidelity CodeFormer ---------------------------

def _face_analysis(sharp, noise, min_px=120, res=(2000, 1500)):
    return {"has_faces": True, "face_count": 1, "face_sharpness": sharp,
            "face_noise": noise, "face_min_px": min_px, "resolution": res,
            "blur_score": 400, "noise_level": 3.0}


def test_face_fidelity_monotonic_in_sharpness():
    ac = AutoConfigurator()
    lo = ac._face_params(_face_analysis(sharp=0.1, noise=5.0))["fidelity"]
    hi = ac._face_params(_face_analysis(sharp=0.9, noise=5.0))["fidelity"]
    assert hi > lo


def test_face_fidelity_drops_with_noise():
    ac = AutoConfigurator()
    clean = ac._face_params(_face_analysis(sharp=0.6, noise=2.0))["fidelity"]
    noisy = ac._face_params(_face_analysis(sharp=0.6, noise=25.0))["fidelity"]
    assert noisy < clean


def test_face_fidelity_bounds():
    ac = AutoConfigurator()
    worst = ac._face_params(_face_analysis(sharp=0.0, noise=50.0, min_px=20))
    best = ac._face_params(_face_analysis(sharp=1.0, noise=0.0))
    assert worst["fidelity"] >= 0.3
    assert best["fidelity"] <= 0.9


def test_face_min_px_scales_with_resolution():
    ac = AutoConfigurator()
    small = ac._face_params(_face_analysis(0.5, 5.0, res=(800, 600)))
    large = ac._face_params(_face_analysis(0.5, 5.0, res=(10000, 8000)))
    assert 16 <= small["min_face_px"] <= 96
    assert large["min_face_px"] == 96  # кламп сверху
    assert small["min_face_px"] < large["min_face_px"]


def test_recommend_uses_computed_face_params():
    ac = AutoConfigurator()
    cfg = ac.recommend(_face_analysis(sharp=0.1, noise=20.0), scale=4)
    assert cfg["face"]["enabled"] is True
    assert cfg["face"]["fidelity"] < 0.7  # деградированное лицо -> сильнее реставрация
    assert cfg["face"]["upscale_background"] is False


def test_describe_shows_computed_fidelity():
    ac = AutoConfigurator()
    cfg = ac.recommend(_face_analysis(sharp=0.9, noise=1.0), scale=4)
    desc = ac.describe(cfg)
    assert f"w={cfg['face']['fidelity']}" in desc


# --- Этап 1: полный deblur-конфиг из оценки ---------------------------------

def test_deblur_config_passes_extra_params():
    cfg = AutoConfigurator().recommend(
        {"blur_score": 40.0, "noise_level": 3.0, "blur_assessment": {
            "needs_deblur": True, "blur_type": "focus", "radius": 9.0,
            "angle": 0.0, "smooth": 42.5, "method": "tv", "sharpness": 0.2,
            "tv_iterations": 180, "edge_feather": 8.0,
            "correction_strength": 24.0, "edge_taper": True,
        }}, scale=4)
    d = cfg["deblur"]
    assert d["tv_iterations"] == 180
    assert d["edge_feather"] == 8.0
    assert d["correction_strength"] == 24.0
    assert d["smooth"] == 42.5
    assert d["method"] == "tv"


def test_deblur_config_omits_absent_extras():
    cfg = AutoConfigurator().recommend(
        {"blur_score": 40.0, "noise_level": 3.0, "blur_assessment": {
            "needs_deblur": True, "blur_type": "gaussian", "radius": 2.0,
            "angle": 0.0, "smooth": 25.0, "method": "wiener", "sharpness": 0.3,
        }}, scale=4)
    d = cfg["deblur"]
    assert "tv_iterations" not in d
    assert "edge_feather" not in d
    assert "correction_strength" not in d


# --- Этап 2: describe() упоминает нестандартный порядок ---------------------

def test_describe_mentions_custom_order():
    ac = AutoConfigurator()
    cfg = ac.recommend({"noise_level": 3.0, "blur_score": 400}, scale=4)
    cfg["order"] = ["upscale", "denoise"]
    desc = ac.describe(cfg)
    assert "Порядок" in desc
    assert "Масштабирование" in desc


def test_describe_no_order_line_by_default():
    ac = AutoConfigurator()
    cfg = ac.recommend({"noise_level": 3.0, "blur_score": 400}, scale=4)
    assert "Порядок" not in ac.describe(cfg)


# --- Этап 4: авто-предуменьшение --------------------------------------------

def _soft_hires_analysis(factor=0.5, res=(3840, 2160)):
    return {"noise_level": 3.0, "blur_score": 400,
            "effective_downscale_factor": factor,
            "resolution": res,
            "megapixels": res[0] * res[1] / 1e6}


def test_predownscale_added_for_soft_hires():
    cfg = AutoConfigurator().recommend(_soft_hires_analysis(), scale=4)
    assert cfg["predownscale"] == {"factor": 0.5}


def test_predownscale_respects_allow_flag():
    cfg = AutoConfigurator().recommend(_soft_hires_analysis(), scale=4,
                                       allow_predownscale=False)
    assert "predownscale" not in cfg


def test_predownscale_skipped_for_sharp():
    cfg = AutoConfigurator().recommend(_soft_hires_analysis(factor=1.0),
                                       scale=4)
    assert "predownscale" not in cfg


def test_predownscale_min_side_floor_skips_small():
    cfg = AutoConfigurator().recommend(
        _soft_hires_analysis(res=(1280, 720)), scale=4)
    assert "predownscale" not in cfg  # 720*0.75=540 < 720


def test_predownscale_floor_bumps_factor():
    # 1500*0.33=495 < 720, но 1500*0.5=750 >= 720 -> фактор поднят до 0.5
    cfg = AutoConfigurator().recommend(
        _soft_hires_analysis(factor=0.33, res=(2000, 1500)), scale=4)
    assert cfg["predownscale"] == {"factor": 0.5}


def test_predownscale_rescales_deblur_radius():
    analysis = _soft_hires_analysis()
    analysis["blur_assessment"] = {
        "needs_deblur": True, "blur_type": "gaussian", "radius": 6.0,
        "angle": 0.0, "smooth": 30.0, "method": "wiener", "sharpness": 0.2,
    }
    cfg = AutoConfigurator().recommend(analysis, scale=4)
    assert cfg["deblur"]["radius"] == 3.0  # 6.0 * 0.5


def test_predownscale_affects_upscaler_choice():
    # 12 МП обычно форсируют Real-ESRGAN; после сжатия 0.5 -> 3 МП,
    # чистый детальный кадр выбирает HAT-S.
    analysis = {
        "noise_level": 3.0, "blur_score": 800, "detail_level": 26.0,
        "effective_downscale_factor": 0.5, "resolution": (4243, 2828),
        "megapixels": 12.0, "gray_entropy": 7.4, "edge_density": 0.06,
        "saturation_mean": 60, "glcm_homogeneity": 0.35,
        "fft_high_energy_ratio": 0.3, "wavelet_detail_ratio": 0.35,
        "fractal_dimension": 1.4,
    }
    cfg = AutoConfigurator().recommend(analysis, scale=4)
    assert cfg["predownscale"] == {"factor": 0.5}
    assert cfg["upscale"]["plugin"] == "HAT-S"


def test_describe_mentions_predownscale():
    ac = AutoConfigurator()
    cfg = ac.recommend(_soft_hires_analysis(), scale=4)
    assert "Уменьшение: до 50%" in ac.describe(cfg)


# --- Этап B: авто-правила новых корректоров ----------------------------------


def _photo_base(**kw):
    d = {"noise_level": 3.0, "blur_score": 400, "gray_entropy": 7.2,
         "edge_density": 0.06, "saturation_mean": 60,
         "glcm_homogeneity": 0.35, "brightness": 120, "contrast": 50,
         "dynamic_range": 255}
    d.update(kw)
    return d


def test_dehaze_rule_fires_on_haze():
    cfg = AutoConfigurator().recommend(_photo_base(haze_level=0.7), scale=4)
    assert "Dehaze" in cfg["adjust"]
    assert 0.0 < cfg["adjust"]["Dehaze"]["strength"] <= 0.8


def test_dehaze_rule_skipped_when_clear():
    cfg = AutoConfigurator().recommend(_photo_base(haze_level=0.2), scale=4)
    assert "Dehaze" not in cfg["adjust"]


def test_shadows_highlights_rule():
    cfg = AutoConfigurator().recommend(
        _photo_base(shadow_mass=0.5, highlight_clip=0.05), scale=4)
    sh = cfg["adjust"]["Shadows/Highlights"]
    assert sh["shadows"] > 0.2
    assert sh["highlights"] > 0.2


def test_vibrance_replaces_saturation_rule():
    cfg = AutoConfigurator().recommend(
        _photo_base(saturation_mean=25, is_grayscale=False), scale=4)
    assert "Vibrance" in cfg["adjust"]
    assert "Saturation" not in cfg["adjust"]


def test_white_balance_strong_cast():
    cfg = AutoConfigurator().recommend(
        _photo_base(color_cast=(15.0, 0.0, -10.0)), scale=4)
    assert "White Balance" in cfg["adjust"]


def test_auto_color_moderate_cast_only():
    cfg = AutoConfigurator().recommend(
        _photo_base(color_cast=(10.0, 0.0, -5.0)), scale=4)
    assert "Auto Color" in cfg["adjust"]
    assert "White Balance" not in cfg["adjust"]


def test_auto_levels_on_narrow_range():
    cfg = AutoConfigurator().recommend(
        _photo_base(dynamic_range=140, contrast=30), scale=4)
    assert "Auto Levels" in cfg["adjust"]


def test_auto_levels_suppresses_auto_tone_on_narrow_range():
    cfg = AutoConfigurator().recommend(
        _photo_base(dynamic_range=140, contrast=30), scale=4)
    assert "Auto Levels" in cfg["adjust"]
    assert "Auto Tone" not in cfg["adjust"]


def test_auto_tone_still_fires_on_skew_despite_narrow_range():
    cfg = AutoConfigurator().recommend(
        _photo_base(dynamic_range=140, contrast=30, hist_skewness=2.0),
        scale=4)
    assert "Auto Tone" in cfg["adjust"]
    assert "Auto Levels" in cfg["adjust"]


def test_clarity_on_flat_photo():
    cfg = AutoConfigurator().recommend(
        _photo_base(glcm_contrast=20.0), scale=4)
    assert cfg["adjust"].get("Clarity", {}).get("strength") == 0.2


def test_skin_smooth_on_noisy_faces():
    cfg = AutoConfigurator().recommend(
        _photo_base(has_faces=True, face_count=1, face_sharpness=0.4,
                    face_noise=12.0, face_min_px=120,
                    resolution=(2000, 1500)), scale=4)
    assert "Skin Smooth" in cfg["adjust"]
    assert cfg["adjust"]["Skin Smooth"]["strength"] <= 0.6


def test_stylistic_adjusters_never_auto():
    cfg = AutoConfigurator().recommend(
        _photo_base(haze_level=0.8, shadow_mass=0.6, vignette_strength=0.5),
        scale=4)
    for name in ("Dodge & Burn", "Split Toning", "Optics"):
        assert name not in cfg["adjust"], name
