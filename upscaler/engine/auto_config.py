"""Automatic pipeline configuration based on expanded image analysis."""
import logging

log = logging.getLogger(__name__)


class AutoConfigurator:
    """Recommends pipeline settings based on SourceAnalyzer output.

    Uses expanded metrics (GLCM, frequency, color temperature, histogram
    shape, fractal dimension, etc.) for smarter decisions.
    """

    def recommend(self, analysis: dict, scale: int = 4,
                  enhance_only: bool = False,
                  allow_predownscale: bool = True) -> dict:
        """Build optimal pipeline config from image analysis.

        Returns a config dict ready for PipelineExecutor.
        """
        # --- Basic metrics ---
        noise = analysis.get("noise_level", 0)
        blur = analysis.get("blur_score", 500)
        brightness = analysis.get("brightness", 128)
        contrast = analysis.get("contrast", 50)
        detail = analysis.get("detail_level", 20)
        dynamic_range = analysis.get("dynamic_range", 255)
        megapixels = analysis.get("megapixels", 1.0)
        color_cast = analysis.get("color_cast", (0, 0, 0))
        is_grayscale = analysis.get("is_grayscale", False)

        # --- Expanded metrics ---
        # Histogram shape
        hist_skewness = analysis.get("hist_skewness", 0.0)
        hist_kurtosis = analysis.get("hist_kurtosis", 0.0)

        # GLCM texture
        glcm_homogeneity = analysis.get("glcm_homogeneity", 0.5)

        # Edges / structure
        edge_density = analysis.get("edge_density", 0.05)
        fractal_dimension = analysis.get("fractal_dimension", 1.5)

        # Color
        saturation_mean = analysis.get("saturation_mean", 80)
        color_temperature = analysis.get("color_temperature", 1.0)

        # Frequency / wavelet
        freq_high = analysis.get("fft_high_energy_ratio", 0.1)
        wavelet_detail_ratio = analysis.get("wavelet_detail_ratio", 0.0)

        # --- Expanded metrics (Stage B: haze, shadows/highlights, local contrast) ---
        haze_level = analysis.get("haze_level", 0.0)
        shadow_clip = analysis.get("shadow_clip", 0.0)
        highlight_clip = analysis.get("highlight_clip", 0.0)
        shadow_mass = analysis.get("shadow_mass", 0.0)
        glcm_contrast = analysis.get("glcm_contrast", 100.0)

        config = {
            "scale": scale,
            "enhance_only": enhance_only,
            "denoise": {},
            "adjust": {},
            "upscale": {},
            "post": {},
        }

        log.info(
            f"Auto-config: noise={noise:.1f} blur={blur:.0f} "
            f"brightness={brightness:.0f} contrast={contrast:.0f} "
            f"detail={detail:.1f} DR={dynamic_range:.0f} "
            f"glcm_h={glcm_homogeneity:.3f} edge_d={edge_density:.3f} "
            f"fractal={fractal_dimension:.2f} freq_hi={freq_high:.3f}"
        )

        content = self._classify_content(analysis)
        log.info(f"Auto-config: content class = {content}")

        # --- 0. Предуменьшение (high-res низкого качества) ---
        # Апсемпленное/мыльное изображение уменьшаем до эффективного
        # разрешения; дальше конвейер работает как обычно (масштаб пользователя
        # не пересчитывается — это его ручной воркфлоу).
        eff_mp = megapixels
        eff_factor = float(analysis.get("effective_downscale_factor", 1.0))
        if allow_predownscale and eff_factor < 1.0 and megapixels > 1.0:
            from upscaler.engine.effective_resolution import FACTORS
            w, h = analysis.get("resolution", (0, 0))
            min_side = min(w, h)
            chosen = None
            for cand in sorted(f for f in FACTORS if f >= eff_factor):
                if min_side * cand >= 720:
                    chosen = cand
                    break
            if chosen is not None:
                config["predownscale"] = {"factor": chosen}
                eff_mp = megapixels * chosen * chosen
                log.info("Auto: predownscale to %d%% (effective factor %.2f, "
                         "eff_mp %.1f)", round(chosen * 100), eff_factor,
                         eff_mp)

        # Photos must not be over-smoothed; require clearly visible noise first.
        noise_floor = 10.0 if content == "photo" else 8.0

        # --- 1. Denoising (texture-aware, detail-preserving) ---
        if noise > noise_floor:
            if glcm_homogeneity > 0.6 and noise > 16:
                # Smooth area with heavy real noise -> AI denoise, but capped.
                strength = min(noise / 45.0, 0.7)
                config["denoise"]["SCUNet"] = {"strength": round(strength, 2),
                                               "tile_size": 512}
                log.info(f"Auto: smooth+noisy noise={noise:.1f} SCUNet={strength:.2f}")
            elif noise > 14:
                # Textured/photographic -> gentle AI denoise to protect texture.
                strength = min(noise / 70.0, 0.45)
                config["denoise"]["SCUNet"] = {"strength": round(strength, 2),
                                               "tile_size": 512}
                log.info(f"Auto: textured+noisy noise={noise:.1f} SCUNet gentle={strength:.2f}")
            else:
                # Mild noise -> NL-Means preserves edges/texture better.
                h_param = int(min(noise * 1.2, 14))
                config["denoise"]["NL-Means"] = {
                    "h": h_param, "template_window": 7, "search_window": 21}
                log.info(f"Auto: mild noise={noise:.1f} NL-Means h={h_param}")
        # noise <= floor: keep all detail, no denoise.

        # --- 2. Цветокоррекция: сильный cast -> White Balance, умеренный ->
        # Auto Color (температурное отклонение остаётся за Auto Color).
        temp_deviation = abs(color_temperature - 1.0)
        max_cast = max(abs(c) for c in color_cast)

        if max_cast > 12:
            strength = round(min(max_cast / 25.0, 0.9), 2)
            config["adjust"]["White Balance"] = {"strength": strength}
            log.info(f"Auto: strong cast ({max_cast:.1f}) -> White Balance "
                     f"{strength}")
        elif temp_deviation > 0.08 or max_cast > 8:
            strength_from_temp = min(temp_deviation / 0.4, 1.0)
            strength_from_cast = min(max_cast / 20.0, 1.0)
            strength = round(max(strength_from_temp, strength_from_cast), 2)
            config["adjust"]["Auto Color"] = {"strength": strength}
            log.info(
                f"Auto: color correction (temp_dev={temp_deviation:.2f}, "
                f"cast={max_cast:.1f}), strength={strength}"
            )

        # --- 3. Tone / contrast (histogram-aware) ---
        # Auto Levels (6d) is the preferred corrector on a narrow histogram;
        # Auto Tone's contrast/DR trigger is suppressed there so the two
        # don't both fire on the same narrow-range condition. Skew/peak
        # triggers still fire unconditionally (they address a different
        # problem — distribution shape, not range).
        narrow_range = dynamic_range < 200 and contrast < 45

        needs_tone = False
        tone_strength = 0.2

        if (contrast < 35 or dynamic_range < 180) and not narrow_range:
            needs_tone = True
            tone_strength = 0.3 if contrast < 25 else 0.2

        if abs(hist_skewness) > 1.0:
            # Skewed histogram -> tonal imbalance
            needs_tone = True
            tone_strength = max(tone_strength, min(abs(hist_skewness) / 4.0, 0.5))

        if hist_kurtosis > 3.0:
            # Peaked histogram -> needs spreading
            needs_tone = True
            tone_strength = max(tone_strength, min(hist_kurtosis / 10.0, 0.5))

        if needs_tone:
            config["adjust"]["Auto Tone"] = {
                "strength": round(tone_strength, 2),
                "clip_limit": 1.0,
                "grid_size": 8,
            }
            log.info(
                f"Auto: tone correction (contrast={contrast:.0f}, "
                f"skew={hist_skewness:.2f}, kurt={hist_kurtosis:.2f}), "
                f"strength={tone_strength:.2f}"
            )

        # --- 4. Brightness ---
        if brightness < 60:
            boost = round(min((80 - brightness) / 80.0, 0.6), 2)
            config["adjust"]["Brightness"] = {"strength": boost, "direction": "up"}
            log.info(f"Auto: dark image ({brightness:.0f}), brightness boost={boost}")
        elif brightness > 200:
            reduce = round(min((brightness - 180) / 80.0, 0.5), 2)
            config["adjust"]["Brightness"] = {"strength": reduce, "direction": "down"}
            log.info(f"Auto: bright image ({brightness:.0f}), brightness reduce={reduce}")

        # --- 5. Vibrance (вместо грубого Saturation) ---
        if not is_grayscale and saturation_mean < 40:
            boost = round(min((50 - saturation_mean) / 50.0, 0.5) + 0.1, 2)
            config["adjust"]["Vibrance"] = {"strength": boost}
            log.info(f"Auto: low saturation ({saturation_mean:.0f}), "
                     f"Vibrance={boost}")

        # --- 6. Contrast enhancement ---
        if contrast < 30 and dynamic_range > 100:
            config["adjust"]["Auto Contrast"] = {"strength": 0.3}
            log.info(
                f"Auto: flat image (contrast={contrast:.0f}, DR={dynamic_range:.0f}), "
                "using Auto Contrast"
            )

        # --- 6b. Дымка ---
        if content == "photo" and haze_level > 0.5:
            strength = round(min((haze_level - 0.5) * 1.6, 0.8), 2)
            config["adjust"]["Dehaze"] = {"strength": strength}
            log.info(f"Auto: haze={haze_level:.2f}, Dehaze={strength}")

        # --- 6c. Тени/Света ---
        sh = 0.0
        hi = 0.0
        if shadow_mass > 0.35 or shadow_clip > 0.02:
            sh = min(0.2 + shadow_mass * 0.6, 0.6)
        if highlight_clip > 0.02:
            hi = min(0.2 + highlight_clip * 8.0, 0.5)
        if sh > 0.0 or hi > 0.0:
            config["adjust"]["Shadows/Highlights"] = {
                "shadows": round(sh, 2), "highlights": round(hi, 2),
                "radius": 30}
            log.info(f"Auto: Shadows/Highlights sh={sh:.2f} hi={hi:.2f}")

        # --- 6d. Auto Levels при узкой гистограмме ---
        if narrow_range:
            strength = round(min((200 - dynamic_range) / 150.0 + 0.2, 0.7), 2)
            config["adjust"]["Auto Levels"] = {"strength": strength,
                                               "clip_percent": 0.5}
            log.info(f"Auto: narrow range DR={dynamic_range:.0f}, "
                     f"Auto Levels={strength}")

        # --- 6e. Clarity для плоских фото ---
        if content == "photo" and glcm_contrast < 30.0:
            config["adjust"]["Clarity"] = {"strength": 0.2, "radius": 60}
            log.info("Auto: low local contrast, Clarity=0.2")

        # --- 7. Upscaler selection (frequency & complexity aware) ---
        if not enhance_only:
            upscaler = self._select_upscaler(
                noise, detail, eff_mp, scale,
                freq_high, wavelet_detail_ratio, fractal_dimension,
                content=content,
            )
            config["upscale"] = {"plugin": upscaler, "scale": scale}
            log.info(f"Auto: selected upscaler {upscaler}")
        else:
            upscaler = self._select_upscaler(
                noise, detail, eff_mp, 4,
                freq_high, wavelet_detail_ratio, fractal_dimension,
                content=content,
            )
            config["upscale"] = {"plugin": upscaler, "scale": 4}
            log.info(f"Auto: enhance-only with {upscaler}")

        # --- 8. Refocus / sharpening (edge-aware) ---
        if blur < 100:
            refocus_strength = 0.8
            fine, medium, coarse = 1.2, 0.8, 0.5
        elif blur < 300:
            refocus_strength = 0.6
            fine, medium, coarse = 1.0, 0.6, 0.3
        else:
            refocus_strength = 0.4
            fine, medium, coarse = 0.8, 0.4, 0.2

        # Pull back if image is already sharp (high edge density -> risk of artifacts)
        if edge_density > 0.15:
            dampening = max(0.3, 1.0 - (edge_density - 0.15) * 4.0)
            refocus_strength *= dampening
            fine *= dampening
            medium *= dampening
            coarse *= dampening
            log.info(
                f"Auto: high edge density ({edge_density:.3f}), "
                f"reducing refocus by {1.0 - dampening:.0%}"
            )

        config["adjust"]["Refocus"] = {
            "strength": round(refocus_strength, 2),
            "fine_detail": round(fine, 2),
            "medium_detail": round(medium, 2),
            "coarse_detail": round(coarse, 2),
        }

        # Lighter final unsharp on photos to avoid halos/plastic skin.
        config["post"]["sharpen"] = 0.14 if content == "photo" else 0.2

        # --- 9. Deblur (deconvolution) — only when the assessment says the
        # image is actually blurred. Explicit params + method make the run
        # deterministic; the plugin's quality safeguard reverts if it doesn't
        # genuinely improve the image.
        assessment = analysis.get("blur_assessment")
        if assessment and assessment.get("needs_deblur"):
            config["deblur"] = {
                "auto": False,
                "blur_type": assessment.get("blur_type", "gaussian"),
                "radius": assessment.get("radius", 3.0),
                "angle": assessment.get("angle", 0.0),
                "smooth": assessment.get("smooth", 30.0),
                "method": assessment.get("method", "wiener"),
                "edge_taper": True,
            }
            for key in ("tv_iterations", "edge_feather", "correction_strength"):
                if key in assessment:
                    config["deblur"][key] = assessment[key]
            log.info("Auto: blur detected (sharpness=%.2f), SmartDeblur method=%s",
                     assessment.get("sharpness", 0.0), config["deblur"]["method"])
            # Оценка размытия делалась на оригинале: после предуменьшения
            # радиус ядра масштабируется тем же фактором.
            if config.get("predownscale"):
                f = config["predownscale"]["factor"]
                config["deblur"]["radius"] = max(
                    round(config["deblur"]["radius"] * f, 2), 1.0)
                log.info("Auto: deblur radius rescaled x%.2f for predownscale", f)

        # --- 10. Face restoration (when faces are present) ---
        if analysis.get("has_faces"):
            config["face"] = self._face_params(analysis)
            log.info(f"Auto: {analysis.get('face_count', 1)} face(s) detected, "
                     f"enabling CodeFormer (w={config['face']['fidelity']})")

            face_noise = float(analysis.get("face_noise", 0.0))
            if face_noise > 8.0:
                strength = round(min(0.2 + face_noise / 40.0, 0.6), 2)
                config["adjust"]["Skin Smooth"] = {"strength": strength,
                                                   "radius": 10}
                log.info(f"Auto: noisy faces ({face_noise:.1f}), "
                         f"Skin Smooth={strength}")

        return config

    def _classify_content(self, analysis: dict) -> str:
        """Coarse content class from existing metrics: photo / illustration / graphic.

        Photographs have high entropy, moderate edges and some color/texture;
        flat illustrations/screenshots have low entropy, hard edges, high GLCM
        homogeneity and little texture.
        """
        entropy = analysis.get("gray_entropy", 5.0)
        edge_density = analysis.get("edge_density", 0.05)
        saturation = analysis.get("saturation_mean", 60)
        homogeneity = analysis.get("glcm_homogeneity", 0.5)

        graphic_score = 0
        if entropy < 4.5:
            graphic_score += 1
        if homogeneity > 0.7:
            graphic_score += 1
        if edge_density > 0.18:
            graphic_score += 1
        if saturation < 15:
            graphic_score += 1

        if graphic_score >= 3:
            return "graphic"
        if graphic_score == 2:
            return "illustration"
        return "photo"

    def _face_params(self, analysis: dict) -> dict:
        """Алгоритмические параметры CodeFormer по качеству лицевых кропов.

        Деградированные (размытые/шумные/мелкие) лица получают низкий
        fidelity (сильная реставрация), чистые крупные — высокий (ближе к
        оригиналу). Монотонно по резкости, антимонотонно по шуму.
        """
        sharp = float(analysis.get("face_sharpness", 0.5))          # 0..1
        noise_norm = min(float(analysis.get("face_noise", 5.0)) / 20.0, 1.0)
        quality = 0.65 * sharp + 0.35 * (1.0 - noise_norm)
        face_px = analysis.get("face_min_px")
        if isinstance(face_px, (int, float)) and face_px < 48:
            quality -= 0.15  # у мелких лиц деталям доверять нельзя
        fidelity = round(min(max(0.3 + 0.6 * quality, 0.3), 0.9), 2)

        w, h = analysis.get("resolution", (1024, 1024))
        min_face_px = int(min(max(round(0.015 * min(w, h)), 16), 96))
        log.info("Auto: face params fidelity=%.2f min_face_px=%d "
                 "(sharp=%.2f noise_norm=%.2f)",
                 fidelity, min_face_px, sharp, noise_norm)
        return {"enabled": True, "fidelity": fidelity,
                "min_face_px": min_face_px, "upscale_background": False}

    def _select_upscaler(
        self,
        noise: float,
        detail: float,
        megapixels: float,
        scale: int,
        freq_high: float,
        wavelet_detail_ratio: float,
        fractal_dimension: float,
        content: str = "photo",
    ) -> str:
        """Pick the best upscaler, prioritising natural fidelity for photos."""
        # Large images -> lighter model for memory/speed.
        if megapixels > 8:
            return "Real-ESRGAN"

        if content in ("illustration", "graphic"):
            # Hard edges / flat regions: detail-transformer SR.
            if fractal_dimension > 1.8:
                return "DAT"
            return "SwinIR" if scale in (2, 4) else "Real-ESRGAN"

        # Photographs: fidelity-first.
        # Clean, highly-detailed source -> HAT-S extracts the most real detail.
        # wavelet_detail_ratio is normalised 0-1; > 0.3 means ≥30% energy in
        # detail subbands, indicating genuine high-frequency content.
        if noise <= 8 and (detail > 22 or freq_high > 0.25
                           or wavelet_detail_ratio > 0.3):
            return "HAT-S"
        # Default photographic choice: robust, natural, artifact-resistant.
        return "Real-ESRGAN"

    def describe(self, config: dict) -> str:
        """Human-readable summary of what auto-config chose (i18n: tr())."""
        from upscaler.ui.i18n import tr
        parts = []

        pre = config.get("predownscale")
        if isinstance(pre, dict) and pre.get("factor"):
            parts.append(tr("desc.predownscale",
                            pct=int(round(pre["factor"] * 100))))

        if config.get("denoise"):
            names = ", ".join(config["denoise"].keys())
            parts.append(tr("desc.denoise", names=names))

        # Collect correction steps (everything in adjust except Refocus)
        adjust = config.get("adjust", {})
        correction_names = [k for k in adjust if k != "Refocus"]
        if correction_names:
            parts.append(tr("desc.correction", names=", ".join(correction_names)))

        up = config.get("upscale", {})
        if up.get("plugin"):
            mode = tr("desc.scale_mode_enhance") if config.get("enhance_only") \
                else f'{config.get("scale", "?")}x'
            parts.append(tr("desc.scale", plugin=up["plugin"], mode=mode))

        # Refocus / sharpening
        refocus = adjust.get("Refocus")
        sharpen = config.get("post", {}).get("sharpen")
        sharp_parts = []
        if refocus:
            sharp_parts.append(f"Refocus {refocus['strength']}")
        if sharpen:
            sharp_parts.append(f"sharpen {sharpen}")
        if sharp_parts:
            parts.append(tr("desc.sharpness", parts=", ".join(sharp_parts)))

        if config.get("deblur"):
            parts.append(tr("desc.deblur"))

        if config.get("face"):
            parts.append(
                tr("desc.face", fidelity=config["face"].get("fidelity", 0.7)))

        order = config.get("order")
        if isinstance(order, list) and order:
            from upscaler.engine.pipeline import step_label
            labels = [step_label(t) for t in order[:5]]
            suffix = "\u2026" if len(order) > 5 else ""
            parts.append(tr("desc.order", labels=" \u2192 ".join(labels) + suffix))

        return " \u203a ".join(parts) if parts else tr("desc.none")
