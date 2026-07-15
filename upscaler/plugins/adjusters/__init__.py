"""Adjuster plugins: image enhancement and correction tools."""

from upscaler.plugins.adjusters.auto_contrast import AutoContrastPlugin
from upscaler.plugins.adjusters.auto_tone import AutoTonePlugin
from upscaler.plugins.adjusters.auto_color import AutoColorPlugin
from upscaler.plugins.adjusters.brightness import BrightnessPlugin
from upscaler.plugins.adjusters.contrast import ContrastPlugin
from upscaler.plugins.adjusters.saturation import SaturationPlugin
from upscaler.plugins.adjusters.sharpness import SharpnessPlugin
from upscaler.plugins.adjusters.refocus import RefocusPlugin
from upscaler.plugins.adjusters.auto_levels import AutoLevelsPlugin
from upscaler.plugins.adjusters.shadows_highlights import ShadowsHighlightsPlugin
from upscaler.plugins.adjusters.clarity import ClarityPlugin
from upscaler.plugins.adjusters.dehaze import DehazePlugin
from upscaler.plugins.adjusters.vibrance import VibrancePlugin
from upscaler.plugins.adjusters.white_balance import WhiteBalancePlugin
from upscaler.plugins.adjusters.optics import OpticsPlugin
from upscaler.plugins.adjusters.dodge_burn import DodgeBurnPlugin
from upscaler.plugins.adjusters.split_tone import SplitTonePlugin
from upscaler.plugins.adjusters.skin_smooth import SkinSmoothPlugin

__all__ = [
    "AutoContrastPlugin",
    "AutoTonePlugin",
    "AutoColorPlugin",
    "BrightnessPlugin",
    "ContrastPlugin",
    "SaturationPlugin",
    "SharpnessPlugin",
    "RefocusPlugin",
    "AutoLevelsPlugin",
    "ShadowsHighlightsPlugin",
    "ClarityPlugin",
    "DehazePlugin",
    "VibrancePlugin",
    "WhiteBalancePlugin",
    "OpticsPlugin",
    "DodgeBurnPlugin",
    "SplitTonePlugin",
    "SkinSmoothPlugin",
]
