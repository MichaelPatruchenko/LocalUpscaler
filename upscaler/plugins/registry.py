"""Plugin registry with auto-discovery."""

import importlib
import inspect
import logging
import pkgutil
import sys

from upscaler.plugins.base import BasePlugin

log = logging.getLogger(__name__)

# Explicit list of all plugin modules for frozen (PyInstaller) builds
# where pkgutil.walk_packages cannot traverse bundled packages.
_BUILTIN_PLUGIN_MODULES = [
    "upscaler.plugins.upscalers.real_esrgan",
    "upscaler.plugins.upscalers.hat_s",
    "upscaler.plugins.upscalers.swinir",
    "upscaler.plugins.upscalers.omnisr",
    "upscaler.plugins.upscalers.dat",
    "upscaler.plugins.upscalers.lanczos",
    "upscaler.plugins.upscalers.bicubic",
    "upscaler.plugins.upscalers.sinc",
    "upscaler.plugins.upscalers.nedi",
    "upscaler.plugins.upscalers.dcci",
    "upscaler.plugins.upscalers.eggi_sire",
    "upscaler.plugins.denoisers.scunet",
    "upscaler.plugins.denoisers.nafnet",
    "upscaler.plugins.denoisers.bm3d_plugin",
    "upscaler.plugins.denoisers.nl_means",
    "upscaler.plugins.denoisers.bilateral",
    "upscaler.plugins.denoisers.wavelet",
    "upscaler.plugins.adjusters.auto_tone",
    "upscaler.plugins.adjusters.auto_contrast",
    "upscaler.plugins.adjusters.auto_color",
    "upscaler.plugins.adjusters.brightness",
    "upscaler.plugins.adjusters.contrast",
    "upscaler.plugins.adjusters.saturation",
    "upscaler.plugins.adjusters.sharpness",
    "upscaler.plugins.adjusters.refocus",
    "upscaler.plugins.adjusters.auto_levels",
    "upscaler.plugins.adjusters.shadows_highlights",
    "upscaler.plugins.adjusters.clarity",
    "upscaler.plugins.adjusters.dehaze",
    "upscaler.plugins.adjusters.vibrance",
    "upscaler.plugins.adjusters.white_balance",
    "upscaler.plugins.adjusters.optics",
    "upscaler.plugins.adjusters.dodge_burn",
    "upscaler.plugins.adjusters.split_tone",
    "upscaler.plugins.adjusters.skin_smooth",
    "upscaler.plugins.colorizers.ddcolor",
    "upscaler.plugins.colorizers.deoldify",
    "upscaler.plugins.colorizers.colormnet",
    "upscaler.plugins.deblur.smartdeblur",
    "upscaler.plugins.icedit.icedit",
    "upscaler.plugins.face.codeformer",
]


class PluginRegistry:
    """Registry that stores and discovers BasePlugin subclasses."""

    def __init__(self):
        self._plugins: dict = {}

    def register(self, plugin_cls: type) -> None:
        """Register a plugin class by its name."""
        self._plugins[plugin_cls.name] = plugin_cls

    def get(self, name: str):
        """Get plugin class by name, or None if not found."""
        return self._plugins.get(name)

    def list_plugins(self, category: str = None) -> list:
        """List all plugins, optionally filtered by category."""
        plugins = list(self._plugins.values())
        if category:
            plugins = [p for p in plugins if p.category == category]
        return plugins

    def _register_from_module(self, module) -> None:
        """Scan a module for BasePlugin subclasses and register them."""
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BasePlugin)
                and obj is not BasePlugin
                and obj.name
            ):
                self.register(obj)

    def auto_discover(self, package_name: str) -> None:
        """Scan a package for BasePlugin subclasses and register them."""
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return

        for importer, modname, ispkg in pkgutil.walk_packages(
            package.__path__, prefix=package.__name__ + "."
        ):
            try:
                module = importlib.import_module(modname)
            except Exception:
                continue
            self._register_from_module(module)

    def discover_builtin(self) -> None:
        """Discover all built-in plugins from upscaler.plugins subpackages."""
        if getattr(sys, "frozen", False):
            # Frozen build: import each plugin module explicitly
            for modname in _BUILTIN_PLUGIN_MODULES:
                try:
                    module = importlib.import_module(modname)
                    self._register_from_module(module)
                except Exception as e:
                    log.warning("Failed to load plugin %s: %s", modname, e)
        else:
            for sub in ("upscalers", "denoisers", "adjusters", "colorizers", "deblur", "icedit", "face"):
                self.auto_discover(f"upscaler.plugins.{sub}")
