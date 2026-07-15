import pytest


def test_registry_starts_empty():
    from upscaler.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    assert len(reg.list_plugins()) == 0


def test_register_and_get():
    from upscaler.plugins.registry import PluginRegistry
    from upscaler.plugins.base import BasePlugin

    class FakePlugin(BasePlugin):
        name = "fake"
        category = "upscaler"
        supported_scales = [2]
        gpu_memory_mb = 0
        params_schema = {}
        def initialize(self, device): pass
        def process(self, image, params): return image

    reg = PluginRegistry()
    reg.register(FakePlugin)
    assert reg.get("fake") is FakePlugin
    assert len(reg.list_plugins()) == 1


def test_list_by_category():
    from upscaler.plugins.registry import PluginRegistry
    from upscaler.plugins.base import BasePlugin

    class Up(BasePlugin):
        name = "up1"
        category = "upscaler"
        supported_scales = [2]
        gpu_memory_mb = 0
        params_schema = {}
        def initialize(self, device): pass
        def process(self, image, params): return image

    class Dn(BasePlugin):
        name = "dn1"
        category = "denoiser"
        supported_scales = []
        gpu_memory_mb = 0
        params_schema = {}
        def initialize(self, device): pass
        def process(self, image, params): return image

    reg = PluginRegistry()
    reg.register(Up)
    reg.register(Dn)
    assert len(reg.list_plugins("upscaler")) == 1
    assert len(reg.list_plugins("denoiser")) == 1


def test_get_unknown_returns_none():
    from upscaler.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    assert reg.get("nonexistent") is None


def test_auto_discover(tmp_path):
    """Test that auto_discover finds plugins in a directory."""
    from upscaler.plugins.registry import PluginRegistry

    plugin_code = '''
from upscaler.plugins.base import BasePlugin

class TestDiscoverPlugin(BasePlugin):
    name = "discovered"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {}
    def initialize(self, device): pass
    def process(self, image, params): return image
'''
    pkg = tmp_path / "test_plugins"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "my_plugin.py").write_text(plugin_code)

    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        reg = PluginRegistry()
        reg.auto_discover("test_plugins")
        assert reg.get("discovered") is not None
    finally:
        sys.path.pop(0)


def test_new_adjusters_discovered():
    from upscaler.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    reg.discover_builtin()
    names = {p.name for p in reg.list_plugins("adjuster")}
    for name in ("Auto Levels", "Shadows/Highlights", "Clarity", "Dehaze",
                 "Vibrance", "White Balance", "Optics", "Dodge & Burn",
                 "Split Toning", "Skin Smooth"):
        assert name in names, name
