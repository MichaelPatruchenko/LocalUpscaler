import numpy as np
import pytest


def test_cannot_instantiate_base_plugin():
    from upscaler.plugins.base import BasePlugin
    with pytest.raises(TypeError):
        BasePlugin()


def test_concrete_plugin_must_implement_methods():
    from upscaler.plugins.base import BasePlugin
    class Incomplete(BasePlugin):
        name = "test"
        category = "upscaler"
        supported_scales = [2]
        gpu_memory_mb = 0
        params_schema = {}
    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_plugin_works():
    from upscaler.plugins.base import BasePlugin
    class Dummy(BasePlugin):
        name = "dummy"
        category = "upscaler"
        supported_scales = [2]
        gpu_memory_mb = 0
        params_schema = {}
        def initialize(self, device): pass
        def process(self, image, params): return image
    d = Dummy()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert d.process(img, {}).shape == (4, 4, 3)


def test_plugin_category_enum():
    from upscaler.plugins.base import PluginCategory
    assert PluginCategory.UPSCALER == "upscaler"
    assert PluginCategory.DENOISER == "denoiser"
    assert PluginCategory.ADJUSTER == "adjuster"
