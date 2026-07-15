"""Блок G, задача 5: финальная проверка потока вариантов/смешивания.

Регрессия на уровне сигналов/движка (без запуска worker-процесса):
- движок 6A (greedy_blend_search) по различающимся вариантам даёт валидный
  рецепт — это то, чем пользуется MainWindow._on_blend_auto после сброса
  сессии вариантов на новый прогон.
"""
import numpy as np


def test_auto_result_recipe_nonempty_when_variants_differ():
    """Авто-подбор по различающимся вариантам даёт рецепт (движок 6A)."""
    from upscaler.engine.blend_search import greedy_blend_search, make_proxy
    base = np.random.default_rng(1).integers(0, 256, (128, 128, 3), np.uint8)
    dark = (base * 0.5).astype(np.uint8)
    proxies = {"a": make_proxy(base), "b": make_proxy(dark)}
    recipe = greedy_blend_search(proxies)
    assert recipe["base"] in ("a", "b")
    assert isinstance(recipe.get("layers"), list)
