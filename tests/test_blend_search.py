"""Жадный подбор blend-рецепта на прокси."""
import cv2
import numpy as np
from upscaler.engine.blend import blend
from upscaler.engine.blend_search import (
    make_proxy, greedy_blend_search, apply_recipe,
)
from upscaler.engine.quality_score import quality_score


def _scene(size=192, seed=1):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (90, 110, 130)
    for _ in range(15):
        x, y = rng.integers(0, size - 40, 2)
        color = tuple(int(v) for v in rng.integers(30, 226, 3))
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 30, int(y) + 30),
                      color, -1)
    return img


def test_make_proxy_only_downscales():
    big = np.zeros((1024, 2048, 3), np.uint8)
    p = make_proxy(big, max_side=512)
    assert max(p.shape[:2]) == 512
    small = np.zeros((100, 100, 3), np.uint8)
    assert make_proxy(small, max_side=512).shape == small.shape


def test_base_is_best_candidate():
    sharp = _scene()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4.0)
    recipe = greedy_blend_search({"a": blurred, "b": sharp})
    assert recipe["base"] == "b"


def test_layers_only_accepted_on_improvement():
    img = _scene()
    # Идентичные кандидаты не могут улучшить друг друга
    recipe = greedy_blend_search({"a": img, "b": img.copy()})
    assert recipe["layers"] == []
    assert recipe["score"] == quality_score(img)["score"]


def test_search_deterministic():
    sharp = _scene()
    soft = cv2.GaussianBlur(sharp, (0, 0), 2.0)
    dark = (sharp * 0.5).astype(np.uint8)
    c = {"sharp": sharp, "soft": soft, "dark": dark}
    r1 = greedy_blend_search(dict(c))
    r2 = greedy_blend_search(dict(c))
    assert r1 == r2


def test_search_improves_score_with_useful_layer():
    base = _scene()
    dark = (base * 0.45).astype(np.uint8)   # низкая экспозиция/контраст
    recipe = greedy_blend_search({"dark": dark, "good": base})
    # Итоговый score не ниже собственного score лучшего кандидата
    best_single = quality_score({"dark": dark, "good": base}[recipe["base"]])
    assert recipe["score"] >= best_single["score"] - 1e-9


def test_max_layers_respected():
    img = _scene()
    variants = {f"v{i}": cv2.GaussianBlur(img, (0, 0), 0.5 + i)
                for i in range(5)}
    variants["orig"] = img
    recipe = greedy_blend_search(variants, max_layers=2)
    assert len(recipe["layers"]) <= 2
    # Кандидат не используется дважды
    used = [l["source"] for l in recipe["layers"]]
    assert len(used) == len(set(used))


def test_apply_recipe_full_resolution():
    big_a = cv2.resize(_scene(), (400, 300))
    big_b = cv2.resize(_scene(seed=2), (200, 150))  # другой размер
    recipe = {"base": "a",
              "layers": [{"source": "b", "mode": "soft_light",
                          "opacity": 0.5}],
              "score": 0.0}
    out = apply_recipe(recipe, {"a": big_a, "b": big_b})
    assert out.shape == big_a.shape
    expected = blend(big_a, cv2.resize(big_b, (400, 300),
                                       interpolation=cv2.INTER_LANCZOS4),
                     "soft_light", 0.5)
    assert np.array_equal(out, expected)
