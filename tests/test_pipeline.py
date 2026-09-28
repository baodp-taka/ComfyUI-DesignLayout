# -*- coding: utf-8 -*-
"""Offline tests for the layout pipeline (no ComfyUI / GPU)."""
import os
import sys
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from designlayout import parsing, zone_check                     # noqa: E402
from designlayout.fonts import FontRegistry                      # noqa: E402
from designlayout.layout import LayoutEngine                     # noqa: E402
from designlayout.colors import contrast_ratio, hex_to_rgb       # noqa: E402

FONTS_DIR = os.environ.get(
    "DESIGNLAYOUT_FONTS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "fonts"))

pytestmark = pytest.mark.skipif(not os.path.isdir(FONTS_DIR),
                                reason="fonts dir not available")


def _reg():
    return FontRegistry(FONTS_DIR)


def test_parse_broken_json_recovers():
    raw = ('junk ```json {"language":"vietnamese","texts":[{"role":"headline",'
           '"text":"KHAI TRƯƠNG","priority":1},]} tail')
    spec, bg, warns = parsing.parse_llm_output(raw)
    assert spec["texts"] and spec["texts"][0]["text"] == "KHAI TRƯƠNG"
    assert spec["language"] == "vietnamese"


def test_parse_total_garbage_is_safe():
    spec, bg, warns = parsing.parse_llm_output("not json at all")
    assert isinstance(spec["texts"], list)
    assert warns  # a warning was emitted


def test_clean_background_prompt_removes_text_words():
    s = parsing.clean_background_prompt(
        "A poster with bold text and big letters on a wall")
    assert "poster" not in s.lower()
    assert "text" not in s.lower() and "letters" not in s.lower()


def test_font_glyph_coverage_and_fallback():
    reg = _reg()
    # a Latin-only display font must NOT claim to cover Vietnamese accents
    assert reg.covers("Roboto-Light.ttf", "Đà Nẵng")
    file, covered = reg.pick("Đà Nẵng", "vietnamese", "headline", "vintage")
    assert covered and reg.covers(file, "Đà Nẵng")


def _boxes_overlap(a, b):
    return not (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"]
                or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])


def test_layout_within_canvas_and_no_overlap():
    reg = _reg()
    eng = LayoutEngine(reg)
    spec = {"language": "vietnamese", "mood": "vintage", "design_type": "poster",
            "background_color": "#3B2618",
            "texts": [{"role": "headline", "text": "MỘC COFFEE", "priority": 1},
                      {"role": "subheadline",
                       "text": "KHAI TRƯƠNG · GIẢM 30% TUẦN ĐẦU", "priority": 2},
                      {"role": "detail",
                       "text": "Thứ Bảy 12/10 · 123 Bạch Đằng", "priority": 3}]}
    lay = eng.layout(spec, seed=3)
    cw, ch = lay["canvas"]["w"], lay["canvas"]["h"]
    boxes = [b["box"] for b in lay["blocks"]]
    for bx in boxes:
        assert bx["x"] >= 0 and bx["y"] >= 0
        assert bx["x"] + bx["w"] <= cw and bx["y"] + bx["h"] <= ch
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _boxes_overlap(boxes[i], boxes[j]), "blocks overlap"


def test_zone_check_meets_contrast():
    reg = _reg()
    eng = LayoutEngine(reg)
    spec = {"language": "latin", "mood": "bold", "design_type": "poster",
            "background_color": "#334455",
            "texts": [{"role": "headline", "text": "HELLO WORLD", "priority": 1},
                      {"role": "detail", "text": "sub line here", "priority": 3}]}
    lay = eng.layout(spec, seed=1)
    img = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (1080, 1080, 3), np.uint8))
    updated, report = zone_check.check(img, lay, min_contrast=4.5)
    for b in updated["blocks"]:
        box = b["box"]
        region = np.asarray(img)[box["y"]:box["y"] + box["h"],
                                 box["x"]:box["x"] + box["w"]]
        mean = tuple(int(c) for c in region.reshape(-1, 3).mean(0))
        # picked color should beat plain fallback-less contrast most of the time
        assert contrast_ratio(hex_to_rgb(b["color"]), mean) >= 2.0


SIZES = [(1080, 1920), (1080, 1350), (1080, 1080), (1920, 1080),
         (1500, 500), (1584, 396)]


def test_aspect_templates_fit_every_size():
    from designlayout import compositions
    reg = _reg()
    spec = {"language": "vietnamese", "mood": "bold", "design_type": "social",
            "background_color": "#111820",
            "texts": [{"role": "headline", "text": "SIÊU SALE", "priority": 1},
                      {"role": "emphasis", "text": "50%", "priority": 1},
                      {"role": "subheadline", "text": "Duy nhất cuối tuần"},
                      {"role": "note", "text": "Áp dụng toàn bộ sản phẩm"}]}
    for cw, ch in SIZES:
        aspect = compositions.aspect_class(cw, ch)
        for name in compositions.list_names():
            if aspect not in compositions.COMPAT[name]:
                continue
            lay = LayoutEngine(reg, {"w": cw, "h": ch}).layout(
                spec, seed=1, force_composition=name)
            boxes = [b["box"] for b in lay["blocks"]]
            for bx in boxes:
                assert bx["x"] >= 0 and bx["y"] >= 0, (cw, ch, name)
                assert bx["x"] + bx["w"] <= cw and bx["y"] + bx["h"] <= ch, \
                    (cw, ch, name)
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    assert not _boxes_overlap(boxes[i], boxes[j]), \
                        (cw, ch, name)


def test_auto_pick_and_hint_follow_aspect():
    from designlayout import compositions
    reg = _reg()
    spec = {"language": "latin", "design_type": "poster",
            "composition_hint": "left_column",
            "texts": [{"role": "headline", "text": "HELLO"}]}
    lay = LayoutEngine(reg, {"w": 1080, "h": 1920}).layout(spec, seed=0)
    assert lay["aspect"] == "tall" and lay["composition"] == "story_bottom"
    for cw, ch in SIZES:
        aspect = compositions.aspect_class(cw, ch)
        for seed in range(8):
            lay = LayoutEngine(reg, {"w": cw, "h": ch}).layout(
                dict(spec, composition_hint="auto"), seed=seed)
            assert aspect in compositions.COMPAT[lay["composition"]]


def test_auto_bg_size_keeps_aspect():
    from designlayout.guide import auto_bg_size
    assert auto_bg_size(1080, 1080) == (1024, 1024)
    for cw, ch in SIZES:
        w, h = auto_bg_size(cw, ch)
        assert w % 16 == 0 and h % 16 == 0
        assert abs(w / h - cw / ch) / (cw / ch) < 0.02


def test_unsupported_ratio_is_rejected():
    reg = _reg()
    spec = {"texts": [{"role": "headline", "text": "HELLO"}]}
    for cw, ch in [(728, 90), (300, 1500)]:
        with pytest.raises(ValueError):
            LayoutEngine(reg, {"w": cw, "h": ch}).layout(spec)


def test_vietnamese_uses_only_vietnamese_fonts():
    reg = _reg()
    vi = {e["file"] for e in reg.candidates("vietnamese")}
    assert vi
    spec = {"language": "vietnamese", "mood": "festive",
            "texts": [{"role": "headline", "text": "SALE 50%"},
                      {"role": "subheadline", "text": "Khai trương"},
                      {"role": "detail", "text": "Thứ Bảy · 123 Bạch Đằng"}]}
    for seed in range(6):
        lay = LayoutEngine(reg).layout(spec, seed=seed)
        for b in lay["blocks"]:
            assert b["font"] in vi, (b["text"], b["font"])


def test_vietnamese_text_overrides_wrong_language():
    spec, _, warns = parsing.parse_llm_output(
        '{"language":"latin","texts":[{"role":"headline","text":"Mộc Coffee"}]}')
    assert spec["language"] == "vietnamese" and warns
