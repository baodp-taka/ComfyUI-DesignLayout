# -*- coding: utf-8 -*-
"""Offline tests for the text-first invitation card (no ComfyUI / torch / GPU)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from designlayout import invite_card as ic          # noqa: E402
from designlayout.fonts import FontRegistry         # noqa: E402

FONTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")

PLAN = {
    "language": "vietnamese", "paper": "#00FFFF", "ink": "#2C1810", "accent": "#D4AF37",
    "composition_rank": ["editorial", "classic", "hero_date", "modern_line", "save_the_date"],
    "card_rank": ["ticket", "arch", "rounded"],
    "background_prompt": "A warm garden with lanterns, a large blank paper card in the center, soft light.",
    "components": [
        {"type": "icon", "name": "heart"},
        {"type": "text", "style": "kicker", "text": "Trân trọng kính mời"},
        {"type": "names", "first": "Hoàng Nam", "connector": "và", "second": "Bảo Anh"},
        {"type": "text", "style": "invite", "text": "Đến dự tiệc cưới của chúng tôi vào tối thứ bảy tuần này"},
        {"type": "divider"},
        {"type": "date", "weekday": "Saturday", "month": "December", "day": "12", "year": "2025", "time": "5 PM"},
        {"type": "divider"},
        {"type": "venue", "name": "Nhà hàng Riverside", "address": "12 Tôn Đức Thắng, Quận 1"},
        {"type": "text", "style": "note", "text": "<a short warm closing line written for THIS occasion>"},
    ],
}
REQUEST = "Thiệp cưới Hoàng Nam và Bảo Anh, 11:00 20/12/2025 tại Nhà hàng Riverside, 12 Tôn Đức Thắng, Quận 1."


def test_vi_date_from_request_computes_weekday():
    d = ic.vi_date_from_request(REQUEST)
    assert d == {"day": "20", "month": "Tháng 12", "year": "2025", "weekday": "Thứ Bảy", "time": "11:00"}


def test_code_split_prefers_connectors_and_keeps_names():
    assert ic.code_split("Chúc mừng sinh nhật vui vẻ cho bé Gia Huy!") == \
        "Chúc mừng sinh nhật vui vẻ\ncho bé Gia Huy!"
    assert "\n" not in ic.code_split("Ba từ thôi")


def test_paper_is_kept_printable():
    assert ic.paper_of({"paper": "#00FFFF"}) == (211, 238, 238)     # neon -> pastel
    assert ic.paper_of({"paper": "001A3C"}) == (11, 27, 49)          # no '#' is fine
    assert ic.paper_of({}) == ic.DEFAULT_PAPER
    assert ic.paper_of({"paper": "dark red"})[0] > ic.paper_of({"paper": "dark red"})[2]   # a deep red
    assert ic.color_of("gold") == (255, 215, 0) and ic.color_of("no such colour") is None


def test_parse_plan_accepts_llm_chatter():
    assert ic.parse_plan("Sure!\n```json\n" + json.dumps(PLAN) + "\n```")["paper"] == "#00FFFF"


def test_prepare_locks_the_text_and_fixes_the_plan():
    out = ic.prepare(FontRegistry(FONTS), json.dumps(PLAN, ensure_ascii=False), REQUEST, seed=3, w=448, h=672)
    img, mask, alpha = out["image"], np.asarray(out["mask"]), np.asarray(out["alpha"])
    assert img.size == (448, 672) and mask.shape == alpha.shape == (672, 448)
    glyphs = alpha > 200
    assert glyphs.sum() > 500
    assert mask[glyphs].max() < 10                  # the text is never repainted
    assert mask[0, 0] > 245                         # the scene outside the card is
    rep = out["report"]
    assert rep["composition"] in ("editorial", "classic", "hero_date")   # inside the LLM's top 3
    assert rep["card"] in ("ticket", "arch")                              # inside its top 2
    assert any("Tháng 12" in str(v) for v in rep.values()) is False      # report has no date dump
    assert rep["dropped"]                           # the <placeholder> note was dropped
    assert "blank" not in out["prompt"] and out["prompt"].endswith("no other card.")


def test_placeholder_in_one_name_field_keeps_the_names():
    # a real plan from the endpoint: "<and>" left in the connector, a 0x colour,
    # a Vietnamese scene prompt
    plan = json.loads(json.dumps(PLAN))
    plan["paper"] = "0x000080"
    plan["background_prompt"] = "Một khung cảnh nhà hàng cổ điển sang trọng, tông đỏ đô."
    for c in plan["components"]:
        if c["type"] == "names":
            c["connector"] = "<and>"
    assert ic.color_of("0x000080") == (0, 0, 128)
    out = ic.prepare(FontRegistry(FONTS), plan, REQUEST, seed=2, w=448, h=672)
    assert out["report"]["scene_fallback"] is True
    assert "Một" not in out["prompt"] and out["prompt"].startswith("A softly lit")
    kept, dropped = ic.drop_copied(plan["components"], REQUEST)
    names = [c for c in kept if c["type"] == "names"][0]
    assert names["first"] == "Hoàng Nam" and names["connector"] == ""
    bad = [{"type": "names", "first": "<the MAIN name>", "second": ""}]
    assert ic.drop_copied(bad, REQUEST)[0] == []


def test_composite_puts_the_glyphs_back():
    from PIL import Image
    out = ic.prepare(FontRegistry(FONTS), PLAN, REQUEST, seed=1, w=448, h=672)
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (672, 448, 3), dtype=np.uint8))
    fin = np.asarray(ic.composite(noise, out["image"], out["alpha"]), np.int16)
    src = np.asarray(out["image"], np.int16)
    glyphs = np.asarray(out["alpha"]) > 250
    assert np.abs(fin[glyphs] - src[glyphs]).max() <= 2
