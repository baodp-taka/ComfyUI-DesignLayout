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
        for name in compositions.list_names():   # legacy + ~200 templates
            if aspect not in compositions.aspects_of(name):
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
    # the hint is only a family preference now: whatever template the auto
    # pick lands on must suit the tall canvas (no column layouts on 9:16)
    lay = LayoutEngine(reg, {"w": 1080, "h": 1920}).layout(spec, seed=0)
    assert lay["aspect"] == "tall"
    assert "tall" in compositions.aspects_of(lay["composition"])
    for cw, ch in SIZES:
        aspect = compositions.aspect_class(cw, ch)
        for seed in range(8):
            lay = LayoutEngine(reg, {"w": cw, "h": ch}).layout(
                dict(spec, composition_hint="auto"), seed=seed)
            assert aspect in compositions.aspects_of(lay["composition"])


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


# ---- many-text / common-case regressions ----------------------------------
MANY_TEXTS = [
    ("headline", "LỄ HỘI ẨM THỰC ĐÀ NẴNG"),
    ("subheadline", "Hơn 100 gian hàng món ngon ba miền"),
    ("emphasis", "VÉ 50K"), ("emphasis", "FREE nước uống"),
    ("body", "Biểu diễn âm nhạc đường phố mỗi tối"),
    ("detail", "18:00 - 23:00 · 20/10 đến 22/10"),
    ("detail", "Công viên Biển Đông, Sơn Trà"),
    ("detail", "Hotline: 0905 123 456"),
    ("note", "Trẻ em dưới 1m2 miễn phí vé vào cổng"),
]


def _many_spec():
    return {"language": "vietnamese", "mood": "bold", "design_type": "poster",
            "background_color": "#1E2230",
            "texts": [{"role": r, "text": t, "priority": i + 1}
                      for i, (r, t) in enumerate(MANY_TEXTS)]}


def _ink_box(reg, block):
    from designlayout import measure
    from designlayout.render_preview import _line_positions
    f = measure.get_font(reg, block["font"], block["size_px"])
    x0 = y0 = 10 ** 9
    x1 = y1 = -10 ** 9
    for ln, x, y in _line_positions(reg, block, "vietnamese"):
        bb = f.getbbox(ln, anchor="la")
        x0, y0 = min(x0, x + bb[0]), min(y0, y + bb[1])
        x1, y1 = max(x1, x + bb[2]), max(y1, y + bb[3])
    return x0, y0, x1, y1


def test_wrap_never_drops_words():
    from designlayout import measure
    reg = _reg()
    text = ("Ưu đãi đặc biệt dành riêng cho 100 khách hàng đầu tiên đăng ký "
            "tham gia chương trình khai trương tuần này")
    fit = measure.fit_block(reg, text, "Roboto-Light.ttf", "vietnamese",
                            max_w=500, max_h=10 ** 9, target_size=48,
                            ls_frac=0, allow_multiline=True, max_lines=3)
    assert " ".join(" ".join(fit["lines"]).split()) == " ".join(text.split())
    assert len(fit["lines"]) <= 3 or fit.get("overflow")


@pytest.mark.parametrize("comp", ["", "top_headline", "bottom_band",
                                  "center_stack", "badge_focus"])
def test_many_texts_all_placed_no_ink_collision_hierarchy(comp):
    reg = _reg()
    eng = LayoutEngine(reg, {"w": 1080, "h": 1350})
    lay = eng.layout(_many_spec(), seed=3, force_composition=comp)
    blocks = lay["blocks"]
    shown = " ".join(" ".join(b["lines"]) for b in blocks)
    for _, t in MANY_TEXTS:                      # nothing dropped / truncated
        assert t in shown, t
    inks = [_ink_box(reg, b) for b in blocks]    # real glyphs never touch
    for i in range(len(inks)):
        for j in range(i + 1, len(inks)):
            a, c = inks[i], inks[j]
            assert a[2] <= c[0] or c[2] <= a[0] or a[3] <= c[1] or \
                c[3] <= a[1], (blocks[i]["text"], blocks[j]["text"])
    size = {}
    for b in blocks:
        size[b["role"]] = min(size.get(b["role"], 10 ** 6), b["size_px"])
    chain = [size[r] for r in ("headline", "subheadline", "body", "detail",
                               "note") if r in size]
    assert chain == sorted(chain, reverse=True)  # hierarchy never inverts
    inline = [b["size_px"] for b in blocks
              if b["role"] == "emphasis" and b.get("valign") != "middle"]
    assert all(s <= size["headline"] for s in inline)


# ---- text color: intent hue kept, contrast guaranteed ----------------------
def _hue(rgb):
    import colorsys
    return colorsys.rgb_to_hls(*(c / 255.0 for c in rgb))[0] * 360


def test_fit_contrast_keeps_hue_and_reaches_ratio():
    from designlayout.colors import fit_contrast, contrast_ratio
    blush, cream_bg = (201, 139, 155), (242, 236, 228)
    out = fit_contrast(blush, cream_bg, 3.0)
    assert contrast_ratio(out, cream_bg) >= 3.0
    assert abs(_hue(out) - _hue(blush)) < 3            # still blush pink
    assert fit_contrast((243, 227, 195), (40, 25, 16), 4.5) == (243, 227, 195)


def test_parser_reads_and_validates_colors():
    spec, _, warns = parsing.parse_llm_output(
        '{"texts":[{"role":"headline","text":"A"}],'
        '"text_color":"#f3e3c3","accent_color":"banana"}')
    assert spec["text_color"] == "#F3E3C3"
    assert spec["accent_color"] is None and any("accent_color" in w
                                                for w in warns)


def test_zone_check_uses_intent_not_plain_black_white():
    reg = _reg()
    eng = LayoutEngine(reg, {"w": 1080, "h": 1350})
    spec = {"language": "vietnamese", "mood": "vintage", "design_type": "poster",
            "background_color": "#3B2618", "text_color": "#F3E3C3",
            "accent_color": "#E8B86B",
            "texts": [{"role": "headline", "text": "MỘC COFFEE", "priority": 1},
                      {"role": "subheadline", "text": "Mộc mạc từng ngụm",
                       "priority": 2}]}
    lay = eng.layout(spec, seed=1, force_composition="top_headline")
    img = Image.new("RGB", (1080, 1350), (50, 32, 20))
    out, _ = zone_check.check(img, lay)
    colors = {b["role"]: b["color"] for b in out["blocks"]}
    assert colors["headline"] == "#F3E3C3"          # cream kept as-is
    assert colors["subheadline"] == "#E8B86B"       # gold accent kept
    assert out["palette"] == {"text": "#F3E3C3", "accent": "#E8B86B"}


def test_badge_text_not_recolored_against_image():
    reg = _reg()
    eng = LayoutEngine(reg, {"w": 1080, "h": 1350})
    spec = {"language": "vietnamese", "mood": "bold", "design_type": "social",
            "background_color": "#F4F1EA", "accent_color": "#E53935",
            "texts": [{"role": "headline", "text": "SIÊU SALE", "priority": 1},
                      {"role": "emphasis", "text": "-50%", "priority": 1}]}
    lay = eng.layout(spec, seed=4, force_composition="badge_focus")
    light = Image.new("RGB", (1080, 1350), (248, 245, 238))
    out, _ = zone_check.check(light, lay)
    badge = next(s for s in out["shapes"] if s["type"] == "badge")
    txt = next(b for b in out["blocks"] if b.get("on_shape"))
    assert txt["color"] == "#FFFFFF"                 # white on the red badge
    assert contrast_ratio(hex_to_rgb(txt["color"]),
                          hex_to_rgb(badge["fill"])) >= 4.5
    assert abs(_hue(hex_to_rgb(badge["fill"])) - _hue((229, 57, 53))) < 3


def test_auto_accent_differs_from_background_hue():
    from designlayout.colors import accent_from_palette, contrast_ratio
    teal_bg = (20, 60, 66)
    palette = [(22, 70, 78), (30, 90, 100), (240, 130, 40)]  # teal x2 + orange
    acc = accent_from_palette(palette, teal_bg, 4.5)
    assert acc is not None and contrast_ratio(acc, teal_bg) >= 4.5
    assert _hue(acc) < 60                            # orange, not teal


# ---- template library (~200) -------------------------------------------------
def test_template_library_size_and_coverage():
    from designlayout import templates
    assert len(templates.TEMPLATES) >= 200
    assert len(templates.BY_ID) == len(templates.TEMPLATES)   # unique ids
    common = {"poster": ["portrait", "square", "tall", "landscape"],
              "banner": ["wide", "landscape"],
              "thumbnail": ["landscape", "square", "tall"],
              "invitation": ["portrait", "square"],
              "wedding": ["portrait", "square", "tall"],
              "birthday": ["portrait", "square", "tall"],
              "social": ["square", "portrait", "tall"],
              "logo": ["square", "portrait"]}
    for dt, aspects in common.items():
        for a in aspects:
            native = [t for t in templates.TEMPLATES
                      if dt in t["design_types"] and a in t["aspects"]]
            assert len(native) >= 4, (dt, a, len(native))


def test_auto_pick_varies_with_seed_and_respects_type():
    reg = _reg()
    spec = {"language": "vietnamese", "design_type": "wedding",
            "texts": [{"role": "headline", "text": "Minh & Lan"},
                      {"role": "detail", "text": "17:00 · 12/10"}]}
    picked = set()
    for seed in range(12):
        lay = LayoutEngine(reg, {"w": 1080, "h": 1350}).layout(spec, seed=seed)
        picked.add(lay["template"])
        assert lay["template"].startswith(("wedding.", "invite.")) or \
            "wedding" in __import__("designlayout.templates", fromlist=["x"]
                                    ).BY_ID[lay["template"]]["design_types"]
    assert len(picked) >= 5            # seeds give genuinely different layouts


def _decor_hit_boxes(s):
    w = s.get("width", 3)
    if s["type"] == "divider":
        o = s.get("ornament_size", 6)
        return [(s["x0"], s["y"] - max(w, o), s["x1"], s["y"] + max(w, o))]
    if s["type"] == "accent_bar":
        return [(s["x"], s["y"], s["x"] + s["w"], s["y"] + s["h"])]
    if s["type"] == "dots":
        return [(x - r, y - r, x + r, y + r) for x, y, r in s["points"]]
    if s["type"] == "border":
        x0, y0, x1, y1 = s["x"], s["y"], s["x"] + s["w"], s["y"] + s["h"]
        g = s.get("gap", 0) + w
        return [(x0, y0, x1, y0 + g), (x0, y1 - g, x1, y1),
                (x0, y0, x0 + g, y1), (x1 - g, y0, x1, y1)]
    return []


@pytest.mark.parametrize("tid", ["wedding.monogram.frame_diamond",
                                 "wedding.monogram.double_frame.card",
                                 "birthday.center.confetti_ribbon",
                                 "birthday.age_badge.frame_confetti.star",
                                 "thumb.center_huge.accent_strip",
                                 "thumb.block.left.accent",
                                 "poster.bottom_bar.dark.center",
                                 "banner.badge.middle.pill",
                                 "invite.card.center.light",
                                 "logo.center.ring"])
def test_decorated_templates_never_touch_text(tid):
    from designlayout import templates
    reg = _reg()
    aspect = sorted(templates.BY_ID[tid]["aspects"])[0]
    size = {"tall": (1080, 1920), "portrait": (1080, 1350),
            "square": (1080, 1080), "landscape": (1280, 720),
            "wide": (1920, 640)}[aspect]
    spec = {"language": "vietnamese", "mood": "bold", "design_type": "poster",
            "background_color": "#1E2230", "accent_color": "#FFB300",
            "texts": [{"role": "headline", "text": "HAPPY BIRTHDAY"},
                      {"role": "subheadline", "text": "Bé Bin tròn 5 tuổi"},
                      {"role": "emphasis", "text": "5"},
                      {"role": "detail", "text": "Chủ Nhật 20/10 · 15:00"}]}
    lay = LayoutEngine(reg, {"w": size[0], "h": size[1]}).layout(
        spec, seed=2, force_composition=tid)
    inks = [_ink_box(reg, b) for b in lay["blocks"]]
    for s in lay["shapes"]:
        for db in _decor_hit_boxes(s):
            for k in inks:
                assert db[2] <= k[0] or k[2] <= db[0] or db[3] <= k[1] or \
                    k[3] <= db[1], (tid, s["type"])
    for b in lay["blocks"]:               # text on a plate reads on the plate
        if b.get("on_panel"):
            panel = next(s for s in lay["shapes"] if s["type"] == "panel")
            assert contrast_ratio(hex_to_rgb(b["color"]),
                                  hex_to_rgb(panel["fill"])) >= 3.0, tid


def test_balanced_and_phrase_aware_wrapping():
    from designlayout import measure
    reg = _reg()
    f = measure.fit_block(reg, "10 MẸO NẤU ĂN SIÊU NHANH", "aachenb.ttf",
                          "vietnamese", max_w=560, max_h=10 ** 9,
                          target_size=120, ls_frac=0, max_lines=3)
    assert all(len(ln.split()) >= 2 for ln in f["lines"])   # no lone word
    f = measure.fit_block(reg, "17:00 · Chủ Nhật 12/10/2026",
                          "Roboto-Light.ttf", "vietnamese", max_w=380,
                          max_h=10 ** 9, target_size=40, ls_frac=0,
                          max_lines=2)
    assert f["lines"][0].endswith("·")                       # not mid-date
    f = measure.fit_block(reg, "Minh & Lan", "aachenb.ttf", "vietnamese",
                          max_w=845, max_h=10 ** 9, target_size=157,
                          ls_frac=0.02, max_lines=2)
    assert f["lines"] == ["Minh & Lan"]                      # 1 line, a bit smaller


# ---- image-first layout ------------------------------------------------------
def _half_busy(cw, ch, busy_side="left"):
    """Noise (busy) on one half, flat color on the other."""
    rng = np.random.default_rng(0)
    a = np.full((ch, cw, 3), 200, np.uint8)
    noise = rng.integers(0, 255, (ch, cw // 2, 3), np.uint8)
    if busy_side == "left":
        a[:, :cw // 2] = noise
    else:
        a[:, cw // 2:] = noise
    return Image.fromarray(a)


@pytest.mark.parametrize("busy_side", ["left", "right"])
def test_image_fit_puts_text_on_the_calm_side(busy_side):
    from designlayout import image_fit
    reg = _reg()
    cw = ch = 1080
    spec = {"language": "latin", "mood": "minimal", "design_type": "poster",
            "background_color": "#C8C8C8",
            "texts": [{"role": "headline", "text": "CALM SIDE"},
                      {"role": "detail", "text": "Saturday 12/10"}]}
    img = _half_busy(cw, ch, busy_side)
    lay, report, _ = image_fit.fit_layout(reg, spec, img, {"w": cw, "h": ch},
                                          seed=1)
    xs = [(b[0] + b[2]) / 2 for b, _, _ in image_fit.line_boxes(reg, lay)]
    mean_x = sum(xs) / len(xs)
    if busy_side == "left":
        assert mean_x > 0.5 * cw, (lay["template"], mean_x)
    else:
        assert mean_x < 0.5 * cw, (lay["template"], mean_x)


def test_image_fit_pool_has_no_plates_and_defocus_calms():
    from designlayout import image_fit, compositions
    spec = {"design_type": "poster",
            "texts": [{"role": "headline", "text": "HELLO"}]}
    pool = image_fit.candidate_pool(spec, "square")
    assert pool and not any(t.get("panel") for t in pool)
    reg = _reg()
    img = _half_busy(1080, 1080, "left")
    lay = LayoutEngine(reg, {"w": 1080, "h": 1080}).layout(
        dict(spec, language="latin", background_color="#C8C8C8"), seed=1,
        force_composition="poster.column.left.middle.plain")
    before = image_fit.ImageStats(img, 1080, 1080)
    treated = image_fit.defocus_behind_text(reg, lay, img, 1.0)
    after = image_fit.ImageStats(treated, 1080, 1080)
    for box, _, _ in image_fit.line_boxes(reg, lay):
        assert after.box_cost(box)[1] < 0.6 * before.box_cost(box)[1]
