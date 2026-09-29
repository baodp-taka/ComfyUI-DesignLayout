# -*- coding: utf-8 -*-
"""Draw real text (and shapes) for debug + composite previews.

Rendering happens at CANVAS resolution. `composite_preview` scales the given
background up to the canvas and draws the design on top; `debug_preview` draws
on a plain background_color canvas with zone outlines for quick inspection.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFilter

from . import measure
from .fonts import FontRegistry
from .colors import hex_to_rgb, rel_luminance


def _draw_runs(draw: ImageDraw.ImageDraw, reg: FontRegistry, text: str,
               primary: str, language: str, size: int, ls_px: float,
               x: float, y_top: float, fill, stroke_w: int = 0,
               stroke_fill=None) -> None:
    """Draw one line char-by-char (letter spacing + per-glyph font fallback)."""
    runs = measure.build_runs(reg, text, primary, language)
    cx = x
    for file, sub in runs:
        font = measure.get_font(reg, file, size)
        for ch in sub:
            if stroke_w > 0:
                draw.text((cx, y_top), ch, font=font, fill=stroke_fill,
                          anchor="la", stroke_width=stroke_w,
                          stroke_fill=stroke_fill)
            draw.text((cx, y_top), ch, font=font, fill=fill, anchor="la")
            cx += font.getlength(ch) + ls_px


def _line_positions(reg: FontRegistry, block: Dict,
                    language: str) -> List[Tuple[str, float, float]]:
    """(line, x, y_top) for every line of the block, honoring align/valign."""
    box = block["box"]
    size = int(block["size_px"])
    primary = block["font"]
    ls_px = block.get("letter_spacing", 0.0) * size
    lines: List[str] = block.get("lines") or [block["text"]]
    asc, desc = measure.line_metrics(reg, primary, size)
    lh = (asc + desc) * block.get("line_height", 1.15)
    # pad_top: room reserved for ink above the ascender (stacked accents)
    pt, pb = block.get("pad_top", 0), block.get("pad_bottom", 0)
    full = pt + lh * len(lines) + pb
    y = box["y"] + pt
    if block.get("valign") == "middle":
        y = box["y"] + (box["h"] - full) / 2.0 + pt
    elif block.get("valign") == "bottom":
        y = box["y"] + box["h"] - full + pt
    out = []
    for ln in lines:
        w = measure.line_width(reg, ln, primary, language, size, ls_px)
        if block.get("align") == "left":
            x = box["x"]
        elif block.get("align") == "right":
            x = box["x"] + box["w"] - w
        else:
            x = box["x"] + (box["w"] - w) / 2.0
        out.append((ln, x, y))
        y += lh
    return out


def shadow_style(fill, size: int, bg=None) -> Tuple[Tuple[int, int, int], int,
                                                   int, float]:
    """(color, alpha 0-255, offset px, blur radius) for the text shadow.

    Text LIGHTER than what is behind it -> soft dark drop shadow (slight
    offset). Text DARKER than its background -> soft light glow with NO
    offset: a dark offset copy under dark text reads as a doubled/ghosted
    second layer instead of depth. Compared against the actual background
    (`bg`, mean color under the block) when known: an orange on a dark teal
    is the "lighter" one even though orange itself is fairly dark.
    """
    lighter = (rel_luminance(fill) >= rel_luminance(bg)) if bg is not None \
        else rel_luminance(fill) > 0.45
    if lighter:
        return (0, 0, 0), 150, max(1, round(size * 0.025)), max(1.0, size * 0.035)
    return (255, 255, 255), 170, 0, max(2.0, size * 0.05)


def _mean_under(img: Image.Image, box: Dict):
    x0, y0 = max(0, box["x"]), max(0, box["y"])
    x1 = min(img.width, box["x"] + box["w"])
    y1 = min(img.height, box["y"] + box["h"])
    if x1 <= x0 or y1 <= y0:
        return None
    small = img.crop((x0, y0, x1, y1)).resize((1, 1), Image.BOX)
    return small.getpixel((0, 0))[:3]


def draw_block(img: Image.Image, reg: FontRegistry, block: Dict,
               language: str) -> Image.Image:
    """Draw a text block onto `img` (RGB); returns the new image."""
    size = int(block["size_px"])
    primary = block["font"]
    ls_px = block.get("letter_spacing", 0.0) * size
    fill = hex_to_rgb(block.get("color", "#FFFFFF"))
    positions = _line_positions(reg, block, language)

    if block.get("shadow"):
        col, alpha, off, blur = shadow_style(fill, size,
                                             _mean_under(img, block["box"]))
        mask = Image.new("L", img.size, 0)
        md = ImageDraw.Draw(mask)
        for ln, x, y in positions:
            _draw_runs(md, reg, ln, primary, language, size, ls_px,
                       x + off, y + off, alpha)
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
        img = Image.composite(Image.new("RGB", img.size, col), img, mask)

    draw = ImageDraw.Draw(img)
    ow = int(block.get("outline_width", 0))
    oc = hex_to_rgb(block.get("outline_color", "#000000"))
    for ln, x, y in positions:
        _draw_runs(draw, reg, ln, primary, language, size, ls_px, x, y, fill,
                   stroke_w=ow, stroke_fill=oc)
    return img


SS = 2  # supersampling for smooth circles / diagonals


def _poly_star(cx, cy, r, points=14, inner=0.82):
    pts = []
    for i in range(points * 2):
        ang = math.pi * i / points - math.pi / 2
        rr = r if i % 2 == 0 else r * inner
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    return pts


def _rgba(hexcolor, opacity=1.0):
    return hex_to_rgb(hexcolor) + (int(round(255 * float(opacity))),)


def draw_shapes(img: Image.Image, layout: Dict) -> Image.Image:
    """Draw every shape (panels, badges, ribbons, frames, dividers, confetti,
    rings) on one RGBA layer at SSx resolution, then composite it."""
    shapes = layout.get("shapes", [])
    if not shapes:
        return img
    k = SS
    layer = Image.new("RGBA", (img.width * k, img.height * k), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    def S(v):
        return v * k

    for sh in shapes:
        t = sh["type"]
        if t == "panel":
            d.rounded_rectangle(
                [S(sh["x"]), S(sh["y"]), S(sh["x"] + sh["w"]),
                 S(sh["y"] + sh["h"])], radius=S(sh.get("radius", 0)),
                fill=_rgba(sh["fill"], sh.get("opacity", 1.0)))
        elif t == "ribbon":
            x, y, w, h, n = (S(sh[q]) for q in ("x", "y", "w", "h", "notch"))
            d.polygon([(x, y), (x + w, y), (x + w - n, y + h / 2),
                       (x + w, y + h), (x, y + h), (x + n, y + h / 2)],
                      fill=_rgba(sh["fill"]))
        elif t == "badge":
            fill = _rgba(sh["fill"])
            kind = sh.get("shape", "circle")
            if kind == "pill":
                d.rounded_rectangle(
                    [S(sh["x"]), S(sh["y"]), S(sh["x"] + sh["w"]),
                     S(sh["y"] + sh["h"])], radius=S(sh["h"]) // 2, fill=fill)
            else:
                cx, cy, r = S(sh["cx"]), S(sh["cy"]), S(sh["r"])
                if kind == "star":
                    d.polygon(_poly_star(cx, cy, r), fill=fill)
                elif kind == "diamond":
                    d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r),
                               (cx - r, cy)], fill=fill)
                else:
                    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        elif t == "border":
            col, w = _rgba(sh["stroke"]), S(int(sh.get("width", 3)))
            rects = [(0, w)]
            if sh.get("style") == "double":
                rects.append((S(sh.get("gap", 10)), max(k, w // 2)))
            for off, ww in rects:
                box = [S(sh["x"]) + off, S(sh["y"]) + off,
                       S(sh["x"] + sh["w"]) - off, S(sh["y"] + sh["h"]) - off]
                if sh.get("radius"):
                    d.rounded_rectangle(box, radius=S(sh["radius"]),
                                        outline=col, width=ww)
                else:
                    d.rectangle(box, outline=col, width=ww)
        elif t == "corners":
            col, w, L = _rgba(sh["stroke"]), S(int(sh.get("width", 3))),                 S(sh["len"])
            x0, y0 = S(sh["x"]), S(sh["y"])
            x1, y1 = S(sh["x"] + sh["w"]), S(sh["y"] + sh["h"])
            for (cx, cy, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1),
                                     (x0, y1, 1, -1), (x1, y1, -1, -1)]:
                d.line([(cx, cy), (cx + dx * L, cy)], fill=col, width=w)
                d.line([(cx, cy), (cx, cy + dy * L)], fill=col, width=w)
        elif t == "divider":
            col, w = _rgba(sh["stroke"]), S(int(sh.get("width", 2)))
            x0, x1, y = S(sh["x0"]), S(sh["x1"]), S(sh["y"])
            cx, o = (x0 + x1) / 2.0, S(sh.get("ornament_size", 6))
            orn = sh.get("ornament", "none")
            gap = 0 if orn == "none" else (o * 4.2 if orn == "dots" else o * 1.9)
            if gap:
                d.line([(x0, y), (cx - gap, y)], fill=col, width=w)
                d.line([(cx + gap, y), (x1, y)], fill=col, width=w)
            else:
                d.line([(x0, y), (x1, y)], fill=col, width=w)
            if orn == "diamond":
                d.polygon([(cx, y - o), (cx + o, y), (cx, y + o), (cx - o, y)],
                          fill=col)
            elif orn == "dot":
                d.ellipse([cx - o * .7, y - o * .7, cx + o * .7, y + o * .7],
                          fill=col)
            elif orn == "dots":
                for dx in (-2.4 * o, 0, 2.4 * o):
                    d.ellipse([cx + dx - o * .5, y - o * .5, cx + dx + o * .5,
                               y + o * .5], fill=col)
        elif t == "accent_bar":
            d.rounded_rectangle(
                [S(sh["x"]), S(sh["y"]), S(sh["x"] + sh["w"]),
                 S(sh["y"] + sh["h"])], radius=S(sh["h"]) // 2,
                fill=_rgba(sh["fill"]))
        elif t == "dots":
            op = sh.get("opacity", 1.0)
            for (x, y, r), f in zip(sh["points"], sh["fills"]):
                d.ellipse([S(x - r), S(y - r), S(x + r), S(y + r)],
                          fill=_rgba(f, op))
        elif t == "ring":
            cx, cy, r = S(sh["cx"]), S(sh["cy"]), S(sh["r"])
            d.ellipse([cx - r, cy - r, cx + r, cy + r],
                      outline=_rgba(sh["stroke"]),
                      width=S(int(sh.get("width", 3))))
        elif t == "line":
            d.line([S(sh["x0"]), S(sh["y0"]), S(sh["x1"]), S(sh["y1"])],
                   fill=_rgba(sh.get("stroke", "#FFFFFF")),
                   width=S(int(sh.get("width", 2))))
    layer = layer.resize(img.size, Image.LANCZOS)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def _draw_scrims(img: Image.Image, layout: Dict) -> Image.Image:
    scrims = [b for b in layout.get("blocks", []) if b.get("scrim")]
    if not scrims:
        return img
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    pad = 14
    for b in scrims:
        s = b["scrim"]
        bx = s.get("box", b["box"])
        col = hex_to_rgb(s.get("color", "#000000"))
        a = int(255 * float(s.get("opacity", 0.35)))
        rad = int(s.get("radius", 16))
        od.rounded_rectangle(
            [bx["x"] - pad, bx["y"] - pad,
             bx["x"] + bx["w"] + pad, bx["y"] + bx["h"] + pad],
            radius=rad, fill=col + (a,))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def render_design(base: Image.Image, layout: Dict, reg: FontRegistry) -> Image.Image:
    img = base.convert("RGB").copy()
    language = layout.get("language", "latin")
    img = draw_shapes(img, layout)           # panels, badges, decor
    img = _draw_scrims(img, layout)          # translucent plates behind text
    for block in layout.get("blocks", []):
        img = draw_block(img, reg, block, language)
    return img


def composite_preview(background: Image.Image, layout: Dict,
                      reg: FontRegistry) -> Image.Image:
    cw, ch = layout["canvas"]["w"], layout["canvas"]["h"]
    bg = background.convert("RGB").resize((cw, ch), Image.LANCZOS)
    return render_design(bg, layout, reg)


def debug_preview(layout: Dict, reg: FontRegistry) -> Image.Image:
    cw, ch = layout["canvas"]["w"], layout["canvas"]["h"]
    base = Image.new("RGB", (cw, ch),
                     hex_to_rgb(layout.get("background_color", "#2B2B2B")))
    draw = ImageDraw.Draw(base)
    for z in layout.get("zones", []):
        b = z["box"]
        draw.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]],
                       outline=(120, 120, 120), width=2)
    return render_design(base, layout, reg)
