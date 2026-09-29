# -*- coding: utf-8 -*-
"""Draw real text (and shapes) for debug + composite previews.

Rendering happens at CANVAS resolution. `composite_preview` scales the given
background up to the canvas and draws the design on top; `debug_preview` draws
on a plain background_color canvas with zone outlines for quick inspection.
"""
from __future__ import annotations
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
    y = box["y"]
    if block.get("valign") == "middle":
        y = box["y"] + (box["h"] - lh * len(lines)) / 2.0
    elif block.get("valign") == "bottom":
        y = box["y"] + box["h"] - lh * len(lines)
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


def shadow_style(fill, size: int) -> Tuple[Tuple[int, int, int], int, int,
                                          float]:
    """(color, alpha 0-255, offset px, blur radius) adapted to the text color.

    Light text -> soft dark drop shadow (slight offset). Dark text -> soft
    light glow with NO offset: a dark offset copy under dark text reads as a
    doubled/ghosted second layer instead of depth.
    """
    if rel_luminance(fill) > 0.45:
        return (0, 0, 0), 150, max(1, round(size * 0.025)), max(1.0, size * 0.035)
    return (255, 255, 255), 170, 0, max(2.0, size * 0.05)


def draw_block(img: Image.Image, reg: FontRegistry, block: Dict,
               language: str) -> Image.Image:
    """Draw a text block onto `img` (RGB); returns the new image."""
    size = int(block["size_px"])
    primary = block["font"]
    ls_px = block.get("letter_spacing", 0.0) * size
    fill = hex_to_rgb(block.get("color", "#FFFFFF"))
    positions = _line_positions(reg, block, language)

    if block.get("shadow"):
        col, alpha, off, blur = shadow_style(fill, size)
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


def draw_shapes(draw: ImageDraw.ImageDraw, layout: Dict) -> None:
    for s in layout.get("shapes", []):
        if s["type"] == "badge" and s.get("shape") == "circle":
            r = s["r"]
            draw.ellipse([s["cx"] - r, s["cy"] - r, s["cx"] + r, s["cy"] + r],
                         fill=hex_to_rgb(s["fill"]))
        elif s["type"] == "border":
            draw.rectangle([s["x"], s["y"], s["x"] + s["w"], s["y"] + s["h"]],
                           outline=hex_to_rgb(s["stroke"]),
                           width=int(s.get("width", 3)))
        elif s["type"] == "line":
            draw.line([s["x0"], s["y0"], s["x1"], s["y1"]],
                      fill=hex_to_rgb(s.get("stroke", "#FFFFFF")),
                      width=int(s.get("width", 2)))


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
    draw = ImageDraw.Draw(img)
    draw_shapes(draw, layout)
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
