# -*- coding: utf-8 -*-
"""Guide image + text mask + augmented background prompt.

guide_image is fed to Qwen-Image-Edit's image1 so the model keeps the text
zones as calm, empty surfaces. It is rendered at the BACKGROUND size (default
1024) while the layout is on the CANVAS (default 1080), so coordinates are
scaled.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFilter

from .colors import hex_to_rgb, rel_luminance

Region = Tuple[str, str]


def _shift(rgb, amt):
    return tuple(max(0, min(255, int(c + amt))) for c in rgb)


def auto_bg_size(cw: int, ch: int, megapixels: float = 1.0,
                 multiple: int = 16, max_side: int = 2048) -> Tuple[int, int]:
    """Background (latent) size with the canvas aspect ratio at ~`megapixels`.

    Qwen-Image works best near 1 MP; both sides are rounded to `multiple`
    (latent 8 px x 2x2 patch) and the long side is capped at `max_side`.
    A 1080x1080 canvas gives 1024x1024.
    """
    ar = cw / float(ch)
    area = megapixels * 1024 * 1024
    w = (area * ar) ** 0.5
    h = w / ar
    if max(w, h) > max_side:
        k = max_side / max(w, h)
        w, h = w * k, h * k
    w = max(multiple * 16, int(round(w / multiple)) * multiple)
    h = max(multiple * 16, int(round(h / multiple)) * multiple)
    return w, h


def _scale_box(box: Dict, sx: float, sy: float, pad: int) -> Tuple[int, ...]:
    x = int(box["x"] * sx) - pad
    y = int(box["y"] * sy) - pad
    w = int(box["w"] * sx) + 2 * pad
    h = int(box["h"] * sy) + 2 * pad
    return x, y, x + w, y + h


def build_guide(layout: Dict, bg_w: int = 1024, bg_h: int = 1024,
                guide_mode: str = "flat", feather_px: int = 24,
                zone_padding: int = 18) -> Tuple[Image.Image, Image.Image]:
    """Return (guide_image RGB, text_mask L), both bg_w x bg_h."""
    cw = layout["canvas"]["w"]
    ch = layout["canvas"]["h"]
    sx, sy = bg_w / cw, bg_h / ch
    bg_rgb = hex_to_rgb(layout.get("background_color", "#2B2B2B"))

    if guide_mode == "soft_gradient":
        top = _shift(bg_rgb, 18)
        bot = _shift(bg_rgb, -18)
        base = Image.new("RGB", (bg_w, bg_h))
        px = base.load()
        for y in range(bg_h):
            t = y / max(1, bg_h - 1)
            col = tuple(int(top[i] * (1 - t) + bot[i] * t) for i in range(3))
            for x in range(bg_w):
                px[x, y] = col
    else:
        base = Image.new("RGB", (bg_w, bg_h), bg_rgb)

    # plate color: nudge away from bg so the zone reads as a plain surface
    light = rel_luminance(bg_rgb) > 0.5
    plate = _shift(bg_rgb, -22 if light else 22)

    mask = Image.new("L", (bg_w, bg_h), 0)
    mdraw = ImageDraw.Draw(mask)
    plate_layer = base.copy()
    pdraw = ImageDraw.Draw(plate_layer)

    if guide_mode != "none":
        for z in layout.get("zones", []):
            x0, y0, x1, y1 = _scale_box(z["box"], sx, sy, zone_padding)
            pdraw.rectangle([x0, y0, x1, y1], fill=plate)
            mdraw.rectangle([x0, y0, x1, y1], fill=255)

    if feather_px > 0:
        blur = mask.filter(ImageFilter.GaussianBlur(feather_px))
    else:
        blur = mask
    guide = Image.composite(plate_layer, base, blur)
    return guide, blur


def _region_words(box: Dict, cw: int, ch: int) -> Region:
    cx = (box["x"] + box["w"] / 2) / cw
    cy = (box["y"] + box["h"] / 2) / ch
    vert = ("upper" if cy < 0.34 else "lower" if cy > 0.66 else "central")
    horiz = ("left" if cx < 0.34 else "right" if cx > 0.66 else "center")
    return vert, horiz


def augmented_prompt(layout: Dict, base_prompt: str, max_phrases: int = 3) -> str:
    cw, ch = layout["canvas"]["w"], layout["canvas"]["h"]
    seen: List[str] = []
    for z in layout.get("zones", []):
        vert, horiz = _region_words(z["box"], cw, ch)
        phrase = (f"the {vert} {horiz} area is a smooth, plain, empty surface "
                  f"with nothing on it")
        if phrase not in seen:
            seen.append(phrase)
        if len(seen) >= max_phrases:
            break
    if not seen:
        return base_prompt
    tail = " Keep the main subject away from these areas: " + "; ".join(seen)
    tail += "."
    base = base_prompt.rstrip()
    if base and not base.endswith("."):
        base += "."
    return base + tail
