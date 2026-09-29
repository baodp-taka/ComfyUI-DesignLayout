# -*- coding: utf-8 -*-
"""Panels, ribbons and decorations, built AFTER the text is placed.

Everything here is derived from the final text blocks, so decorations never
cover text: dividers / bars sit in the gap under the headline, frames sit
outside the (inset) text area, confetti avoids every text box, and a ring is
dropped if it would cross a line of text.

Shapes are plain JSON the app can draw (canvas px, hex colors). Stroke / fill
colors of decorations follow the text via "color_ref" ("text" | "accent"),
which ZoneCheck re-resolves after it has fitted the text colors to the image.

Shape types:
  panel    {x,y,w,h,fill,opacity,radius}             plate behind text
  ribbon   {x,y,w,h,fill,notch}                      notched strip, text on it
  border   {x,y,w,h,stroke,width,style,radius,gap}   single|double|rounded
  corners  {x,y,w,h,len,stroke,width}                L brackets in 4 corners
  divider  {x0,x1,y,stroke,width,ornament,ornament_size}
  accent_bar {x,y,w,h,fill}
  dots     {points:[[x,y,r],..], fills:[hex,..], opacity}
  ring     {cx,cy,r,stroke,width}
  badge    {shape: circle|star|diamond (cx,cy,r) | pill (x,y,w,h), fill}
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple

from .colors import (rgb_to_hex, hex_to_rgb, fit_contrast, contrast_ratio,
                     neutral_on, WHITE, BLACK)

RGB = Tuple[int, int, int]
ROLE_TARGETS = ("headline", "subheadline", "emphasis", "body", "detail",
                "note")
PANEL_STYLES = {"light": ((255, 255, 255), 0.86),
                "dark": ((12, 12, 16), 0.62)}


def blend(fg: RGB, bg: RGB, opacity: float) -> RGB:
    return tuple(int(round(f * opacity + b * (1 - opacity)))  # type: ignore
                 for f, b in zip(fg, bg))


def fill_for_text(base: RGB) -> Tuple[RGB, RGB]:
    """(fill, text) for a solid shape carrying text: keep the fill's hue and
    fit it so white (preferred) or near-black text reaches 4.5:1."""
    options = {}
    for txt in (WHITE, BLACK):
        fill = fit_contrast(base, txt, 4.5)
        if contrast_ratio(fill, txt) >= 4.5:
            options[txt] = (sum(abs(a - b) for a, b in zip(fill, base)), fill)
    if WHITE in options and options[WHITE][0] <= 90:
        return options[WHITE][1], WHITE
    if options:
        txt = min(options, key=lambda t: options[t][0])
        return options[txt][1], txt
    return base, neutral_on(base)


def panel_fill(panel: Optional[Dict], accent: Optional[RGB],
               default_accent: RGB) -> Optional[Tuple[RGB, float,
                                                     Optional[RGB]]]:
    """(fill, opacity, pair_text). An accent plate is a solid color with its
    own white / near-black text (like a badge) -- fitting accent-colored
    text onto an accent plate only gives muddy olive-on-yellow."""
    if not panel:
        return None
    style = panel.get("style", "light")
    if style == "accent":
        fill, txt = fill_for_text(accent or default_accent)
        return fill, 0.94, txt
    fill, op = PANEL_STYLES.get(style, PANEL_STYLES["light"])
    return fill, op, None


def text_extent(block: Dict) -> Dict:
    """Box of the actual text inside a (band-wide) block box."""
    box = block["box"]
    tw = min(box["w"], int(block.get("text_w", box["w"])))
    if block.get("align") == "left":
        x = box["x"]
    elif block.get("align") == "right":
        x = box["x"] + box["w"] - tw
    else:
        x = box["x"] + (box["w"] - tw) // 2
    return {"x": x, "y": box["y"], "w": tw, "h": box["h"]}


def _union(boxes: List[Dict]) -> Dict:
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _hits(box: Dict, others: List[Dict], pad: int = 0) -> bool:
    for o in others:
        if not (box["x"] + box["w"] + pad <= o["x"] or
                o["x"] + o["w"] + pad <= box["x"] or
                box["y"] + box["h"] + pad <= o["y"] or
                o["y"] + o["h"] + pad <= box["y"]):
            return True
    return False


def _clamp_box(b: Dict, cw: int, ch: int) -> Dict:
    x0, y0 = max(0, b["x"]), max(0, b["y"])
    x1 = min(cw, b["x"] + b["w"])
    y1 = min(ch, b["y"] + b["h"])
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


# ---- panel / ribbon ----------------------------------------------------------
def panel_shape(panel: Dict, blocks: List[Dict], bands: List[Dict],
                fill: RGB, opacity: float, cw: int, ch: int) -> Optional[Dict]:
    """Plate behind the target band(s); marks covered blocks `on_panel`."""
    target = panel.get("target", "all")
    idx = {"primary": [0], "secondary": [1]}.get(target, [0, 1])
    if target in ROLE_TARGETS:          # a strip behind one role only
        covered = [b for b in blocks if b["role"] == target
                   and not b.get("on_shape")]
        idx = sorted({b["band"] for b in covered})
    else:
        covered = [b for b in blocks if b.get("band") in idx
                   and not b.get("on_shape")]
    if not covered:
        return None
    short = min(cw, ch)
    pad = round(0.035 * short)
    ys = _union([b["box"] for b in covered])
    # a strip behind ONE role sits between neighbouring lines: its vertical
    # padding must stay inside the line gap (0.3 x size) or it touches them
    pad_y = round(0.22 * max(b["size_px"] for b in covered))         if target in ROLE_TARGETS else pad
    if panel.get("full_width"):
        box = {"x": 0, "y": ys["y"] - pad_y, "w": cw,
               "h": ys["h"] + 2 * pad_y}
        radius = 0
    else:
        xs = _union([bands[i]["box"] for i in idx if any(
            b.get("band") == i for b in covered)])
        box = {"x": xs["x"] - pad, "y": ys["y"] - pad_y,
               "w": xs["w"] + 2 * pad, "h": ys["h"] + 2 * pad_y}
        radius = round(0.025 * short)
    box = _clamp_box(box, cw, ch)
    hexfill = rgb_to_hex(fill)
    for b in covered:
        b["on_panel"] = {"fill": hexfill, "opacity": opacity}
    return dict(box, type="panel", fill=hexfill, opacity=opacity,
                radius=radius)


def ribbon_shape(block: Dict, base: RGB, cw: int, ch: int) -> Dict:
    """Notched strip behind a (sub)headline; recolors that block onto it."""
    size = int(block["size_px"])
    ext = text_extent(block)
    px, py = round(0.55 * size), round(0.20 * size)
    box = _clamp_box({"x": ext["x"] - px, "y": ext["y"] - py,
                      "w": ext["w"] + 2 * px, "h": ext["h"] + 2 * py}, cw, ch)
    fill, txt = fill_for_text(base)
    block["on_shape"] = True
    block["color"] = rgb_to_hex(txt)
    block["shadow"] = False
    block["intent_color"] = None
    return dict(box, type="ribbon", fill=rgb_to_hex(fill),
                notch=round(0.32 * box["h"]))


# ---- decorations -------------------------------------------------------------
def _headline_gap_y(blocks: List[Dict], limits: Dict) -> Optional[Tuple]:
    """(y, headline_block) for a line placed in the gap under the headline."""
    heads = [b for b in blocks if b["role"] == "headline"
             and not b.get("on_shape")]
    if not heads:
        return None
    h = heads[-1]
    hb = h["box"]
    y0 = hb["y"] + hb["h"]
    below = [b for b in blocks if b is not h and b["box"]["y"] >= y0 - 1
             and not (b["box"]["x"] + b["box"]["w"] <= hb["x"] or
                      hb["x"] + hb["w"] <= b["box"]["x"])]
    if below:
        y1 = min(b["box"]["y"] for b in below)
    else:
        y1 = min(y0 + int(0.6 * h["size_px"]), limits["y"] + limits["h"])
    if y1 - y0 < 6:
        return None
    return (y0 + y1) / 2.0, h


def _aligned_span(block: Dict, width: int) -> Tuple[int, int]:
    ext = text_extent(block)
    a = block.get("align")
    if a == "left":
        x0 = ext["x"]
    elif a == "right":
        x0 = ext["x"] + ext["w"] - width
    else:
        x0 = ext["x"] + (ext["w"] - width) // 2
    return int(x0), int(x0 + width)


def build_decor(decor: List[Dict], blocks: List[Dict], bands: List[Dict],
                limits: Dict, cw: int, ch: int, colors: Dict[str, RGB],
                seed: int, avoid: List[Dict]) -> List[Dict]:
    short = min(cw, ch)
    stroke_w = max(2, round(0.0035 * short))
    text_boxes = [b["box"] for b in blocks]
    shapes: List[Dict] = []

    def ref(name: str) -> Dict:
        return {"color_ref": name, "stroke": rgb_to_hex(colors[name]),
                "fill": rgb_to_hex(colors[name])}

    for d in decor:
        kind = d["type"]
        if kind == "border":
            ins = round(0.03 * short)
            style = d.get("style", "single")
            s = {"type": "border", "x": ins, "y": ins, "w": cw - 2 * ins,
                 "h": ch - 2 * ins, "width": stroke_w, "style": style,
                 "radius": round(0.035 * short) if style == "rounded" else 0,
                 "gap": round(0.012 * short) if style == "double" else 0}
            s.update(ref("text"))
            shapes.append(s)
        elif kind == "corners":
            ins = round(0.03 * short)
            s = {"type": "corners", "x": ins, "y": ins, "w": cw - 2 * ins,
                 "h": ch - 2 * ins, "len": round(0.09 * short),
                 "width": stroke_w}
            s.update(ref("text"))
            shapes.append(s)
        elif kind in ("divider", "accent_bar"):
            got = _headline_gap_y(blocks, limits)
            if not got:
                continue
            y, h = got
            ext = text_extent(h)
            if kind == "divider":
                width = int(max(0.20 * h["box"]["w"],
                                min(0.36 * h["box"]["w"], 0.8 * ext["w"])))
                x0, x1 = _aligned_span(h, width)
                s = {"type": "divider", "x0": x0, "x1": x1, "y": int(y),
                     "width": stroke_w, "ornament": d.get("ornament", "none"),
                     "ornament_size": max(4, round(0.011 * short))}
                probe = {"x": x0, "y": int(y) - 6, "w": x1 - x0, "h": 12}
            else:
                width = max(30, int(0.12 * h["box"]["w"]))
                bh = max(4, round(0.011 * short))
                x0, x1 = _aligned_span(h, width)
                s = {"type": "accent_bar", "x": x0, "y": int(y - bh / 2),
                     "w": width, "h": bh}
                probe = {"x": x0, "y": s["y"], "w": width, "h": bh}
            if _hits(probe, text_boxes) or _hits(probe, avoid):
                continue
            s.update(ref("accent"))
            shapes.append(s)
        elif kind == "dots":
            rng = random.Random(seed * 7919 + 17)
            pad = round(0.03 * short)
            keep_out = text_boxes + avoid
            points, fills = [], []
            for _ in range(400):
                if len(points) >= 22:
                    break
                r = rng.uniform(0.006, 0.015) * short
                x = rng.uniform(r, cw - r)
                y = rng.uniform(r, ch - r)
                box = {"x": int(x - r), "y": int(y - r), "w": int(2 * r),
                       "h": int(2 * r)}
                if _hits(box, keep_out, pad):
                    continue
                if any(math.hypot(x - px, y - py) < pr + r + pad
                       for px, py, pr in points):
                    continue
                points.append([int(x), int(y), round(r, 1)])
                fills.append("accent" if len(points) % 3 else "text")
            if points:
                shapes.append({"type": "dots", "points": points,
                               "fill_refs": fills,
                               "fills": [rgb_to_hex(colors[f]) for f in fills],
                               "opacity": 0.85})
        elif kind == "ring":
            prim = [b for b in blocks if b.get("band") == 0
                    and not b.get("on_shape")]
            if not prim:
                continue
            u = _union([text_extent(b) for b in prim])
            cx, cy = u["x"] + u["w"] / 2.0, u["y"] + u["h"] / 2.0
            r = math.hypot(u["w"], u["h"]) / 2.0 + 0.04 * short
            r = min(r, cx, cw - cx, cy, ch - cy) - stroke_w
            if r < math.hypot(u["w"], u["h"]) / 2.0 + 4:
                continue  # would cut through the text
            others = [text_extent(b) for b in blocks if b not in prim]
            if any(_crosses_circle(o, cx, cy, r, stroke_w) for o in others):
                continue
            s = {"type": "ring", "cx": int(cx), "cy": int(cy), "r": int(r),
                 "width": max(stroke_w, round(0.005 * short))}
            s.update(ref("accent"))
            shapes.append(s)
    return shapes


def _crosses_circle(b: Dict, cx: float, cy: float, r: float,
                    w: float) -> bool:
    """True if the circle outline (radius r, width w) cuts through box b."""
    nx = min(max(cx, b["x"]), b["x"] + b["w"])
    ny = min(max(cy, b["y"]), b["y"] + b["h"])
    near = math.hypot(nx - cx, ny - cy)
    far = max(math.hypot(px - cx, py - cy)
              for px in (b["x"], b["x"] + b["w"])
              for py in (b["y"], b["y"] + b["h"]))
    return near <= r + w and far >= r - w


def resolve_colors(shapes: List[Dict], colors: Dict[str, RGB]) -> None:
    """Re-apply color_ref / fill_refs after ZoneCheck fitted the text."""
    for s in shapes:
        ref = s.get("color_ref")
        if ref in colors:
            s["stroke"] = s["fill"] = rgb_to_hex(colors[ref])
        if s.get("fill_refs"):
            s["fills"] = [rgb_to_hex(colors.get(f, colors["text"]))
                          for f in s["fill_refs"]]
