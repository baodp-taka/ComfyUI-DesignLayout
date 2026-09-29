# -*- coding: utf-8 -*-
"""Post-render zone analysis (Node 3): readability against the REAL image.

For each text block: measure how busy the area behind it is (edge density +
luminance spread), extract a small palette, and pick a text color that meets a
WCAG contrast target. Busy areas get a shadow/outline or a translucent scrim.
Uses OpenCV when available, otherwise a pure-numpy fallback.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

from .decor import blend, resolve_colors

from .colors import (hex_to_rgb, rgb_to_hex, contrast_ratio, pick_text_color,
                     required_contrast, neutral_on, WHITE, BLACK,
                     rel_luminance)

CONTRAST_MARGIN = 1.1

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


def _gray(region: np.ndarray) -> np.ndarray:
    return region[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _edge_density(region: np.ndarray) -> float:
    g = _gray(region).astype(np.float32)
    if _HAS_CV2:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    else:
        gx = np.zeros_like(g)
        gy = np.zeros_like(g)
        gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
        gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.clip(mag.mean() / 64.0, 0, 1))


def _palette(region: np.ndarray, k: int = 5, iters: int = 8) -> List[Tuple[int, int, int]]:
    pts = region[..., :3].reshape(-1, 3).astype(np.float32)
    if len(pts) > 4000:
        idx = np.random.default_rng(0).choice(len(pts), 4000, replace=False)
        pts = pts[idx]
    rng = np.random.default_rng(0)
    centers = pts[rng.choice(len(pts), k, replace=False)]
    for _ in range(iters):
        d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            m = pts[lab == j]
            if len(m):
                centers[j] = m.mean(0)
    return [tuple(int(c) for c in ctr) for ctr in centers]


def check(background: Image.Image, layout: Dict, busy_threshold: float = 0.18,
          min_contrast: float = 4.5) -> Tuple[Dict, List[str]]:
    """Return (updated_layout, report_lines). Mutates a copy of layout."""
    cw, ch = layout["canvas"]["w"], layout["canvas"]["h"]
    img = np.asarray(background.convert("RGB").resize((cw, ch)), dtype=np.uint8)
    report: List[str] = []
    out = dict(layout)
    out["blocks"] = [dict(b) for b in layout.get("blocks", [])]
    out["shapes"] = [dict(s) for s in layout.get("shapes", [])]

    # palette of the WHOLE image: accent roles without an LLM intent borrow a
    # saturated color from the actual subject (e.g. the copper machine)
    image_palette = _palette(img.astype(np.float32), k=6)
    short = min(cw, ch)

    for b in out["blocks"]:
        if b.get("on_shape"):
            # text on a badge/ribbon: its colors were fitted to the shape fill
            report.append(f"{b['role']:11} on shape, color={b['color']} (kept)")
            continue
        box = b["box"]
        x0 = max(0, box["x"]); y0 = max(0, box["y"])
        x1 = min(cw, box["x"] + box["w"]); y1 = min(ch, box["y"] + box["h"])
        if x1 <= x0 or y1 <= y0:
            continue
        region = img[y0:y1, x0:x1].astype(np.float32)
        busy = _edge_density(region)
        mean_rgb = tuple(int(c) for c in region.reshape(-1, 3).mean(0))
        panel = b.get("on_panel")
        if panel:
            # text sits on a plate: what the eye sees behind it is the plate
            # blended over the image; the plate also calms a busy area
            op = float(panel.get("opacity", 1.0))
            mean_rgb = blend(hex_to_rgb(panel["fill"]), mean_rgb, op)
            busy *= (1.0 - op)
        # "enough contrast, prefer color": keep the design's hue (LLM intent)
        # and only fit its lightness; accents without intent take a hue from
        # the image; white / near-black only as the last resort. (The old
        # "highest contrast wins" rule always ended in pure white or black.)
        # +10% margin: the mean hides lighter/darker spots under the glyphs
        need = required_contrast(int(b.get("size_px", 0)), short,
                                 min_contrast) * CONTRAST_MARGIN
        intent = b.get("intent_color")
        color = pick_text_color(
            mean_rgb, hex_to_rgb(intent) if intent else None, image_palette,
            need, accent=b["role"] in ("subheadline", "emphasis"))
        b["color"] = rgb_to_hex(color)
        if int(b.get("outline_width", 0)) > 0:
            # thumbnail-style outline: always the opposite of the fill
            b["outline_color"] = rgb_to_hex(
                BLACK if rel_luminance(color) > 0.4 else WHITE)

        if busy >= busy_threshold and not b.get("outline_width"):
            b["shadow"] = True
            if busy >= busy_threshold * 2 and not b.get("scrim"):
                dark = rel_luminance(mean_rgb) > 0.5
                scrim_rgb = (0, 0, 0) if not dark else (255, 255, 255)
                b["scrim"] = {"color": rgb_to_hex(scrim_rgb), "opacity": 0.35,
                              "box": dict(box), "radius": 16}
        cr = contrast_ratio(color, mean_rgb)
        report.append(f"{b['role']:11} busy={busy:.2f} "
                      f"contrast={cr:.1f} color={b['color']}"
                      f"{' +scrim' if b.get('scrim') else ''}"
                      f"{' (on panel)' if panel else ''}")

    # decorations follow the FINAL text colors (headline = "text", first
    # accent-role line on the image = "accent")
    on_img = [b for b in out["blocks"] if not b.get("on_shape")
              and not b.get("on_panel")]
    head = next((b for b in on_img if b["role"] == "headline"), None)
    acc = next((b for b in on_img if b["role"] in ("subheadline", "emphasis")),
               None)
    global_mean = tuple(int(c) for c in img.reshape(-1, 3).mean(0))
    text_c = hex_to_rgb(head["color"]) if head else neutral_on(global_mean)
    if acc:
        accent_c = hex_to_rgb(acc["color"])
    else:
        pal = (layout.get("palette") or {}).get("accent")
        accent_c = pick_text_color(global_mean,
                                   hex_to_rgb(pal) if pal else None,
                                   image_palette, 3.0, accent=True)
    resolve_colors(out["shapes"], {"text": text_c, "accent": accent_c})
    return out, report
