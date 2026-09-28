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

from .colors import (hex_to_rgb, rgb_to_hex, contrast_ratio, best_on_color,
                     rel_luminance)

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

    for b in out["blocks"]:
        box = b["box"]
        x0 = max(0, box["x"]); y0 = max(0, box["y"])
        x1 = min(cw, box["x"] + box["w"]); y1 = min(ch, box["y"] + box["h"])
        if x1 <= x0 or y1 <= y0:
            continue
        region = img[y0:y1, x0:x1].astype(np.float32)
        busy = _edge_density(region)
        mean_rgb = tuple(int(c) for c in region.reshape(-1, 3).mean(0))
        pal = _palette(region)

        want_accent = b["role"] in ("subheadline", "emphasis")
        cur = hex_to_rgb(b.get("color", "#FFFFFF"))
        cands = pal + ([cur] if want_accent else [])
        color = best_on_color(mean_rgb, cands, min_contrast)
        b["color"] = rgb_to_hex(color)

        if busy >= busy_threshold:
            b["shadow"] = True
            if busy >= busy_threshold * 2 and not b.get("scrim"):
                dark = rel_luminance(mean_rgb) > 0.5
                scrim_rgb = (0, 0, 0) if not dark else (255, 255, 255)
                b["scrim"] = {"color": rgb_to_hex(scrim_rgb), "opacity": 0.35,
                              "box": dict(box), "radius": 16}
        cr = contrast_ratio(color, mean_rgb)
        report.append(f"{b['role']:11} busy={busy:.2f} "
                      f"contrast={cr:.1f} color={b['color']}"
                      f"{' +scrim' if b.get('scrim') else ''}")
    return out, report
