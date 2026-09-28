# -*- coding: utf-8 -*-
"""Color helpers: hex/rgb, WCAG luminance + contrast, on-color pick."""
from __future__ import annotations
from typing import Tuple

RGB = Tuple[int, int, int]


def hex_to_rgb(h: str) -> RGB:
    h = (h or "#000000").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        h = "000000"
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore


def rgb_to_hex(rgb: RGB) -> str:
    return "#%02X%02X%02X" % tuple(max(0, min(255, int(c))) for c in rgb)


def _lin(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def rel_luminance(rgb: RGB) -> float:
    r, g, b = (_lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: RGB, b: RGB) -> float:
    la, lb = rel_luminance(a), rel_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def best_on_color(bg: RGB, candidates=None, min_ratio: float = 4.5) -> RGB:
    """Pick the palette color with best contrast on bg; fallback white/black."""
    white, black = (255, 255, 255), (17, 17, 17)
    base = black if contrast_ratio(bg, black) >= contrast_ratio(bg, white) \
        else white
    best, best_c = base, contrast_ratio(bg, base)
    for c in candidates or []:
        cr = contrast_ratio(bg, c)
        if cr >= min_ratio and cr > best_c:
            best, best_c = c, cr
    return best


def ensure_contrast(fg: RGB, bg: RGB, min_ratio: float = 4.5) -> RGB:
    if contrast_ratio(fg, bg) >= min_ratio:
        return fg
    white, black = (255, 255, 255), (17, 17, 17)
    return white if contrast_ratio(bg, white) >= contrast_ratio(bg, black) \
        else black
