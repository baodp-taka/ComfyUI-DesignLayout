# -*- coding: utf-8 -*-
"""Color helpers: hex/rgb, WCAG luminance + contrast, on-color pick."""
from __future__ import annotations
import colorsys
from typing import Iterable, Optional, Tuple

RGB = Tuple[int, int, int]
WHITE: RGB = (255, 255, 255)
BLACK: RGB = (17, 17, 17)


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


# ---- hue-preserving color fitting -------------------------------------------
def _to_hls(rgb: RGB) -> Tuple[float, float, float]:
    return colorsys.rgb_to_hls(*(c / 255.0 for c in rgb))


def _from_hls(h: float, l: float, s: float) -> RGB:
    return tuple(int(round(c * 255)) for c in colorsys.hls_to_rgb(h, l, s))  # type: ignore


def saturation(rgb: RGB) -> float:
    """HLS saturation 0..1 (0 = gray)."""
    return _to_hls(rgb)[2]


def neutral_on(bg: RGB) -> RGB:
    """White or near-black, whichever reads better on bg."""
    return BLACK if contrast_ratio(bg, BLACK) >= contrast_ratio(bg, WHITE) \
        else WHITE


def fit_contrast(fg: RGB, bg: RGB, min_ratio: float) -> RGB:
    """Keep fg's hue + saturation; change ONLY its lightness, by the smallest
    amount that reaches `min_ratio` on bg (tries lighter and darker). Falls
    back to white/black when no lightness of that hue can make it.

    This keeps a designer's color intent (a coffee gold stays gold, a blush
    pink stays pink) while guaranteeing readability.
    """
    if contrast_ratio(fg, bg) >= min_ratio:
        return fg
    h, l0, s = _to_hls(fg)
    best: Optional[Tuple[float, RGB]] = None
    for end in (1.0, 0.0):                       # towards lighter / darker
        if contrast_ratio(_from_hls(h, end, s), bg) < min_ratio:
            continue                             # not reachable this way
        a, b = l0, end
        for _ in range(24):                      # binary search the lightness
            m = (a + b) / 2.0
            if contrast_ratio(_from_hls(h, m, s), bg) >= min_ratio:
                b = m
            else:
                a = m
        cand = _from_hls(h, b, s)
        if best is None or abs(b - l0) < best[0]:
            best = (abs(b - l0), cand)
    return best[1] if best else neutral_on(bg)


def _hue_distance(a: RGB, b: RGB) -> float:
    """Circular hue distance 0..1 (1 = opposite hues)."""
    d = abs(_to_hls(a)[0] - _to_hls(b)[0])
    return min(d, 1.0 - d) * 2.0


def fit_intent(intent: RGB, bg: RGB, min_ratio: float,
               max_shift: float = 0.30, min_sat: float = 0.25) -> RGB:
    """fit_contrast, but a near-neutral intent (white, cream, gray) that has
    to cross to the other side of the background becomes plain white /
    near-black instead of a muddy gray: cream text on a white plate would
    otherwise turn into beige-gray. Saturated hues (pink, gold) keep their
    hue however far they move."""
    fitted = fit_contrast(intent, bg, min_ratio)
    if saturation(intent) < min_sat and \
            abs(_to_hls(fitted)[1] - _to_hls(intent)[1]) > max_shift:
        return neutral_on(bg)
    return fitted


def accent_from_palette(palette: Iterable[RGB], bg: RGB, min_ratio: float,
                        min_saturation: float = 0.25) -> Optional[RGB]:
    """A saturated palette color that STANDS OUT from bg, lightness-fitted.

    Scored by saturation x hue distance from the background, so on a teal
    backdrop an orange subject beats another teal (same-hue accents read as
    part of the background). On a grayish bg the hue term is ignored.
    """
    gray_bg = saturation(bg) < 0.15

    def score(c: RGB) -> float:
        if gray_bg:
            return saturation(c)
        return saturation(c) * (0.35 + 0.65 * _hue_distance(c, bg))

    colorful = sorted((c for c in palette if saturation(c) >= min_saturation),
                      key=score, reverse=True)
    for c in colorful:
        fitted = fit_contrast(c, bg, min_ratio)
        if saturation(fitted) >= min_saturation * 0.5:  # still has a hue
            return fitted
    return None


def pick_text_color(bg: RGB, intent: Optional[RGB], palette: Iterable[RGB],
                    min_ratio: float, accent: bool) -> RGB:
    """Enough contrast, preferring a colored choice over plain white/black.

    1. the designer/LLM intent color, lightness-fitted to min_ratio;
    2. for accent roles, a saturated color taken from the image palette;
    3. white / near-black.
    """
    if intent is not None:
        return fit_intent(intent, bg, min_ratio)
    if accent:
        # an auto-picked accent must POP, not just pass: aim for 4.5:1 even
        # for large text (image colors are often muddied by the background)
        acc = accent_from_palette(palette, bg, max(min_ratio, 4.5))
        if acc is not None:
            return acc
    return neutral_on(bg)


def required_contrast(size_px: int, canvas_short: int,
                      min_contrast: float = 4.5) -> float:
    """WCAG-style target: large text (>= 4% of the canvas short side, about
    43px on a 1080 canvas) may use 3.0; normal text needs `min_contrast`."""
    return min(3.0, min_contrast) if size_px >= 0.04 * canvas_short \
        else min_contrast


def parse_hex(v) -> Optional[str]:
    """'#abc123' / 'abc123' / '#ABC' -> '#ABC123'; anything else -> None."""
    s = str(v or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return None
    try:
        int(s, 16)
    except ValueError:
        return None
    return "#" + s.upper()
