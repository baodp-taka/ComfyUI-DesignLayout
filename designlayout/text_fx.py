# -*- coding: utf-8 -*-
"""Text effect presets for the image-first flow.

Every preset is a recipe driven by ONE base color (shades are derived from it),
so the look varies with the colors instead of being baked in:

    metal3d      extruded, beveled metallic face (gold / copper / blue metal...)
    neon         glowing tube (dark backgrounds only)
    candy        white + dark double outline around a glossy face
    longshadow   flat face with a short diagonal shadow in the background tone
    outline_pop  thick dark outline with a hard offset shadow

`style_layer` picks the preset (mood + background brightness + seed) and the
colors (LLM intent + palette of the real background + harmony partners +
neutrals), never a color that blends into the region behind the headline. It
returns an RGBA layer to composite over the background; the glyphs come from
the same renderer as the flat text, so spelling / accents are always exact.
"""
from __future__ import annotations

import colorsys
import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from .colors import contrast_ratio, fit_contrast, hex_to_rgb, rgb_to_hex
from .fonts import FontRegistry
from .render_preview import draw_block

MODERN = ("inflated", "chrome", "glass", "holo", "gradient")
CLASSIC = ("metal3d", "neon", "candy", "longshadow", "outline_pop")
PRESETS = MODERN + CLASSIC
EFFECT_CHOICES = ["none", "auto"] + list(PRESETS)
EFFECT_ROLES = ("headline", "emphasis", "subheadline")
# modern looks first (listed twice = picked twice as often as a classic one)
BY_MOOD = {
    "festive": ["chrome", "inflated", "holo", "gradient", "chrome", "inflated",
                "holo", "neon", "metal3d"],
    "playful": ["inflated", "holo", "gradient", "inflated", "holo", "candy",
                "outline_pop"],
    "elegant": ["chrome", "glass", "gradient", "chrome", "glass", "metal3d"],
    "vintage": ["metal3d", "chrome", "gradient", "longshadow"],
    "bold": ["chrome", "gradient", "inflated", "chrome", "neon", "outline_pop"],
    "minimal": ["glass", "gradient", "glass", "longshadow"],
}
NO_OUTLINE = {"longshadow", "gradient", "glass"}
# glow / see-through looks vanish on light backgrounds
DARK_ONLY = {"neon", "glass"}
# which harmony partners suit the mood (hue offsets in turns)
HARMONY = {"festive": (0.5, 1 / 3, -1 / 3, 1 / 12, -1 / 12),
           "bold": (0.5, 1 / 3, -1 / 3, 1 / 12, -1 / 12)}
SOFT_HARMONY = (0.5, 1 / 12, -1 / 12)
NEUTRALS = ["#FFF6E5", "#3A2626", "#1F2A44"]
NEON_MAX_LUMA = 0.55            # mean background luminance above this: no neon


# ---- color helpers ------------------------------------------------------------
def _hls(c: str) -> Tuple[float, float, float]:
    r, g, b = [v / 255 for v in hex_to_rgb(c)]
    return colorsys.rgb_to_hls(r, g, b)


def _from_hls(h: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h % 1, min(1, max(0, l)), min(1, max(0, s)))
    return rgb_to_hex((round(r * 255), round(g * 255), round(b * 255)))


def shade(c: str, dl: float) -> str:
    h, l, s = _hls(c)
    return _from_hls(h, l + dl, s)


def is_neutral(c: str) -> bool:
    """Near white / near black / gray: hue means nothing, judge by contrast."""
    h, l, s = _hls(c)
    return s < 0.2 or l > 0.88 or l < 0.25


def hue_dist(a: str, b: str) -> float:
    d = abs(_hls(a)[0] - _hls(b)[0]) % 1
    return min(d, 1 - d)


def harmonies(c: str, mood: str) -> List[str]:
    h, l, s = _hls(c)
    s, l = max(s, 0.6), min(max(l, 0.45), 0.6)
    return [_from_hls(h + d, l, s) for d in HARMONY.get(mood, SOFT_HARMONY)]


def stands_out(c: str, region: str) -> bool:
    """Readable on `region`: strong contrast, or a clearly different hue with
    at least moderate contrast (two dark colors of different hue still blur)."""
    cr = contrast_ratio(hex_to_rgb(c), hex_to_rgb(region))
    # a dark navy sky still has a hue: the region is neutral only when grayish
    if is_neutral(c) or _hls(region)[2] < 0.2:
        return cr >= 4.5
    if hue_dist(c, region) >= 0.12:
        return cr >= 3.0
    # same hue as the region: only a vivid, strongly contrasting tone (deep pink
    # on light pink yes, dull gray-blue on a blue sky no)
    return cr >= 4.5 and _hls(c)[2] >= 0.5


def deepened(c: str, region: str, ratio: float = 4.5) -> str:
    """Same hue, lightness moved until it reads on `region` (deep pink on light
    pink, pale gold on navy...)."""
    return rgb_to_hex(fit_contrast(hex_to_rgb(c), hex_to_rgb(region), ratio))


def bg_palette(img: Image.Image, n: int = 8) -> List[str]:
    """Colorful dominant colors of the image, most frequent first."""
    small = img.convert("RGB").resize((160, max(1, int(160 * img.height / img.width))))
    q = small.quantize(n)
    pal = q.getpalette()[:n * 3]
    out = []
    for _, idx in sorted(q.getcolors(), reverse=True):
        c = rgb_to_hex(tuple(pal[idx * 3: idx * 3 + 3]))
        if not is_neutral(c) and 0.2 < _hls(c)[1] < 0.85:
            out.append(c)
    return out


def region_color(img: Image.Image, box: Dict) -> str:
    x, y = max(0, int(box["x"])), max(0, int(box["y"]))
    a = np.asarray(img.convert("RGB").crop((x, y, x + max(1, int(box["w"])),
                                            y + max(1, int(box["h"])))), np.float32)
    return rgb_to_hex(tuple(int(v) for v in a.reshape(-1, 3).mean(0)))



def _lab(rgb_arr: np.ndarray) -> np.ndarray:
    """sRGB (N,3, 0..255) -> CIE Lab (D65)."""
    c = rgb_arr.astype(np.float32) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], 1)


def salient_colors(img: Image.Image, n: int = 24, min_share: float = 0.004,
                   top: int = 4) -> List[str]:
    """Colors that STAND OUT from the image's overall color, most salient first.

    The image is quantized to `n` clusters; the overall color is the biggest
    cluster. A cluster scores high when it is far from that color (Lab
    distance), vivid (chroma) and present -- but a small area is fine, so gold
    balloons on a pink backdrop or warm stage lights on a night sky win."""
    small = img.convert("RGB").resize((200, max(1, int(200 * img.height / img.width))))
    q = small.quantize(n, method=Image.Quantize.MEDIANCUT)
    idx_map = np.asarray(q).reshape(-1)
    px = np.asarray(small, np.float32).reshape(-1, 3)
    px_lab = _lab(px)
    px_chroma = np.hypot(px_lab[:, 1], px_lab[:, 2])
    k = int(idx_map.max()) + 1
    share = np.bincount(idx_map, minlength=k).astype(np.float32) / len(idx_map)
    # cluster color = its VIVID pixels (top 30% chroma), not the plain mean:
    # the mean of a gold balloon mixes in its shadows and turns dull tan
    pal = np.zeros((k, 3), np.float32)
    for i in range(k):
        m = idx_map == i
        if not m.any():
            continue
        ch = px_chroma[m]
        sel = ch >= np.quantile(ch, 0.7)
        pal[i] = np.median(px[m][sel], axis=0)
    lab = _lab(pal)
    dom = int(share.argmax())
    dist = np.linalg.norm(lab - lab[dom], axis=1)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    score = dist * (chroma / 100 + 0.3) * np.power(np.maximum(share, 1e-6), 0.3)
    out: List[str] = []
    for i in np.argsort(-score):
        if i == dom or share[i] < min_share or chroma[i] < 18 or dist[i] < 20:
            continue
        c = rgb_to_hex(tuple(int(v) for v in pal[i]))
        if all(hue_dist(c, o) >= 0.04 or abs(_hls(c)[1] - _hls(o)[1]) > 0.2
               for o in out):                       # skip near-duplicates
            out.append(c)
        if len(out) >= top:
            break
    return out

# ---- float-mask helpers ------------------------------------------------------
def _f3(c: str) -> np.ndarray:
    return np.array(hex_to_rgb(c), np.float32) / 255


def _pil(a):
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


def _arr(im):
    return np.asarray(im, np.float32) / 255


def _shift(a, dx, dy):
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        a[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def _dilate(a, r):
    return _arr(_pil(a).filter(ImageFilter.MaxFilter(2 * r + 1))) if r > 0 else a


def _blur(a, r):
    return _arr(_pil(a).filter(ImageFilter.GaussianBlur(r)))


def _vgrad(a, stops):
    h, w = a.shape
    ys = np.where(a.max(1) > 0.1)[0]
    y0, y1 = (ys.min(), ys.max()) if len(ys) else (0, h - 1)
    t = np.clip((np.arange(h) - y0) / max(1, y1 - y0), 0, 1)
    cols = np.stack([_f3(c) for _, c in stops])
    col = np.stack([np.interp(t, [s for s, _ in stops], cols[:, k])
                    for k in range(3)], 1)
    return np.broadcast_to(col[:, None, :], (h, w, 3))


class Layer:
    """RGBA accumulator, painter's algorithm (straight alpha)."""

    def __init__(self, h: int, w: int):
        self.c = np.zeros((h, w, 3), np.float32)
        self.a = np.zeros((h, w), np.float32)

    def paint(self, color, alpha):
        if np.ndim(color) == 1:
            color = np.broadcast_to(color, self.c.shape)
        a = np.clip(alpha, 0, 1)
        na = a + self.a * (1 - a)
        keep = (self.a * (1 - a))[..., None]
        self.c = np.where(na[..., None] > 1e-6,
                          (color * a[..., None] + self.c * keep)
                          / np.maximum(na, 1e-6)[..., None], self.c)
        self.a = na

    def image(self) -> Image.Image:
        rgba = np.dstack([self.c, self.a[..., None]]) * 255
        return Image.fromarray(rgba.clip(0, 255).astype(np.uint8), "RGBA")


# ---- presets (A = glyph alpha, c = base color, bg = region color) -------------
def _metal3d(L, A, size, c, bg):
    d = max(3, int(size * 0.07))
    L.paint(_f3("#000000"), _blur(_shift(A, d + 4, d + 6), size * 0.08) * 0.4)
    for i in range(d, 0, -1):
        L.paint(_f3(shade(c, -0.32 - 0.08 * i / d)), _shift(A, i, i))
    L.paint(_f3(shade(c, -0.42)), _dilate(A, max(1, size // 40)))
    L.paint(_vgrad(A, [(0, shade(c, 0.32)), (0.45, c), (0.55, shade(c, -0.12)),
                       (1, shade(c, 0.2))]), A)
    k = max(1, size // 28)
    L.paint(_f3("#FFFFFF"), np.clip(A - _shift(A, k, k), 0, 1) * 0.75)


def _neon(L, A, size, c, bg):
    c = _from_hls(_hls(c)[0], 0.6, 1.0)
    L.paint(_f3(c), _blur(_dilate(A, max(2, size // 18)), size * 0.35) * 0.9)
    L.paint(_f3(c), _blur(_dilate(A, max(1, size // 30)), size * 0.10))
    L.paint(_f3(shade(c, 0.2)), A)
    L.paint(_f3("#FFFFFF"), np.clip(A - _dilate(1 - A, max(1, size // 22)), 0, 1) * 0.9)


def _candy(L, A, size, c, bg):
    o1, o2 = max(2, int(size * 0.07)), max(3, int(size * 0.12))
    L.paint(_f3("#000000"), _blur(_shift(_dilate(A, o2), 0, int(size * 0.08)),
                                  size * 0.06) * 0.3)
    L.paint(_f3(shade(c, -0.3)), _dilate(A, o2))
    L.paint(_f3("#FFFFFF"), _dilate(A, o1))
    L.paint(_vgrad(A, [(0, shade(c, 0.3)), (0.2, c), (1, shade(c, -0.08))]), A)
    L.paint(_f3("#FFFFFF"), np.clip(A - _shift(A, 0, max(2, size // 10)), 0, 1) * 0.5)


def _longshadow(L, A, size, c, bg):
    n = max(4, int(size * 0.35))
    sh = np.zeros_like(A)
    for i in range(1, n, 2):
        sh = np.maximum(sh, _shift(A, i, i) * (1 - i / n * 0.6))
    L.paint(_f3(shade(bg, -0.18)), sh)
    L.paint(_vgrad(A, [(0, shade(c, 0.18)), (1, c)]), A)


def _outline_pop(L, A, size, c, bg):
    o = max(3, int(size * 0.09))
    off = max(2, size // 18)
    L.paint(_f3(shade(c, -0.38)), _shift(_dilate(A, o), off, off))
    L.paint(_f3(shade(c, -0.38)), _dilate(A, o))
    L.paint(_f3(c), A)



# ---- modern presets: height map -> normals -> lighting --------------------------
# One surface model for every material: the glyph's distance-to-edge becomes a
# height field (rounded "puffy" profile), its gradient the surface normal; each
# material is just a different way of lighting that normal.
try:                                                    # scipy ships with ComfyUI
    from scipy.ndimage import distance_transform_edt as _edt
except Exception:                                       # pragma: no cover
    _edt = None

_LIGHT = np.array([-0.45, -0.65, 0.62], np.float32)
_LIGHT /= np.linalg.norm(_LIGHT)


def _distance(mask: np.ndarray) -> np.ndarray:
    if _edt is not None:
        return _edt(mask).astype(np.float32)
    d = mask.astype(np.float32)                         # blur fallback
    acc = np.zeros_like(d)
    for r in (1, 2, 4, 8, 16):
        acc += _blur(d, r)
    return acc * 4 * d


def _surface(A: np.ndarray, radius: float, puff: float = 1.0):
    """(height, nx, ny, nz) of a rounded bevel of `radius` px over glyphs A."""
    inside = A > 0.5
    d = _distance(inside)
    t = np.clip(d / max(1.0, radius), 0, 1)
    h = np.sqrt(1 - (1 - t) ** 2) * radius * puff        # quarter-circle profile
    h = _blur(h / max(1e-6, h.max() or 1), 1.2) * radius * puff
    gy, gx = np.gradient(h)
    n = np.stack([-gx, -gy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return h, n[..., 0], n[..., 1], n[..., 2]


def _lit(nx, ny, nz, shininess=40.0):
    diff = np.clip(nx * _LIGHT[0] + ny * _LIGHT[1] + nz * _LIGHT[2], 0, 1)
    hv = _LIGHT + np.array([0, 0, 1], np.float32)
    hv /= np.linalg.norm(hv)
    spec = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1) ** shininess
    return diff, spec


def _soft_shadow(L, A, size, strength=0.35, dy=0.06, r=0.07):
    L.paint(_f3("#000000"), _blur(_shift(A, 0, max(1, int(size * dy))), size * r) * strength)


def _inflated(L, A, size, c, ctx):
    """Puffy glossy plastic / balloon letters. The glyph is NOT thickened:
    on heavy faces that closes small counters ("é" read as "ó")."""
    _, nx, ny, nz = _surface(A, size * 0.22, 1.2)
    diff, spec = _lit(nx, ny, nz, 30)
    base = _f3(c)
    dark, light = _f3(shade(c, -0.22)), _f3(shade(c, 0.18))
    col = dark + (light - dark) * diff[..., None]
    col = col * 0.85 + base * 0.15
    rim = np.clip(1 - nz, 0, 1)[..., None]               # edges turn deeper
    col = col * (1 - 0.35 * rim) + _f3(shade(c, -0.35)) * 0.35 * rim
    col = col + spec[..., None] * 0.9
    _soft_shadow(L, A, size, 0.3, 0.07, 0.08)
    L.paint(np.clip(col, 0, 1), A)


def _line_t(A):
    """0..1 down each glyph row band (per text line, from the mask rows)."""
    h, w = A.shape
    rows = A.max(1) > 0.3
    t = np.zeros(h, np.float32)
    y = 0
    while y < h:
        if rows[y]:
            y0 = y
            while y < h and rows[y]:
                y += 1
            t[y0:y] = np.linspace(0, 1, y - y0, dtype=np.float32)
        y += 1
    return np.broadcast_to(t[:, None], (h, w))


def _chrome(L, A, size, c, ctx):
    """Liquid chrome: sky / horizon / ground reflection across each line,
    bent by the bevel normals, tinted by the base color."""
    _, nx, ny, nz = _surface(A, size * 0.14, 1.0)
    t = np.clip(_line_t(A) + ny * 0.55, 0, 1)            # bevels bend the horizon
    tint = _f3(shade(c, 0.2))
    sky_top = tint * 0.35 + np.array([0.98, 0.99, 1.0], np.float32) * 0.65
    sky_low = tint * 0.55 + np.array([0.55, 0.62, 0.72], np.float32) * 0.45
    horizon = _f3(shade(c, -0.42)) * 0.6 + 0.4 * np.array([0.08, 0.08, 0.1], np.float32)
    ground = tint * 0.6 + np.array([0.85, 0.8, 0.75], np.float32) * 0.4
    stops = [(0.0, sky_top), (0.5, sky_low), (0.56, horizon), (0.68, ground * 0.85),
             (1.0, ground * 1.1)]
    col = np.zeros(A.shape + (3,), np.float32)
    for k in range(3):
        col[..., k] = np.interp(t, [s_ for s_, _ in stops], [v[k] for _, v in stops])
    _, spec = _lit(nx, ny, nz, 60)
    col = col + spec[..., None] * 0.9
    _soft_shadow(L, A, size, 0.4, 0.05, 0.06)
    L.paint(_f3(shade(c, -0.45)), _dilate(A, max(1, size // 40)))
    L.paint(np.clip(col, 0, 1), A)

def _glass(L, A, size, c, ctx):
    """Tinted frosted glass: the blurred scene through the letters, colored by
    the base color, bright fresnel rim, crisp highlight, soft drop shadow."""
    img = ctx.get("img")
    _, nx, ny, nz = _surface(A, size * 0.12, 1.0)
    if img is not None:
        frosted = np.asarray(Image.fromarray((img * 255).astype(np.uint8))
                             .filter(ImageFilter.GaussianBlur(max(4, size * 0.2))),
                             np.float32) / 255
        off = int(size * 0.06)
        frosted = np.roll(frosted, (off, off), (0, 1))
    else:
        frosted = np.broadcast_to(_f3(ctx.get("bg", "#888888")), A.shape + (3,))
    bg_l = float(np.mean(frosted))
    tint = _f3(shade(c, -0.15 if bg_l > 0.55 else 0.1))
    col = frosted * 0.4 + tint * 0.6                       # tinted glass body
    fres = (np.clip(1 - nz, 0, 1) ** 0.7)[..., None]
    col = col * (1 - fres * 0.55) + fres * 0.55
    _, spec = _lit(nx, ny, nz, 70)
    col = col + spec[..., None] * 0.9
    L.paint(_f3("#000000"), _blur(_shift(A, 0, max(1, int(size * 0.06))), size * 0.09) * 0.35)
    L.paint(_f3(shade(c, -0.35)), _dilate(A, 1) * 0.6)
    L.paint(np.clip(col, 0, 1), A)
    edge = np.clip(A - _erode(A, max(1, size // 45)), 0, 1)
    L.paint(_f3("#FFFFFF"), edge * 0.8)

def _holo(L, A, size, c, ctx):
    """Holographic / iridescent foil: hue flows with the surface angle."""
    _, nx, ny, nz = _surface(A, size * 0.16, 1.0)
    h, w = A.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    hue = (_hls(c)[0] + 0.35 * nx + 0.25 * ny + (xx + yy) / (2.2 * max(h, w))) % 1
    sat, lig = 0.55, 0.72
    q = np.where(lig < 0.5, lig * (1 + sat), lig + sat - lig * sat)
    p_ = 2 * lig - q

    def chan(t):
        t = t % 1
        return np.where(t < 1 / 6, p_ + (q - p_) * 6 * t,
               np.where(t < 1 / 2, q, np.where(t < 2 / 3, p_ + (q - p_) * (2 / 3 - t) * 6, p_)))
    col = np.stack([chan(hue + 1 / 3), chan(hue), chan(hue - 1 / 3)], -1).astype(np.float32)
    diff, spec = _lit(nx, ny, nz, 50)
    col = col * (0.75 + 0.35 * diff[..., None]) + spec[..., None] * 0.7
    _soft_shadow(L, A, size, 0.3, 0.05, 0.07)
    bg_light = _hls(ctx.get("bg", "#000000"))[1] > 0.6
    rim = _f3(shade(c, -0.45)) if bg_light else _f3("#FFFFFF")
    L.paint(rim, _dilate(A, max(1, size // 30 if bg_light else size // 40)) * 0.9)
    L.paint(np.clip(col, 0, 1), A)


def _gradient_grain(L, A, size, c, ctx):
    """Soft diagonal gradient between two NEIGHBOUR hues of the base color
    (same lightness, so it always reads), fine grain, subtle color glow."""
    h_, l_, s_ = _hls(c)
    s_ = max(s_, 0.55)
    ca = _from_hls(h_ - 0.06, l_ + 0.06, s_)
    cb = _from_hls(h_ + 0.09, l_ - 0.02, s_)
    h, w = A.shape
    ys, xs = np.where(A > 0.3)
    x0, x1, y0, y1 = (xs.min(), xs.max(), ys.min(), ys.max()) if len(xs) else (0, w, 0, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.clip((xx - x0) / max(1, x1 - x0) * 0.75 + (yy - y0) / max(1, y1 - y0) * 0.25, 0, 1)
    col = _f3(ca) + (_f3(cb) - _f3(ca)) * t[..., None]
    rng = np.random.default_rng(7)
    col = col + rng.standard_normal((h, w, 1)).astype(np.float32) * 0.022
    L.paint(_f3(c), _blur(_dilate(A, max(1, size // 25)), size * 0.3) * 0.3)
    L.paint(_f3("#000000"), _blur(_shift(A, 0, max(1, size // 20)), size * 0.05) * 0.25)
    L.paint(np.clip(col, 0, 1), A)

def _erode(a, r):
    return _arr(_pil(a).filter(ImageFilter.MinFilter(2 * r + 1))) if r > 0 else a


_FX = {"metal3d": _metal3d, "neon": _neon, "candy": _candy,
       "longshadow": _longshadow, "outline_pop": _outline_pop,
       "inflated": _inflated, "chrome": _chrome, "glass": _glass,
       "holo": _holo, "gradient": _gradient_grain}
_CTX_FX = {"inflated", "chrome", "glass", "holo", "gradient"}


# ---- public API ----------------------------------------------------------------
def glyph_alpha(reg: FontRegistry, layout: Dict, block: Dict, w: int, h: int):
    b = copy.deepcopy(block)
    b.update(color="#FFFFFF", shadow=False, outline_color="#FFFFFF", outline_width=0)
    im = draw_block(Image.new("RGB", (w, h)), reg, b, layout.get("language", "latin"))
    return _arr(im.convert("L"))


def choose_style(img: Image.Image, layout: Dict, mood: str, intents: List[str],
                 seed: int, effect: str = "auto") -> Dict:
    """{"preset", "main", "accent"} for this design (deterministic per seed)."""
    rng = random.Random((int(seed) * 2654435761 + 17) & 0xFFFFFFFF)
    luma = float(np.asarray(img.convert("L"), np.float32).mean()) / 255
    if effect in PRESETS:
        preset = effect
    else:
        pool = [p for p in BY_MOOD.get(mood, list(PRESETS))
                if not (p in DARK_ONLY and luma > NEON_MAX_LUMA)]
        preset = rng.choice(pool)
    head = next((b for b in layout["blocks"] if b["role"] == "headline"),
                layout["blocks"][0])
    head_bg = region_color(img, head["box"])
    salient = salient_colors(img)
    # text colors = the colors that stand out in the picture; the effect's
    # outline / extrude / glow carries the contrast, so the face keeps the hue
    readable = [c for c in salient if hue_dist(c, head_bg) >= 0.06
                or abs(_hls(c)[1] - _hls(head_bg)[1]) >= 0.25]
    pool = readable or intents or NEUTRALS
    main = pool[0] if len(pool) == 1 else rng.choice(pool[:2])
    rest = [c for c in salient + intents if c != main and hue_dist(c, main) >= 0.06]
    accent = rest[0] if rest else shade(main, 0.25 if _hls(main)[1] < 0.5 else -0.25)
    return {"preset": preset, "main": main, "accent": accent}


def style_layer(reg: FontRegistry, layout: Dict, img: Image.Image, style: Dict,
                min_face_contrast: float = 3.0) -> Image.Image:
    """RGBA text layer (canvas size): effect roles get the preset, the other
    lines keep their fitted color with a soft legibility shadow."""
    w, h = layout["canvas"]["w"], layout["canvas"]["h"]
    img = img.convert("RGB").resize((w, h))
    img_arr = np.asarray(img, np.float32) / 255
    L = Layer(h, w)
    for b in layout["blocks"]:
        A = glyph_alpha(reg, layout, b, w, h)
        bgc = region_color(img, b["box"])
        if b["role"] in EFFECT_ROLES:
            base = style["main"] if b["role"] == "headline" else style["accent"]
            if style["preset"] in NO_OUTLINE:     # no outline: the face must read
                base = rgb_to_hex(fit_contrast(hex_to_rgb(base), hex_to_rgb(bgc),
                                               max(min_face_contrast, 4.5)))
            b["effect"] = {"preset": style["preset"], "color": base}
            fn = _FX[style["preset"]]
            if style["preset"] in _CTX_FX:
                fn(L, A, int(b["size_px"]), base,
                   {"bg": bgc, "img": img_arr, "accent": style["accent"]
                    if b["role"] == "headline" else style["main"]})
            else:
                fn(L, A, int(b["size_px"]), base, bgc)
        else:
            light = contrast_ratio(hex_to_rgb(b["color"]), (0, 0, 0)) > 7
            L.paint(_f3("#000000" if light else "#FFFFFF"), _blur(_dilate(A, 2), 3) * 0.55)
            L.paint(_f3(b["color"]), A)
    return L.image()


def render_with_effects(reg: FontRegistry, layout: Dict, img: Image.Image,
                        mood: str, intents: List[str], seed: int,
                        effect: str = "auto") -> Tuple[Image.Image, Optional[Dict]]:
    """Composite = image + effect text layer. effect "none" -> (None, None)."""
    if effect == "none" or not layout.get("blocks"):
        return None, None
    w, h = layout["canvas"]["w"], layout["canvas"]["h"]
    base = img.convert("RGB").resize((w, h))
    style = choose_style(base, layout, mood, intents, seed, effect)
    layer = style_layer(reg, layout, base, style)
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB"), style
