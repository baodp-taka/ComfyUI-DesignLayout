# -*- coding: utf-8 -*-
"""Generated picture + foreground alpha -> transparent PNG (logo / sticker).

The AI draws on a plain background; BiRefNet decides what the subject is (by
meaning, not by colour), so a white part inside the logo is kept.

- Edge clean-up: soft edge pixels are a mix of subject and background. When
  the picture's border is one plain colour, that colour is taken out of them:
  F = (C - (1 - a) * bg) / a -- no white / coloured halo on a dark page.
  A busy border (a scene, a gradient) skips this step: cut only.
- Sticker: a white outline drawn here, after the cut (like Canva), so it is
  always even. Holes narrower than the outline are filled white, wide holes
  stay transparent. The canvas grows only when the outline would hit an edge.

numpy + scipy only, so it can be tested offline.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy import ndimage

_PLAIN = 0.08     # border pixel this close to the border colour = background
_PLAIN_SHARE = 0.8  # this share of the border plain -> known background
_CLEAR = 0.02     # alpha below = transparent, above 1 - _CLEAR = solid
_SOFT = 1.5       # outline's anti-aliased outer edge (px)


def border_colour(rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Median colour of the picture's 4 edges, and whether they are plain."""
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(edge, axis=0)
    near = np.linalg.norm(edge - bg, axis=1) / np.sqrt(3) < _PLAIN
    return bg, bool(near.mean() >= _PLAIN_SHARE)


def clean_edges(rgb: np.ndarray, alpha: np.ndarray, bg: np.ndarray
                ) -> np.ndarray:
    """Take the background colour out of the soft edge pixels."""
    out = rgb.copy()
    band = (alpha > _CLEAR) & (alpha < 1 - _CLEAR)
    a = alpha[band, None]
    out[band] = np.clip((rgb[band] - (1 - a) * bg) / a, 0, 1)
    return out


def add_outline(rgb: np.ndarray, alpha: np.ndarray, width: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """White outline `width` px around the subject (subject drawn on top)."""
    r = max(width, 0.0)
    ys, xs = np.nonzero(alpha >= 0.5)
    if r <= 0 or len(ys) == 0:
        return rgb, alpha
    h, w = alpha.shape
    room = min(ys.min(), xs.min(), h - 1 - ys.max(), w - 1 - xs.max())
    pad = int(np.ceil(max(0.0, r + _SOFT + 1 - room)))
    if pad:
        rgb = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)))
        alpha = np.pad(alpha, pad)
    dist = ndimage.distance_transform_edt(alpha < 0.5)   # px to the subject
    white = np.clip((r + _SOFT - dist) / _SOFT, 0, 1)    # 1 under the subject
    out_a = alpha + white * (1 - alpha)                   # subject over white
    mix = rgb * alpha[..., None] + white[..., None] * (1 - alpha[..., None])
    out = np.where(out_a[..., None] > 0,
                   mix / np.maximum(out_a[..., None], 1e-6), 0.0)
    return np.clip(out, 0, 1), out_a


def transparent(rgb: np.ndarray, alpha: np.ndarray, outline: float = 0.0
                ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """rgb float [H, W, 3] 0..1, alpha float [H, W] 0..1 (BiRefNet).

    outline: white outline width, fraction of the picture's long side
    (0 = logo, no outline). Returns (rgb, alpha, plain_background).
    """
    a = alpha.astype(np.float32).copy()
    a[a < _CLEAR] = 0.0
    a[a > 1 - _CLEAR] = 1.0
    bg, plain = border_colour(rgb)
    out = clean_edges(rgb, a, bg) if plain else rgb.copy()
    out, a = add_outline(out, a, outline * max(a.shape))
    return out.astype(np.float32), a.astype(np.float32), plain
