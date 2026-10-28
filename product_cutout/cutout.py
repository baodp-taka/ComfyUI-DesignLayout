# -*- coding: utf-8 -*-
"""Product photo + foreground alpha -> clean reference for Qwen-Image-Edit 2511.

The product is put on white and cropped tight (with a margin), so it fills the
reference picture: 2511 then sees labels and logos clearly even when the photo
was taken from afar or on a cluttered table. The AI still repaints the whole
picture, so light and shadows stay natural.

This step only removes the old background and zooms in; it does not decide
which object is the product. Several separate objects (a bottle next to a
stack of towels): all are kept, except objects cut by the photo's frame
(a table edge, a leaf in the corner) when something else is left. Returning
the whole photo instead let 2511 keep the old scene almost unchanged.

numpy + scipy only, so it can be tested offline.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy import ndimage

DEFAULT_PADDING = 0.08   # margin, fraction of the product's long side
DEFAULT_MIN_PART = 0.15  # smaller than this (vs the largest) = a speck
_THRESHOLD = 0.5         # alpha above this = product
_EDGE = 3                # keep the soft edge around the product (px)
_FRAME = 3               # an object within this of the border is cut by it


def _touches_frame(region: np.ndarray) -> bool:
    return bool(region[:_FRAME].any() or region[-_FRAME:].any()
                or region[:, :_FRAME].any() or region[:, -_FRAME:].any())


def cutout(rgb: np.ndarray, alpha: np.ndarray,
           padding: float = DEFAULT_PADDING,
           min_part: float = DEFAULT_MIN_PART
           ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """rgb float [H, W, 3] 0..1, alpha float [H, W] 0..1.

    Returns (image, mask, cropped). cropped=False -> original photo and a
    full mask (nothing found).
    """
    h, w = alpha.shape
    keep_all = (rgb, np.ones((h, w), np.float32), False)
    labels, n = ndimage.label(alpha > _THRESHOLD)
    if n == 0:
        return keep_all
    sizes = ndimage.sum(np.ones_like(alpha), labels, range(1, n + 1))
    big = [i for i in range(n) if sizes[i] >= sizes.max() * min_part]
    # objects cut by the frame are background, unless nothing else is left
    inner = [i for i in big if not _touches_frame(labels == i + 1)] or big

    # those objects only (drops specks), soft edge kept
    main = np.isin(labels, [i + 1 for i in inner])
    main = ndimage.binary_dilation(main, iterations=_EDGE)
    a = (alpha * main).astype(np.float32)

    ys, xs = np.nonzero(a > _THRESHOLD)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = int(round(max(y1 - y0, x1 - x0) * padding))
    # white canvas: the margin may go past the photo's edges
    ch, cw = (y1 - y0) + 2 * pad, (x1 - x0) + 2 * pad
    out = np.ones((ch, cw, 3), np.float32)
    mask = np.zeros((ch, cw), np.float32)
    piece = a[y0:y1, x0:x1, None]
    out[pad:pad + y1 - y0, pad:pad + x1 - x0] = \
        rgb[y0:y1, x0:x1] * piece + (1.0 - piece)
    mask[pad:pad + y1 - y0, pad:pad + x1 - x0] = a[y0:y1, x0:x1]
    return out, mask, True
