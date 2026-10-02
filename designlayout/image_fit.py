# -*- coding: utf-8 -*-
"""Image-first layout: pick the template AFTER the background exists.

The designer's order: look at the picture, put the text where it is calm.
1. busy map of the real background (edges, spread to a neighbourhood) plus
   luminance statistics, as integral images for O(1) box means;
2. every template that suits the design type + canvas aspect is laid out
   (cheap, CPU) and scored by how busy the scene is under each TEXT LINE
   (not the whole band), weighted by role importance;
3. among the near-best candidates the seed picks one (variety, but always a
   calm placement);
4. a lens-like defocus + gentle tone shift is applied behind each line,
   stronger where the scene is busier -- no flat plates, no gray scrims.
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import measure, templates, compositions
from .fonts import FontRegistry
from .colors import hex_to_rgb, rel_luminance
from .render_preview import _line_positions

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

ROLE_WEIGHT = {"headline": 1.0, "subheadline": 0.9, "emphasis": 0.9,
               "body": 0.8, "detail": 0.7, "note": 0.5}
NEAR_BEST = 1.15          # candidates within +15% of the best cost are "tied"
TOP_K = 5                 # ... of which the seed picks one
LINE_PAD_EM = 0.45        # calm area around a line, x font size


# ---- image statistics --------------------------------------------------------
OBJECT_FREE = 0.20        # share of a line that may cover objects for free
OBJECT_WEIGHT = 1.6       # text over an object costs more than over a busy wall
BACKDROP_CHAIN_DE = 14.0  # shades closer than this to the backdrop join it
OBJECT_DE = 22.0          # Lab distance from the backdrop that is "an object"


def _lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], 1)


def object_map(rgb: Image.Image, cw: int, ch: int, k: int = 16) -> np.ndarray:
    """0..1 per pixel: how much it is an OBJECT rather than the backdrop.

    Edges alone miss big smooth objects (a balloon is flat inside). The
    backdrop is the most common color plus every shade chained to it (so a
    sky gradient stays backdrop); anything clearly off those colors -- a
    balloon, a crown, a stage -- is an object even where it is smooth."""
    small = rgb.convert("RGB").resize((160, max(1, int(160 * ch / cw))))
    q = small.quantize(k, method=Image.Quantize.MEDIANCUT)
    idx = np.asarray(q).reshape(-1)
    px = np.asarray(small, np.float32).reshape(-1, 3)
    kk = int(idx.max()) + 1
    share = np.bincount(idx, minlength=kk) / len(idx)
    cent = np.stack([px[idx == i].mean(0) if (idx == i).any() else np.zeros(3)
                     for i in range(kk)])
    lab = _lab(cent)
    back = {int(share.argmax())}
    grown = True
    while grown:
        grown = False
        for i in range(kk):
            if i in back or share[i] < 0.01:
                continue
            if min(np.linalg.norm(lab[i] - lab[j]) for j in back) < BACKDROP_CHAIN_DE:
                back.add(i)
                grown = True
    d = np.min(np.linalg.norm(_lab(px)[:, None, :] - lab[sorted(back)][None], axis=2),
               axis=1)
    obj = np.clip((d - OBJECT_DE * 0.5) / OBJECT_DE, 0, 1)
    im = Image.fromarray((obj.reshape(small.size[1], small.size[0]) * 255)
                         .astype(np.uint8)).resize((cw, ch), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255


class ImageStats:
    """Busy map + luminance integrals of the background at canvas size."""

    def __init__(self, img: Image.Image, cw: int, ch: int) -> None:
        self.cw, self.ch = cw, ch
        rgb = img.convert("RGB").resize((cw, ch), Image.LANCZOS)
        gray = np.asarray(rgb.convert("L"), dtype=np.float32)
        if _HAS_CV2:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        else:
            gx = np.zeros_like(gray)
            gy = np.zeros_like(gray)
            gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
            gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        busy = np.clip(np.hypot(gx, gy) / 64.0, 0, 1)
        # a line is hard to read if detail is NEAR it, not only under it
        r = max(3, int(0.012 * min(cw, ch)))
        if _HAS_CV2:
            busy = cv2.blur(busy, (2 * r + 1, 2 * r + 1))
        else:
            k = Image.fromarray((busy * 255).astype(np.uint8))
            busy = np.asarray(k.filter(ImageFilter.BoxBlur(r)),
                              dtype=np.float32) / 255.0
        self.busy = busy
        self._ib = self._integral(busy)
        self.objects = object_map(rgb, cw, ch)
        self._io = self._integral(self.objects)
        lum = gray / 255.0
        self._il = self._integral(lum)
        self._il2 = self._integral(lum * lum)
        self.image = rgb

    @staticmethod
    def _integral(a: np.ndarray) -> np.ndarray:
        return np.pad(a.astype(np.float64).cumsum(0).cumsum(1),
                      ((1, 0), (1, 0)))

    def _clip(self, box) -> Optional[Tuple[int, int, int, int]]:
        x0 = max(0, int(box[0])); y0 = max(0, int(box[1]))
        x1 = min(self.cw, int(box[2])); y1 = min(self.ch, int(box[3]))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _mean(self, I, b) -> float:
        x0, y0, x1, y1 = b
        s = I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]
        return float(s) / ((x1 - x0) * (y1 - y0))

    def box_cost(self, box) -> Tuple[float, float, float]:
        """(cost, busy, luminance std) of a box;
        cost = busy + 0.8 x std + OBJECT_WEIGHT x object share.

        std catches a line crossing a light/dark boundary (one text color
        cannot work on both halves) even when there are few edges.
        """
        b = self._clip(box)
        if b is None:
            return 1.0, 1.0, 0.5
        busy = self._mean(self._ib, b)
        m = self._mean(self._il, b)
        var = max(0.0, self._mean(self._il2, b) - m * m)
        std = math.sqrt(var)
        obj = self._mean(self._io, b)
        # a moderate overlap is fine (text may sit on part of the scene); the
        # cost only grows once a line covers more than OBJECT_FREE of objects
        over = max(0.0, obj - OBJECT_FREE) / (1 - OBJECT_FREE)
        return busy + 0.8 * std + OBJECT_WEIGHT * over, busy, std


# ---- scoring -----------------------------------------------------------------
def line_boxes(reg: FontRegistry, layout: Dict,
               pad_em: float = LINE_PAD_EM) -> List[Tuple[list, float, Dict]]:
    """(box [x0,y0,x1,y1], role weight, block) for every text line."""
    out = []
    lang = layout.get("language", "latin")
    for b in layout.get("blocks", []):
        f = measure.get_font(reg, b["font"], b["size_px"])
        pad = pad_em * b["size_px"]
        for ln, x, y in _line_positions(reg, b, lang):
            bb = f.getbbox(ln, anchor="la")
            out.append(([x + bb[0] - pad, y + bb[1] - pad,
                         x + bb[2] + pad, y + bb[3] + pad],
                        ROLE_WEIGHT.get(b["role"], 0.6), b))
    return out


def line_cost(stats: ImageStats, box) -> float:
    """50% mean + 50% WORST segment along the line.

    A line that is mostly over a calm wall but whose end runs into an object
    is hard to read there; a plain mean would dilute that into "calm"."""
    x0, y0, x1, y1 = box
    n = int(min(8, max(3, round((x1 - x0) / max(1.0, y1 - y0)))))
    step = (x1 - x0) / n
    segs = [stats.box_cost([x0 + i * step, y0, x0 + (i + 1) * step, y1])[0]
            for i in range(n)]
    return 0.5 * (sum(segs) / n) + 0.5 * max(segs)


def score_layout(reg: FontRegistry, layout: Dict, stats: ImageStats,
                 warnings: List[str]) -> float:
    total = wsum = 0.0
    for box, w, b in line_boxes(reg, layout):
        # always judged on the RAW scene: a badge / ribbon over a busy spot
        # would hide the subject, so it is not "free" either
        cost = line_cost(stats, box)
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        ww = w * math.sqrt(area)
        total += cost * ww
        wsum += ww
    base = total / wsum if wsum else 1.0
    dropped = sum(1 for m in warnings if m.startswith("dropped"))
    return base + 0.5 * dropped


def candidate_pool(spec: Dict, aspect: str) -> List[Dict]:
    texts = [t for t in spec.get("texts", []) if str(t.get("text", "")).strip()]
    has_emph = any(t.get("role") == "emphasis" for t in texts)
    pool = templates.pool(spec.get("design_type", "poster"), aspect)
    if not has_emph:  # a badge template without an emphasis wastes its slot
        pool = [t for t in pool if not templates.is_badge(t)] or pool
    # natural look: no plates / strips behind the text (they would always
    # "win" readability and cover the picture); forcing one still works
    pool = [t for t in pool if not t.get("panel")] or pool
    return pool


# ---- empty areas ---------------------------------------------------------------
FREE_MIN_W, FREE_MIN_H = 0.42, 0.22   # smallest area worth fitting text into
FREE_CLEAN = 0.90                     # mean emptiness required inside it
FREE_SAFE = 0.03                      # objects grown by this (x short side)
FREE_RECTS = 3
HEADLINE_SIZE_WEIGHT = 0.35          # cost of a headline at 0 vs the biggest size


def empty_rects(stats: "ImageStats", n: int = FREE_RECTS) -> List[Tuple[float, float, float, float]]:
    """Largest clearly empty rectangles (canvas fractions x, y, w, h).

    Empty = no object (backdrop color) and little detail. Searched on a coarse
    grid with integral images; returns up to `n` that do not overlap much."""
    g = 24
    cw, ch = stats.cw, stats.ch
    occ = np.clip(stats.objects + 1.5 * stats.busy, 0, 1)
    # keep a safety gap around every object, so text never grazes its edge
    r = max(1, int(FREE_SAFE * min(cw, ch)))
    occ_im = Image.fromarray((occ * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(2 * (r // 2) + 1))
    small = np.asarray(occ_im.resize((g, g), Image.BOX), np.float32) / 255
    small = np.maximum(small, (small > 0.35) * 1.0)   # any real object = full
    free = 1 - small
    I = np.pad(free.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    cands = []
    mw, mh = int(round(FREE_MIN_W * g)), int(round(FREE_MIN_H * g))
    for y0 in range(0, g - mh + 1):
        for y1 in range(y0 + mh, g + 1):
            for x0 in range(0, g - mw + 1):
                for x1 in range(x0 + mw, g + 1):
                    a = (x1 - x0) * (y1 - y0)
                    m = (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / a
                    if m >= FREE_CLEAN:
                        cands.append((a * m, x0, y0, x1, y1))
    cands.sort(reverse=True)
    out: List[Tuple[int, int, int, int]] = []
    for _, x0, y0, x1, y1 in cands:
        area = (x1 - x0) * (y1 - y0)
        if all(max(0, min(x1, b[2]) - max(x0, b[0])) *
               max(0, min(y1, b[3]) - max(y0, b[1])) < 0.5 * area for b in out):
            out.append((x0, y0, x1, y1))
        if len(out) >= n:
            break
    return [(x0 / g, y0 / g, (x1 - x0) / g, (y1 - y0) / g) for x0, y0, x1, y1 in out]


def _fitted_variants(pool: List[Dict], rects) -> List[Dict]:
    """Every plain template once more per empty area (frames / badges stay
    canvas-wide, so only templates without them are squeezed)."""
    out = []
    for t in pool:
        if templates.is_badge(t) or any(d["type"] in ("border", "corners")
                                        for d in t["decor"]):
            continue
        for i, r in enumerate(rects):
            v = dict(t, id=f"{t['id']}@free{i}", content_rect=r)
            out.append(v)
    return out


# Flexible layouts: built FROM the empty area instead of a fixed template --
# (name, primary box, primary valign, secondary box, secondary valign) in
# fractions of the empty area. spread = headline at the top of the area and
# the details at its bottom; center = the block in the middle; tall =
# a big headline taking most of the area.
FLEX_KINDS = [
    ("spread", (0, 0, 1, .58), "top", (0, .60, 1, .40), "bottom"),
    ("center", (0, .08, 1, .52), "bottom", (0, .62, 1, .32), "top"),
    ("tall", (0, 0, 1, .68), "middle", (0, .71, 1, .29), "top"),
]
FLEX_TYPE_SCALE = 1.3     # start bigger; the fit shrinks to the area
FILL_WEIGHT = 0.30        # cost of a text block that fills none of the area


def _flex_templates(spec: Dict, rects) -> List[Dict]:
    out = []
    dtype = spec.get("design_type", "poster")
    for i, r in enumerate(rects):
        r = _widen(r)
        x, _, w, _ = r
        # an area hugging one side reads best aligned to that side
        align = "left" if x < 0.06 and w < 0.7 else \
                "right" if x + w > 0.94 and w < 0.7 else "center"
        for kind, pbox, pv, sbox, sv in FLEX_KINDS:
            out.append({
                "id": f"flex.{kind}.{align}@free{i}", "family": "flex",
                "design_types": {dtype}, "aspects": set(templates.GENERAL),
                "primary": {"box": pbox, "align": align, "valign": pv},
                "secondary": {"box": sbox, "align": align, "valign": sv},
                "badge": None, "badge_shape": "circle", "decor": [],
                "panel": None, "ribbon": False, "outline": 0.0,
                "type_scale": FLEX_TYPE_SCALE, "inset": 0.0, "weight": 1.0,
                "content_rect": r,
            })
    return out


FLEX_MIN_W = 0.66         # a text area narrower than this is widened around
                          # its center (a moderate overlap is allowed; the
                          # object cost still judges it)


def _widen(r, min_w: float = FLEX_MIN_W):
    x, y, w, h = r
    if w >= min_w:
        return r
    cx = x + w / 2
    x0 = min(max(0.0, cx - min_w / 2), 1.0 - min_w)
    return (x0, y, min_w, h)


def _two_zone_templates(spec: Dict, rects) -> List[Dict]:
    """Empty space is often L / T shaped (a wide band + a column between two
    objects): headline + emphasis go to the upper area, the details to the
    part of the lower area below it -- one layout over two empty areas."""
    out = []
    dtype = spec.get("design_type", "poster")
    for ia, a in enumerate(rects):
        for ib, b in enumerate(rects):
            if ia == ib:
                continue
            ay1 = a[1] + a[3]
            by0 = max(b[1], ay1)
            bh = b[1] + b[3] - by0
            if a[1] + a[3] / 2 >= b[1] + b[3] / 2 or bh < 0.14 or a[3] < 0.12:
                continue
            for kind, sv in (("spread", "bottom"), ("near", "top")):
                out.append({
                    "id": f"flex2.{kind}@free{ia}.{ib}", "family": "flex",
                    "design_types": {dtype}, "aspects": set(templates.GENERAL),
                    "primary": {"box": a, "align": "center", "valign": "middle"},
                    "secondary": {"box": _widen((b[0], by0, b[2], bh)),
                                  "align": "center", "valign": sv},
                    "badge": None, "badge_shape": "circle", "decor": [],
                    "panel": None, "ribbon": False, "outline": 0.0,
                    "type_scale": FLEX_TYPE_SCALE, "inset": 0.0, "weight": 1.0,
                    # boxes are canvas fractions: the content area is the canvas
                    "content_rect": (0.0, 0.0, 1.0, 1.0),
                })
    return out


def text_fill(reg: FontRegistry, layout: Dict, rect, cw: int, ch: int) -> float:
    """0..1: how much of the empty area's height x width the text block spans
    (the union of all lines), so a block crammed in one corner scores low."""
    boxes = [b for b, _, _ in line_boxes(reg, layout, pad_em=0)]
    if not boxes or not rect:
        return 0.0
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    rw, rh = rect[2] * cw, rect[3] * ch
    return min(1.0, (y1 - y0) / max(1.0, rh)) * 0.6 + \
        min(1.0, (x1 - x0) / max(1.0, rw)) * 0.4


def fit_layout(reg: FontRegistry, spec: Dict, background: Image.Image,
               canvas: Dict, seed: int = 0, candidates: int = 0,
               force: str = "") -> Tuple[Dict, List[str], List[str]]:
    """Return (layout, report lines, warnings) with the calmest placement."""
    from .layout import LayoutEngine
    cw, ch = canvas["w"], canvas["h"]
    if force:
        eng = LayoutEngine(reg, canvas)
        lay = eng.layout(spec, seed=seed, force_composition=force)
        return lay, [f"forced template {lay['template']}"], eng.warnings
    stats = ImageStats(background, cw, ch)
    aspect = compositions.aspect_class(cw, ch)
    pool = candidate_pool(spec, aspect)
    rng = random.Random(seed)
    if candidates and len(pool) > candidates:
        pool = rng.sample(pool, candidates)
    fam = templates.FAMILY_ALIASES.get(spec.get("composition_hint", "auto"),
                                       spec.get("composition_hint", "auto"))
    rects = empty_rects(stats)
    variants = (_fitted_variants(pool, rects) + _flex_templates(spec, rects)
                + _two_zone_templates(spec, rects))
    for v in variants:                      # visible to compositions.choose
        templates.BY_ID[v["id"]] = v
    scored = []
    try:
        for t in pool + variants:
            eng = LayoutEngine(reg, canvas)
            lay = eng.layout(spec, seed=seed, force_composition=t["id"])
            cost = score_layout(reg, lay, stats, eng.warnings)
            if t["family"] == fam:
                cost *= 0.92      # the LLM's layout family, a soft preference
            scored.append((cost, t["id"], lay, list(eng.warnings)))
    finally:
        for v in variants:
            templates.BY_ID.pop(v["id"], None)
    # a calm spot is worth little if the headline ends up tiny: cost grows as
    # the headline shrinks below the biggest one any candidate achieved
    def head_px(lay):
        return max([b["size_px"] for b in lay["blocks"] if b["role"] == "headline"]
                   or [0])
    top = max([head_px(l) for _, _, l, _ in scored] or [1]) or 1
    big = None
    if rects:
        bx0 = min(r[0] for r in rects); by0 = min(r[1] for r in rects)
        bx1 = max(r[0] + r[2] for r in rects); by1 = max(r[1] + r[3] for r in rects)
        big = (bx0, by0, bx1 - bx0, by1 - by0)
    scored = [(c + HEADLINE_SIZE_WEIGHT * (1 - head_px(l) / top)
               + (FILL_WEIGHT * (1 - text_fill(reg, l, big, cw, ch)) if big else 0),
               i, l, w) for c, i, l, w in scored]
    scored.sort(key=lambda s: s[0])
    best = scored[0][0]
    near = [s for s in scored if s[0] <= best * NEAR_BEST + 1e-9][:TOP_K]
    cost, tid, lay, warns = rng.choice(near)
    report = [f"image-fit: {len(scored)} layouts scored ({len(variants)} fitted "
              f"into {len(rects)} empty areas), picked {tid} "
              f"(cost {cost:.3f}, best {best:.3f}, {len(near)} near-best)"]
    report += [f"  empty area x={r[0]:.2f} y={r[1]:.2f} w={r[2]:.2f} h={r[3]:.2f}"
               for r in rects]
    report += [f"  {c:.3f} {i}" for c, i, _, _ in scored[:TOP_K]]
    return lay, report, warns


# ---- treatment behind the text -----------------------------------------------
def defocus_behind_text(reg: FontRegistry, layout: Dict,
                        background: Image.Image, strength: float = 1.0,
                        report: Optional[List[str]] = None) -> Image.Image:
    """Lens-like blur + tone push away from the text color, per line,
    scaled by how busy the scene is there. Returns a canvas-size image."""
    cw, ch = layout["canvas"]["w"], layout["canvas"]["h"]
    img = background.convert("RGB").resize((cw, ch), Image.LANCZOS)
    if strength <= 0:
        return img
    stats = ImageStats(img, cw, ch)
    short = min(cw, ch)
    for box, _, b in line_boxes(reg, layout):
        if b.get("on_shape") or b.get("on_panel"):
            continue
        _, busy, std = stats.box_cost(box)
        level = min(1.0, busy + 0.5 * std)
        radius = (0.009 + 0.024 * level) * short * strength
        tone = min(0.5, (0.08 + 0.30 * level) * strength)
        feather = max(8, int(0.02 * short))
        m = Image.new("L", (cw, ch), 0)
        pad = (box[3] - box[1]) * 0.25
        ImageDraw.Draw(m).rounded_rectangle(box, radius=pad, fill=255)
        m = m.filter(ImageFilter.GaussianBlur(feather))
        blur = img.filter(ImageFilter.GaussianBlur(radius))
        a = np.asarray(blur, dtype=np.float32)
        text_light = rel_luminance(hex_to_rgb(b.get("color", "#FFFFFF"))) > 0.5
        target = 0.0 if text_light else 255.0
        a = a * (1 - tone) + target * tone
        treated = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
        img = Image.composite(treated, img, m)
        if report is not None:
            report.append(f"  defocus {b['role']:11} busy={busy:.2f} "
                          f"std={std:.2f} blur={radius:.0f}px tone={tone:.2f}")
    return img
