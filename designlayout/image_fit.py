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
        """(cost, busy, luminance std) of a box; cost = busy + 0.8 x std.

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
        return busy + 0.8 * std, busy, std


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
    scored = []
    for t in pool:
        eng = LayoutEngine(reg, canvas)
        lay = eng.layout(spec, seed=seed, force_composition=t["id"])
        cost = score_layout(reg, lay, stats, eng.warnings)
        if t["family"] == fam:
            cost *= 0.92          # the LLM's layout family, a soft preference
        scored.append((cost, t["id"], lay, list(eng.warnings)))
    scored.sort(key=lambda s: s[0])
    best = scored[0][0]
    near = [s for s in scored if s[0] <= best * NEAR_BEST + 1e-9][:TOP_K]
    cost, tid, lay, warns = rng.choice(near)
    report = [f"image-fit: {len(scored)} templates scored, picked {tid} "
              f"(cost {cost:.3f}, best {best:.3f}, {len(near)} near-best)"]
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
