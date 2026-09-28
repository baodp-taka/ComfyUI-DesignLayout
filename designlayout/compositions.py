# -*- coding: utf-8 -*-
"""Named composition templates, chosen by canvas aspect ratio.

A composition maps the design into two stacking BANDS (primary = headline /
subheadline / emphasis, secondary = body / detail / note) plus options
(alignment, emphasis-as-badge, border, type scale). The layout engine measures
real text and stacks it inside the bands, so compositions stay
resolution-independent and purely geometric.

The canvas is classified into an ASPECT class (tall 9:16, portrait 4:5,
square, landscape 16:9, wide banner up to 4:1). Each class has its own pool of
templates and auto-pick weights; a template requested for an aspect it does not
suit is mapped to its closest equivalent (see ALIASES).

Each builder is deterministic given the RNG, so the same seed reproduces the
same design. Builders read a seeded `random.Random` for their variant knobs.
"""
from __future__ import annotations
import random
from typing import Callable, Dict, List, Optional

from .schema import SAFE_MARGIN, DEFAULT_CANVAS

Box = Dict[str, int]

ASPECTS = ["tall", "portrait", "square", "landscape", "wide"]

# supported canvas ratio range (w/h); leaderboard-style strips such as 728x90
# are out of scope
MAX_ASPECT = 4.0


def check_supported(cw: int, ch: int) -> None:
    ar = cw / float(ch)
    if ar > MAX_ASPECT or ar < 1.0 / MAX_ASPECT:
        raise ValueError(
            f"canvas {cw}x{ch} (ratio {ar:.2f}) is not supported; the ratio "
            f"must be between 1:{MAX_ASPECT:g} and {MAX_ASPECT:g}:1")


def aspect_class(cw: int, ch: int) -> str:
    """tall < 0.65 <= portrait < 0.9 <= square <= 1.12 < landscape < 2.1 <= wide
    (<= MAX_ASPECT)."""
    ar = cw / float(ch)
    if ar < 0.65:
        return "tall"
    if ar < 0.9:
        return "portrait"
    if ar <= 1.12:
        return "square"
    if ar < 2.1:
        return "landscape"
    return "wide"


def _box(x: int, y: int, w: int, h: int) -> Box:
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _content(cw: int, ch: int) -> Box:
    # SAFE_MARGIN is defined on the 1080 canvas; scale it with the short side,
    # with a minimum horizontal margin for very wide banners.
    s = min(cw, ch) / float(DEFAULT_CANVAS["h"])
    my = round(SAFE_MARGIN * s)
    mx = max(my, round(0.03 * cw))
    return _box(mx, my, cw - 2 * mx, ch - 2 * my)


def _gap(cw: int, ch: int, px_at_1080: int) -> int:
    return round(px_at_1080 * min(cw, ch) / float(DEFAULT_CANVAS["h"]))


PRIMARY_ROLES = ["headline", "subheadline", "emphasis"]
SECONDARY_ROLES = ["body", "detail", "note"]


def _base(name: str, align: str, primary: Box, secondary: Box,
          **opts) -> Dict:
    d = {
        "name": name,
        "align": align,
        "primary": {"box": primary, "valign": opts.get("p_valign", "middle"),
                    "align": opts.get("p_align", align),
                    "roles": opts.get("primary_roles", PRIMARY_ROLES)},
        "secondary": {"box": secondary,
                      "valign": opts.get("s_valign", "middle"),
                      "align": opts.get("s_align", align),
                      "roles": opts.get("secondary_roles", SECONDARY_ROLES)},
        "emphasis_mode": opts.get("emphasis_mode", "inline"),
        "border": opts.get("border", False),
        "variant": opts.get("variant", ""),
        "type_scale": opts.get("type_scale", 1.0),
    }
    return d


# ---- general builders (tall / portrait / square / landscape) ---------------
def top_headline(cw: int, ch: int, rng: random.Random) -> Dict:
    c = _content(cw, ch)
    align = rng.choice(["center", "center", "left"])
    ph = int(c["h"] * rng.choice([0.34, 0.40]))
    g = _gap(cw, ch, 30)
    primary = _box(c["x"], c["y"], c["w"], ph)
    secondary = _box(c["x"], c["y"] + ph + g, c["w"], c["h"] - ph - g)
    return _base("top_headline", align, primary, secondary,
                 p_valign="middle", s_valign="top", variant=align)


def bottom_band(cw: int, ch: int, rng: random.Random) -> Dict:
    c = _content(cw, ch)
    align = rng.choice(["center", "left"])
    ph = int(c["h"] * 0.34)
    sh = int(c["h"] * 0.14)
    primary = _box(c["x"], c["y"] + c["h"] - ph, c["w"], ph)
    secondary = _box(c["x"], c["y"] + c["h"] - ph - _gap(cw, ch, 24) - sh,
                     c["w"], sh)
    return _base("bottom_band", align, primary, secondary,
                 p_valign="middle", s_valign="bottom", variant=align)


def center_stack(cw: int, ch: int, rng: random.Random) -> Dict:
    c = _content(cw, ch)
    ph = int(c["h"] * 0.44)
    top = c["y"] + int(c["h"] * 0.20)
    primary = _box(c["x"], top, c["w"], ph)
    secondary = _box(c["x"], top + ph + _gap(cw, ch, 20), c["w"],
                     int(c["h"] * 0.18))
    return _base("center_stack", "center", primary, secondary,
                 p_valign="middle", s_valign="top")


def left_column(cw: int, ch: int, rng: random.Random) -> Dict:
    c = _content(cw, ch)
    fracs = [0.48, 0.55] if cw > ch * 1.12 else [0.56, 0.62]
    colw = int(c["w"] * rng.choice(fracs))
    ph = int(c["h"] * 0.45)
    top = c["y"] + int(c["h"] * 0.10)
    primary = _box(c["x"], top, colw, ph)
    secondary = _box(c["x"], top + ph + _gap(cw, ch, 20), colw,
                     int(c["h"] * 0.25))
    return _base("left_column", "left", primary, secondary,
                 p_valign="top", s_valign="top")


def right_column(cw: int, ch: int, rng: random.Random) -> Dict:
    c = _content(cw, ch)
    colw = int(c["w"] * (0.50 if cw > ch * 1.12 else 0.58))
    x = c["x"] + c["w"] - colw
    ph = int(c["h"] * 0.45)
    top = c["y"] + int(c["h"] * 0.10)
    primary = _box(x, top, colw, ph)
    secondary = _box(x, top + ph + _gap(cw, ch, 20), colw, int(c["h"] * 0.25))
    return _base("right_column", "right", primary, secondary,
                 p_valign="top", s_valign="top")


def frame_border(cw: int, ch: int, rng: random.Random) -> Dict:
    c = _content(cw, ch)
    inset = int(min(cw, ch) * 0.06)
    ci = _box(c["x"] + inset, c["y"] + inset, c["w"] - 2 * inset,
              c["h"] - 2 * inset)
    ph = int(ci["h"] * 0.34)
    primary = _box(ci["x"], ci["y"], ci["w"], ph)
    secondary = _box(ci["x"], ci["y"] + ci["h"] - int(ci["h"] * 0.16),
                     ci["w"], int(ci["h"] * 0.16))
    return _base("frame_border", "center", primary, secondary,
                 p_valign="top", s_valign="bottom", border=True)


def split_diagonal(cw: int, ch: int, rng: random.Random) -> Dict:
    c = _content(cw, ch)
    primary = _box(c["x"], c["y"] + int(c["h"] * 0.06),
                   int(c["w"] * 0.66), int(c["h"] * 0.42))
    secondary = _box(c["x"] + int(c["w"] * 0.30),
                     c["y"] + c["h"] - int(c["h"] * 0.30),
                     int(c["w"] * 0.70), int(c["h"] * 0.24))
    return _base("split_diagonal", "left", primary, secondary,
                 p_valign="top", s_valign="bottom",
                 secondary_roles=["body", "detail", "note"])


def badge_focus(cw: int, ch: int, rng: random.Random) -> Dict:
    c = _content(cw, ch)
    top = c["y"] + int(c["h"] * 0.10)
    ph = int(c["h"] * 0.42)
    badge_w = min(int(c["w"] * 0.36), int(ph * 1.1))
    gap = _gap(cw, ch, 30)
    prim_w = c["w"] - badge_w - gap
    primary = _box(c["x"], top, prim_w, ph)
    badge_box = _box(c["x"] + prim_w + gap, top, badge_w, ph)
    secondary = _box(c["x"], c["y"] + c["h"] - int(c["h"] * 0.16), c["w"],
                     int(c["h"] * 0.16))
    comp = _base("badge_focus", "left", primary, secondary,
                 p_valign="middle", s_valign="bottom", emphasis_mode="badge",
                 primary_roles=["headline", "subheadline"])
    comp["badge_box"] = badge_box
    return comp


# ---- tall / portrait (9:16 stories, 4:5, 2:3) ------------------------------
def story_top(cw: int, ch: int, rng: random.Random) -> Dict:
    """Text in the upper third, subject free in the middle / bottom."""
    c = _content(cw, ch)
    align = rng.choice(["center", "center", "left"])
    top = c["y"] + int(c["h"] * 0.05)
    ph = int(c["h"] * rng.choice([0.26, 0.30]))
    primary = _box(c["x"], top, c["w"], ph)
    secondary = _box(c["x"], top + ph + _gap(cw, ch, 24), c["w"],
                     int(c["h"] * 0.16))
    return _base("story_top", align, primary, secondary,
                 p_valign="top", s_valign="top", variant=align)


def story_bottom(cw: int, ch: int, rng: random.Random) -> Dict:
    """Subject in the upper half, all text stacked at the bottom."""
    c = _content(cw, ch)
    align = rng.choice(["center", "left"])
    sh = int(c["h"] * 0.14)
    ph = int(c["h"] * rng.choice([0.26, 0.30]))
    g = _gap(cw, ch, 24)
    secondary = _box(c["x"], c["y"] + c["h"] - sh, c["w"], sh)
    primary = _box(c["x"], c["y"] + c["h"] - sh - g - ph, c["w"], ph)
    return _base("story_bottom", align, primary, secondary,
                 p_valign="bottom", s_valign="bottom", variant=align)


def top_bottom(cw: int, ch: int, rng: random.Random) -> Dict:
    """Headline at the top, details at the bottom, subject in between."""
    c = _content(cw, ch)
    align = rng.choice(["center", "center", "left"])
    ph = int(c["h"] * rng.choice([0.28, 0.32]))
    sh = int(c["h"] * 0.18)
    primary = _box(c["x"], c["y"], c["w"], ph)
    secondary = _box(c["x"], c["y"] + c["h"] - sh, c["w"], sh)
    return _base("top_bottom", align, primary, secondary,
                 p_valign="top", s_valign="bottom", variant=align)


def badge_stack(cw: int, ch: int, rng: random.Random) -> Dict:
    """Headline on top, emphasis badge centered below, details at the bottom."""
    c = _content(cw, ch)
    g = _gap(cw, ch, 30)
    ph = int(c["h"] * 0.26)
    sh = int(c["h"] * 0.16)
    free_top = c["y"] + ph + g
    free = (c["y"] + c["h"] - sh - g) - free_top
    side = max(0, min(free, int(c["w"] * 0.5)))
    primary = _box(c["x"], c["y"], c["w"], ph)
    secondary = _box(c["x"], c["y"] + c["h"] - sh, c["w"], sh)
    comp = _base("badge_stack", "center", primary, secondary,
                 p_valign="bottom", s_valign="bottom", emphasis_mode="badge",
                 primary_roles=["headline", "subheadline"])
    comp["badge_box"] = _box(c["x"] + (c["w"] - side) // 2,
                             free_top + (free - side) // 2, side, side)
    return comp


# ---- landscape (4:3 .. 16:9 .. 1.91:1) -------------------------------------
def center_wide(cw: int, ch: int, rng: random.Random) -> Dict:
    """Centered stack on a narrower measure so lines stay readable."""
    c = _content(cw, ch)
    x = c["x"] + int(c["w"] * 0.12)
    w = int(c["w"] * 0.76)
    top = c["y"] + int(c["h"] * 0.12)
    ph = int(c["h"] * 0.46)
    primary = _box(x, top, w, ph)
    secondary = _box(x, top + ph + _gap(cw, ch, 20), w, int(c["h"] * 0.24))
    return _base("center_wide", "center", primary, secondary,
                 p_valign="bottom", s_valign="top")


def lower_third(cw: int, ch: int, rng: random.Random) -> Dict:
    """Video-style lower third: text bottom-left, subject top / right."""
    c = _content(cw, ch)
    pw = int(c["w"] * rng.choice([0.56, 0.64]))
    g = _gap(cw, ch, 20)
    py = c["y"] + int(c["h"] * 0.40)
    ph = int(c["h"] * 0.34)
    primary = _box(c["x"], py, pw, ph)
    secondary = _box(c["x"], py + ph + g, pw, c["y"] + c["h"] - py - ph - g)
    return _base("lower_third", "left", primary, secondary,
                 p_valign="bottom", s_valign="top", variant=str(pw))


# ---- wide banners (2:1 .. 3:1 .. 4:1) --------------------------------------
# short canvases make the type unit tiny, so wide templates push text sizes up;
# the band fit shrinks them again when they do not fit.
WIDE_TYPE_SCALE = 1.6


def _wide_side(cw: int, ch: int, rng: random.Random, right: bool) -> Dict:
    """Stacked text column on one side, subject on the other."""
    c = _content(cw, ch)
    colw = int(c["w"] * rng.choice([0.50, 0.56]))
    x = c["x"] + c["w"] - colw if right else c["x"]
    ph = int(c["h"] * 0.62)
    g = _gap(cw, ch, 16)
    primary = _box(x, c["y"], colw, ph)
    secondary = _box(x, c["y"] + ph + g, colw, c["h"] - ph - g)
    return _base("wide_right" if right else "wide_left",
                 "right" if right else "left", primary, secondary,
                 p_valign="bottom", s_valign="top",
                 type_scale=WIDE_TYPE_SCALE, variant=str(colw))


def wide_left(cw: int, ch: int, rng: random.Random) -> Dict:
    return _wide_side(cw, ch, rng, right=False)


def wide_right(cw: int, ch: int, rng: random.Random) -> Dict:
    return _wide_side(cw, ch, rng, right=True)


def wide_center(cw: int, ch: int, rng: random.Random) -> Dict:
    c = _content(cw, ch)
    x = c["x"] + int(c["w"] * 0.14)
    w = int(c["w"] * 0.72)
    ph = int(c["h"] * 0.62)
    g = _gap(cw, ch, 16)
    primary = _box(x, c["y"], w, ph)
    secondary = _box(x, c["y"] + ph + g, w, c["h"] - ph - g)
    return _base("wide_center", "center", primary, secondary,
                 p_valign="bottom", s_valign="top",
                 type_scale=WIDE_TYPE_SCALE)


def wide_badge(cw: int, ch: int, rng: random.Random) -> Dict:
    """Headline left, emphasis badge in the middle, details right."""
    c = _content(cw, ch)
    primary = _box(c["x"], c["y"], int(c["w"] * 0.46), c["h"])
    bw = min(c["h"], int(c["w"] * 0.20))
    badge_box = _box(c["x"] + int(c["w"] * 0.50), c["y"] + (c["h"] - bw) // 2,
                     bw, bw)
    secondary = _box(c["x"] + int(c["w"] * 0.72), c["y"], int(c["w"] * 0.28),
                     c["h"])
    comp = _base("wide_badge", "left", primary, secondary,
                 p_valign="middle", s_valign="middle", s_align="right",
                 emphasis_mode="badge", type_scale=WIDE_TYPE_SCALE,
                 primary_roles=["headline", "subheadline"])
    comp["badge_box"] = badge_box
    return comp


BUILDERS: Dict[str, Callable[[int, int, random.Random], Dict]] = {
    "top_headline": top_headline,
    "bottom_band": bottom_band,
    "center_stack": center_stack,
    "left_column": left_column,
    "right_column": right_column,
    "frame_border": frame_border,
    "split_diagonal": split_diagonal,
    "badge_focus": badge_focus,
    "story_top": story_top,
    "story_bottom": story_bottom,
    "top_bottom": top_bottom,
    "badge_stack": badge_stack,
    "center_wide": center_wide,
    "lower_third": lower_third,
    "wide_left": wide_left,
    "wide_right": wide_right,
    "wide_center": wide_center,
    "wide_badge": wide_badge,
}

_GENERAL = {"tall", "portrait", "square", "landscape"}
# which aspect classes a template is designed for
COMPAT: Dict[str, set] = {
    "top_headline": _GENERAL,
    "bottom_band": _GENERAL,
    "center_stack": _GENERAL,
    "frame_border": _GENERAL,
    "top_bottom": _GENERAL,
    "left_column": {"portrait", "square", "landscape"},
    "right_column": {"portrait", "square", "landscape"},
    "split_diagonal": {"portrait", "square", "landscape"},
    "badge_focus": {"portrait", "square", "landscape"},
    "story_top": {"tall", "portrait"},
    "story_bottom": {"tall", "portrait"},
    "badge_stack": {"tall", "portrait", "square"},
    "center_wide": {"landscape"},
    "lower_third": {"square", "landscape"},
    "wide_left": {"wide"},
    "wide_right": {"wide"},
    "wide_center": {"wide"},
    "wide_badge": {"wide"},
}

# closest equivalent when a hint does not suit the aspect class
ALIASES: Dict[str, Dict[str, str]] = {
    "tall": {"left_column": "story_bottom", "right_column": "story_bottom",
             "lower_third": "story_bottom", "split_diagonal": "top_bottom",
             "badge_focus": "badge_stack", "center_wide": "center_stack",
             "wide_left": "story_top", "wide_right": "story_top",
             "wide_center": "center_stack", "wide_badge": "badge_stack"},
    "portrait": {"center_wide": "center_stack", "lower_third": "bottom_band",
                 "wide_left": "left_column", "wide_right": "right_column",
                 "wide_center": "center_stack", "wide_badge": "badge_focus"},
    "square": {"story_top": "top_headline", "story_bottom": "bottom_band",
               "center_wide": "center_stack", "wide_left": "left_column",
               "wide_right": "right_column", "wide_center": "center_stack",
               "wide_badge": "badge_focus"},
    "landscape": {"story_top": "top_headline", "story_bottom": "lower_third",
                  "badge_stack": "badge_focus", "wide_left": "left_column",
                  "wide_right": "right_column", "wide_center": "center_wide",
                  "wide_badge": "badge_focus"},
    "wide": {"top_headline": "wide_center", "bottom_band": "wide_center",
             "center_stack": "wide_center", "frame_border": "wide_center",
             "top_bottom": "wide_center", "center_wide": "wide_center",
             "left_column": "wide_left", "lower_third": "wide_left",
             "split_diagonal": "wide_left", "story_top": "wide_left",
             "story_bottom": "wide_left", "right_column": "wide_right",
             "badge_focus": "wide_badge", "badge_stack": "wide_badge"},
}

# the badge template used for "emphasis" when the pool has none
BADGE_FOR = {"tall": "badge_stack", "portrait": "badge_focus",
             "square": "badge_focus", "landscape": "badge_focus",
             "wide": "wide_badge"}

# weighted auto-pick by aspect class, then design_type
WEIGHTS: Dict[str, Dict[str, Dict[str, int]]] = {
    "tall": {
        "poster": {"story_top": 3, "top_bottom": 3, "story_bottom": 2,
                   "center_stack": 2, "frame_border": 1, "badge_stack": 1},
        "banner": {"story_top": 2, "top_bottom": 2, "story_bottom": 2},
        "invitation": {"center_stack": 3, "frame_border": 3, "top_bottom": 1},
        "logo": {"center_stack": 4, "badge_stack": 1},
        "social": {"story_bottom": 3, "story_top": 2, "top_bottom": 2,
                   "center_stack": 2, "badge_stack": 1},
    },
    "portrait": {
        "poster": {"top_headline": 3, "top_bottom": 2, "bottom_band": 2,
                   "center_stack": 2, "frame_border": 1, "badge_focus": 1,
                   "split_diagonal": 1},
        "banner": {"top_headline": 2, "bottom_band": 2, "top_bottom": 2},
        "invitation": {"center_stack": 3, "frame_border": 3,
                       "top_headline": 1},
        "logo": {"center_stack": 4, "badge_focus": 1},
        "social": {"top_headline": 2, "bottom_band": 2, "center_stack": 2,
                   "top_bottom": 2, "story_bottom": 1, "badge_focus": 1},
    },
    "square": {
        "poster": {"top_headline": 3, "center_stack": 2, "bottom_band": 2,
                   "frame_border": 1, "badge_focus": 2, "split_diagonal": 1,
                   "left_column": 1},
        "banner": {"left_column": 3, "right_column": 2, "split_diagonal": 2,
                   "center_stack": 1, "bottom_band": 1},
        "invitation": {"center_stack": 3, "frame_border": 3,
                       "top_headline": 1},
        "logo": {"center_stack": 4, "badge_focus": 1},
        "social": {"top_headline": 2, "center_stack": 2, "badge_focus": 2,
                   "bottom_band": 2, "frame_border": 1},
    },
    "landscape": {
        "poster": {"left_column": 3, "right_column": 2, "center_wide": 2,
                   "lower_third": 2, "split_diagonal": 1, "frame_border": 1,
                   "badge_focus": 1, "top_headline": 1},
        "banner": {"left_column": 3, "right_column": 2, "split_diagonal": 2,
                   "lower_third": 2, "center_wide": 1},
        "invitation": {"center_wide": 3, "frame_border": 3},
        "logo": {"center_wide": 4, "badge_focus": 1},
        "social": {"left_column": 2, "right_column": 2, "center_wide": 2,
                   "lower_third": 2, "badge_focus": 1},
    },
    "wide": {
        "poster": {"wide_left": 3, "wide_right": 2, "wide_center": 2,
                   "wide_badge": 1},
        "banner": {"wide_left": 3, "wide_right": 2, "wide_center": 2,
                   "wide_badge": 1},
        "invitation": {"wide_center": 3, "wide_left": 1},
        "logo": {"wide_center": 4},
        "social": {"wide_left": 2, "wide_right": 2, "wide_center": 2,
                   "wide_badge": 1},
    },
}


def list_names() -> List[str]:
    return list(BUILDERS.keys())


def is_badge(name: str) -> bool:
    return name in ("badge_focus", "badge_stack", "wide_badge")


def resolve(name: str, aspect: str) -> Optional[str]:
    """Map a template name to one that suits `aspect` (None if unknown)."""
    if name not in BUILDERS:
        return None
    if aspect in COMPAT.get(name, ()):
        return name
    return ALIASES.get(aspect, {}).get(name)


def choose(cw: int, ch: int, seed: int, design_type: str,
           hint: str = "auto", has_emphasis: bool = False,
           force: bool = False) -> Dict:
    """Pick + build a composition deterministically from `seed`.

    `hint` (from the LLM) is honored only if it suits the canvas aspect, else
    mapped via ALIASES or ignored. `force=True` (the node's composition widget)
    builds exactly `hint` whatever the aspect.
    """
    check_supported(cw, ch)
    rng = random.Random(seed)
    aspect = aspect_class(cw, ch)
    name = None
    if hint in BUILDERS:
        name = hint if force else resolve(hint, aspect)
    if name is None:
        pool = WEIGHTS[aspect]
        weights = dict(pool.get(design_type, pool["poster"]))
        if has_emphasis:
            badges = [n for n in weights if is_badge(n)] or [BADGE_FOR[aspect]]
            for n in badges:
                weights[n] = weights.get(n, 0) + 3
        else:
            # a badge template without an emphasis text wastes its badge slot
            weights = {n: w for n, w in weights.items() if not is_badge(n)}
        names = list(weights.keys())
        wts = [weights[n] for n in names]
        name = rng.choices(names, weights=wts, k=1)[0]
    comp = BUILDERS[name](cw, ch, rng)
    comp["seed"] = seed
    comp["aspect"] = aspect
    comp["requested"] = hint
    return comp
