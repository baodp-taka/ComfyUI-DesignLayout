# -*- coding: utf-8 -*-
"""Declarative layout templates (~200), generated from archetypes x variants.

A template is plain data: where the two text bands go (fractions of the safe
area), their alignment, an optional badge, a panel / ribbon, decorations and
which design types + canvas aspects it suits. `build()` turns one into the
`comp` dict the layout engine consumes, so the band fitting, hierarchy and
no-overlap guarantees apply to every template.

Adding a template = appending one `_t(...)` call (or a new archetype loop).
Ids are stable, e.g. "wedding.monogram.double_frame".

Fractions are of the CONTENT box (safe margins + optional frame inset):
    box = (x, y, w, h)  in 0..1
Decor specs are resolved after the text is placed (see decor.py):
    {"type": "border", "style": "single|double|rounded"}
    {"type": "corners"}                      corner brackets
    {"type": "divider", "ornament": "none|diamond|dot|dots"}
    {"type": "accent_bar"}                   short bar under the headline
    {"type": "dots"}                         confetti in free margins
    {"type": "ring"}                         circle around the primary band
Panel: {"target": "all|primary|secondary", "style": "light|dark|accent",
        "full_width": bool} -- a plate behind the text.
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional

from .schema import SAFE_MARGIN, DEFAULT_CANVAS

DESIGN_TYPES = ["poster", "banner", "thumbnail", "invitation", "wedding",
                "birthday", "social", "logo"]
GENERAL = {"tall", "portrait", "square", "landscape"}
PORTRAITISH = {"tall", "portrait", "square"}
FLATISH = {"portrait", "square", "landscape"}
WIDE = {"wide", "landscape"}

PRIMARY_ROLES = ["headline", "subheadline", "emphasis"]
SECONDARY_ROLES = ["body", "detail", "note"]
FRAME_INSET = 0.045       # extra text inset when a frame is drawn (x short)
WIDE_TYPE_SCALE = 1.6     # short canvases -> bigger type unit, fit shrinks

TEMPLATES: List[Dict] = []


def _t(tid: str, family: str, design_types, aspects, primary, secondary,
       align: str = "center", p_valign: str = "middle",
       s_valign: str = "top", s_align: Optional[str] = None,
       badge=None, badge_shape: str = "circle", decor=None, panel=None,
       ribbon: bool = False, outline: float = 0.0, type_scale: float = 1.0,
       inset: float = 0.0, weight: float = 1.0) -> None:
    decor = list(decor or [])
    framed = any(d["type"] in ("border", "corners") for d in decor)
    TEMPLATES.append({
        "id": tid, "family": family,
        "design_types": set(design_types), "aspects": set(aspects),
        "primary": {"box": primary, "align": align, "valign": p_valign},
        "secondary": {"box": secondary, "align": s_align or align,
                      "valign": s_valign},
        "badge": badge, "badge_shape": badge_shape,
        "decor": decor, "panel": panel, "ribbon": ribbon,
        "outline": outline, "type_scale": type_scale,
        "inset": inset if inset else (FRAME_INSET if framed else 0.0),
        "weight": weight,
    })


# ---- decor presets -----------------------------------------------------------
D_NONE: List[Dict] = []
D_DIVIDER = [{"type": "divider", "ornament": "none"}]
D_DIV_DIAMOND = [{"type": "divider", "ornament": "diamond"}]
D_DIV_DOT = [{"type": "divider", "ornament": "dot"}]
D_DIV_DOTS = [{"type": "divider", "ornament": "dots"}]
D_BAR = [{"type": "accent_bar"}]
D_BORDER = [{"type": "border", "style": "single"}]
D_DOUBLE = [{"type": "border", "style": "double"}]
D_ROUNDED = [{"type": "border", "style": "rounded"}]
D_CORNERS = [{"type": "corners"}]
D_DOTS = [{"type": "dots"}]
D_RING = [{"type": "ring"}]

_ALIGNS3 = ["left", "center", "right"]


def _mirror(box):
    x, y, w, h = box
    return (1.0 - x - w, y, w, h)


# ============================================================================
# POSTER
# ============================================================================
P = ["poster"]
for a in _ALIGNS3:
    for dn, d in [("plain", D_NONE), ("divider", D_DIVIDER), ("bar", D_BAR),
                  ("corners", D_CORNERS)]:
        _t(f"poster.stack_top.{a}.{dn}", "top", P + ["social"], GENERAL,
           (0, 0, 1, .36), (0, .42, 1, .30), align=a, s_valign="top", decor=d)
for a in ["left", "center"]:
    for dn, d in [("plain", D_NONE), ("bar", D_BAR), ("divider", D_DIVIDER)]:
        _t(f"poster.stack_bottom.above.{a}.{dn}", "bottom", P + ["social"],
           GENERAL, (0, .62, 1, .38), (0, .42, 1, .16), align=a,
           p_valign="middle", s_valign="bottom", decor=d)
        _t(f"poster.stack_bottom.below.{a}.{dn}", "bottom", P + ["social"],
           GENERAL, (0, .48, 1, .32), (0, .84, 1, .16), align=a,
           p_valign="bottom", s_valign="top", decor=d)
for a in _ALIGNS3:
    for dn, d in [("plain", D_NONE), ("divider", D_DIVIDER), ("bar", D_BAR)]:
        _t(f"poster.top_bottom.{a}.{dn}", "top_bottom", P, GENERAL,
           (0, 0, 1, .32), (0, .80, 1, .20), align=a, p_valign="top",
           s_valign="bottom", decor=d)
for dn, d in [("plain", D_NONE), ("divider", D_DIV_DIAMOND),
              ("frame", D_BORDER), ("corners", D_CORNERS),
              ("double", D_DOUBLE)]:
    _t(f"poster.center.{dn}", "center", P + ["invitation"], GENERAL,
       (0, .22, 1, .40), (0, .66, 1, .20), decor=d)
for side in ["left", "right"]:
    for vpos, pb, sb in [("top", (0, .04, .56, .40), (0, .48, .56, .26)),
                         ("middle", (0, .18, .56, .36), (0, .58, .56, .24)),
                         ("bottom", (0, .40, .56, .34), (0, .78, .56, .22))]:
        if side == "right":
            pb, sb = _mirror(pb), _mirror(sb)
        for dn, d in [("plain", D_NONE), ("bar", D_BAR)]:
            _t(f"poster.column.{side}.{vpos}.{dn}", "column",
               P + ["banner"], FLATISH, pb, sb, align=side,
               p_valign="bottom" if vpos != "top" else "top", decor=d)
for side in ["left", "right"]:
    for dn, d in [("plain", D_NONE), ("bar", D_BAR)]:
        pb, sb = (0, .04, .66, .42), (.34, .72, .66, .28)
        pa, sa = "left", "right"
        if side == "right":
            pb, sb, pa, sa = _mirror(pb), _mirror(sb), "right", "left"
        _t(f"poster.split.{side}.{dn}", "split", P, FLATISH, pb, sb,
           align=pa, s_align=sa, p_valign="top", s_valign="bottom", decor=d)
for side in ["right", "left"]:
    for shape in ["circle", "star", "pill"]:
        pb, bb, sb = (0, .08, .60, .42), (.64, .08, .36, .42), (0, .80, 1, .20)
        if side == "left":
            pb, bb = (.40, .08, .60, .42), (0, .08, .36, .42)
        _t(f"poster.badge_side.{side}.{shape}", "badge", P + ["social"],
           FLATISH, pb, sb, align="left", p_valign="middle",
           s_valign="bottom", badge=bb, badge_shape=shape)
for shape in ["circle", "star", "diamond"]:
    _t(f"poster.badge_stack.{shape}", "badge", P + ["social", "birthday"],
       PORTRAITISH, (0, 0, 1, .28), (0, .82, 1, .18), p_valign="bottom",
       s_valign="bottom", badge=(.25, .34, .50, .40), badge_shape=shape)
for style in ["dark", "light", "accent"]:
    for a in ["center", "left"]:
        _t(f"poster.bottom_bar.{style}.{a}", "bottom", P + ["social"],
           GENERAL, (0, .06, 1, .40), (0, .80, 1, .20), align=a,
           s_valign="middle",
           panel={"target": "secondary", "style": style, "full_width": True})
for pos, pb, sb in [("center", (.10, .24, .80, .30), (.10, .58, .80, .18)),
                    ("bottom", (.08, .54, .84, .22), (.08, .80, .84, .14)),
                    ("top", (.08, .06, .84, .22), (.08, .32, .84, .14))]:
    for style in ["light", "dark"]:
        _t(f"poster.card.{pos}.{style}", "card", P + ["social"], GENERAL,
           pb, sb, p_valign="bottom" if pos != "top" else "top",
           panel={"target": "all", "style": style})

# ============================================================================
# BANNER (wide strips + landscape web banners)
# ============================================================================
B = ["banner"]
for side in ["left", "right"]:
    for colw in [.50, .60]:
        for dn, d in [("plain", D_NONE), ("bar", D_BAR),
                      ("divider", D_DIVIDER)]:
            pb, sb = (0, 0, colw, .62), (0, .66, colw, .34)
            if side == "right":
                pb, sb = _mirror(pb), _mirror(sb)
            _t(f"banner.side.{side}.{int(colw * 100)}.{dn}", "side", B,
               WIDE, pb, sb, align=side, p_valign="bottom", decor=d,
               type_scale=WIDE_TYPE_SCALE)
for dn, d in [("plain", D_NONE), ("divider", D_DIV_DIAMOND),
              ("frame", D_BORDER)]:
    _t(f"banner.center.{dn}", "center", B, WIDE, (.14, 0, .72, .62),
       (.14, .66, .72, .34), p_valign="bottom", decor=d,
       type_scale=WIDE_TYPE_SCALE)
for style in ["light", "dark"]:
    _t(f"banner.center.card.{style}", "card", B, WIDE, (.18, .06, .64, .56),
       (.18, .66, .64, .28), p_valign="bottom",
       panel={"target": "all", "style": style}, type_scale=WIDE_TYPE_SCALE)
for pos in ["middle", "right"]:
    for shape in ["circle", "pill", "star"]:
        if pos == "middle":
            pb, bb, sb = (0, 0, .46, 1), (.50, 0, .20, 1), (.72, 0, .28, 1)
        else:
            pb, bb, sb = (0, 0, .50, 1), (.82, 0, .18, 1), (.52, 0, .28, 1)
        _t(f"banner.badge.{pos}.{shape}", "badge", B, WIDE, pb, sb,
           align="left", s_align="right" if pos == "middle" else "left",
           s_valign="middle", badge=bb, badge_shape=shape,
           type_scale=WIDE_TYPE_SCALE)
for sa in ["left", "right"]:
    _t(f"banner.split.{sa}", "split", B, WIDE, (0, 0, .55, 1),
       (.60, 0, .40, 1), align="left", s_align=sa, s_valign="middle",
       type_scale=WIDE_TYPE_SCALE)
for style in ["light", "dark", "accent"]:
    _t(f"banner.panel_left.{style}", "card", B, WIDE, (.02, .06, .46, .56),
       (.02, .66, .46, .28), align="left", p_valign="bottom",
       panel={"target": "all", "style": style}, type_scale=WIDE_TYPE_SCALE)

# ============================================================================
# THUMBNAIL (YouTube 16:9, square, shorts)
# ============================================================================
T = ["thumbnail"]
TH_ASPECTS = {"landscape", "square", "wide"}
TH_ALL = TH_ASPECTS | {"tall", "portrait"}   # Shorts / Reels covers too
for side in ["left", "right"]:
    for oname, o in [("clean", 0.0), ("outline", 0.05)]:
        for bname in ["none", "star", "pill"]:
            pb, sb = (0, .06, .60, .70), (0, .80, .60, .20)
            bb = (.66, .04, .32, .40)
            if side == "right":
                pb, sb, bb = _mirror(pb), _mirror(sb), _mirror(bb)
            _t(f"thumb.big_side.{side}.{oname}.{bname}",
               "badge" if bname != "none" else "side", T, TH_ASPECTS,
               pb, sb, align=side, p_valign="middle",
               badge=bb if bname != "none" else None,
               badge_shape=bname if bname != "none" else "circle",
               outline=o, type_scale=1.25)
for pos in ["top", "bottom"]:
    for oname, o in [("clean", 0.0), ("outline", 0.05)]:
        for dn, d in [("plain", D_NONE), ("bar", D_BAR)]:
            if pos == "top":
                pb, sb, pv, sv = (0, 0, 1, .50), (0, .56, 1, .16), "top", "top"
            else:
                pb, sb, pv, sv = (0, .44, 1, .42), (0, .88, 1, .12), \
                    "bottom", "top"
            _t(f"thumb.big_{pos}.{oname}.{dn}", pos, T, TH_ALL, pb, sb,
               p_valign=pv, s_valign=sv, outline=o, decor=d,
               type_scale=1.25)
for style in ["dark", "accent"]:
    _t(f"thumb.center_huge.{style}_strip", "center", T, TH_ALL,
       (.04, .14, .92, .52), (.10, .74, .80, .16), outline=0.05,
       panel={"target": "subheadline", "style": style, "full_width": True},
       type_scale=1.3)
for side in ["left", "right"]:
    for style in ["accent", "dark"]:
        pb, sb = (0, .08, .56, .60), (0, .76, .56, .20)
        if side == "right":
            pb, sb = _mirror(pb), _mirror(sb)
        _t(f"thumb.block.{side}.{style}", "card", T, TH_ASPECTS, pb, sb,
           align=side, panel={"target": "primary", "style": style},
           type_scale=1.2)

# ============================================================================
# WEDDING
# ============================================================================
W = ["wedding"]
WED_DECOR = [("double_frame", D_DOUBLE), ("corners", D_CORNERS),
             ("frame_diamond", D_BORDER + D_DIV_DIAMOND),
             ("rounded_dot", D_ROUNDED + D_DIV_DOT),
             ("corners_diamond", D_CORNERS + D_DIV_DIAMOND),
             ("diamond", D_DIV_DIAMOND)]
for dn, d in WED_DECOR:
    _t(f"wedding.monogram.{dn}", "frame", W, PORTRAITISH,
       (.06, .22, .88, .36), (.06, .66, .88, .22), decor=d, weight=1.5)
    _t(f"wedding.monogram.{dn}.card", "card", W, PORTRAITISH,
       (.10, .24, .80, .32), (.10, .62, .80, .20), decor=d,
       panel={"target": "all", "style": "light"})
for dn, d in [("double_frame", D_DOUBLE), ("corners", D_CORNERS),
              ("diamond", D_DIV_DIAMOND)]:
    _t(f"wedding.names_top.{dn}", "top_bottom", W, PORTRAITISH,
       (.06, .06, .88, .34), (.06, .74, .88, .22), p_valign="middle",
       s_valign="bottom", decor=d)
for dn, d in [("corners", D_CORNERS), ("frame", D_BORDER)]:
    _t(f"wedding.names_bottom.{dn}", "bottom", W, PORTRAITISH,
       (.06, .48, .88, .30), (.06, .82, .88, .18), p_valign="bottom",
       decor=d)
for side in ["left", "right"]:
    for dn, d in [("frame", D_BORDER), ("corners", D_CORNERS)]:
        pb, sb = (0, .16, .60, .40), (0, .62, .60, .24)
        if side == "right":
            pb, sb = _mirror(pb), _mirror(sb)
        _t(f"wedding.side.{side}.{dn}", "column", W, FLATISH, pb, sb,
           align=side, p_valign="bottom", decor=d)

# ============================================================================
# BIRTHDAY
# ============================================================================
BD = ["birthday"]
for dn, d, rib in [("confetti", D_DOTS, False),
                   ("confetti_ribbon", D_DOTS, True),
                   ("frame_confetti", D_ROUNDED + D_DOTS, False)]:
    _t(f"birthday.center.{dn}", "center", BD, GENERAL, (.04, .20, .92, .38),
       (.04, .64, .92, .22), decor=d, ribbon=rib, weight=1.3)
    for shape in ["star", "circle"]:
        _t(f"birthday.age_badge.{dn}.{shape}", "badge", BD, PORTRAITISH,
           (0, .02, 1, .30), (0, .78, 1, .22), p_valign="bottom",
           badge=(.28, .38, .44, .34), badge_shape=shape, decor=d,
           ribbon=rib)
for a in ["center", "left"]:
    for dn, d, rib in [("confetti", D_DOTS, False), ("bar", D_BAR, False),
                       ("ribbon", D_NONE, True)]:
        _t(f"birthday.top.{a}.{dn}", "top", BD, GENERAL, (0, 0, 1, .38),
           (0, .44, 1, .28), align=a, decor=d, ribbon=rib)
for style in ["light", "accent"]:
    for dn, d in [("plain", D_NONE), ("confetti", D_DOTS)]:
        _t(f"birthday.card.{style}.{dn}", "card", BD, GENERAL,
           (.10, .24, .80, .30), (.10, .58, .80, .18), p_valign="bottom",
           decor=d, panel={"target": "all", "style": style})

# ============================================================================
# INVITATION (events, openings, parties)
# ============================================================================
I = ["invitation"]
for dn, d in [("frame", D_BORDER), ("corners", D_CORNERS),
              ("divider", D_DIV_DIAMOND), ("double", D_DOUBLE)]:
    _t(f"invite.center.{dn}", "frame", I, GENERAL, (.04, .20, .92, .40),
       (.04, .66, .92, .22), decor=d, weight=1.3)
for style in ["light", "dark"]:
    for pos, pb, sb in [("center", (.10, .24, .80, .30), (.10, .58, .80, .18)),
                        ("bottom", (.08, .52, .84, .22), (.08, .78, .84, .16))]:
        _t(f"invite.card.{pos}.{style}", "card", I, GENERAL, pb, sb,
           p_valign="bottom", decor=D_DIV_DOT,
           panel={"target": "all", "style": style})
for dn, d in [("divider", D_DIVIDER), ("corners", D_CORNERS)]:
    _t(f"invite.top_bottom.{dn}", "top_bottom", I, GENERAL, (0, 0, 1, .34),
       (0, .78, 1, .22), p_valign="top", s_valign="bottom", decor=d)
for side in ["left", "right"]:
    for dn, d in [("frame", D_BORDER), ("plain", D_NONE)]:
        pb, sb = (0, .14, .60, .40), (0, .60, .60, .26)
        if side == "right":
            pb, sb = _mirror(pb), _mirror(sb)
        _t(f"invite.side.{side}.{dn}", "column", I, FLATISH, pb, sb,
           align=side, p_valign="bottom", decor=d)

# ============================================================================
# SOCIAL (feed posts, stories)
# ============================================================================
S = ["social"]
for dn, d in [("plain", D_NONE), ("divider", D_DIVIDER),
              ("corners", D_CORNERS), ("bar", D_BAR)]:
    _t(f"social.quote.{dn}", "center", S, PORTRAITISH, (.06, .26, .88, .38),
       (.06, .68, .88, .16), decor=d)
for pos in ["top", "bottom"]:
    for pn, pan in [("plain", None),
                    ("dark_strip", {"target": "all", "style": "dark",
                                    "full_width": True}),
                    ("light_strip", {"target": "all", "style": "light",
                                     "full_width": True})]:
        if pos == "top":
            pb, sb, pv = (0, .04, 1, .24), (0, .30, 1, .12), "top"
        else:
            pb, sb, pv = (0, .62, 1, .22), (0, .86, 1, .12), "bottom"
        _t(f"social.story_{pos}.{pn}", pos, S, {"tall", "portrait"}, pb, sb,
           p_valign=pv, panel=pan)
for shape in ["circle", "star"]:
    _t(f"social.badge.{shape}", "badge", S, FLATISH, (0, .06, .60, .40),
       (0, .80, 1, .20), align="left", s_valign="bottom",
       badge=(.64, .06, .36, .40), badge_shape=shape)

# ============================================================================
# LOGO
# ============================================================================
L = ["logo"]
for dn, d in [("plain", D_NONE), ("ring", D_RING), ("divider", D_DIVIDER),
              ("bar", D_BAR)]:
    _t(f"logo.center.{dn}", "center", L, GENERAL | {"wide"},
       (.10, .30, .80, .28), (.10, .62, .80, .12), decor=d)
for dn, d in [("ring_divider", D_RING + D_DIV_DOT), ("frame", D_BORDER)]:
    _t(f"logo.stack.{dn}", "frame", L, PORTRAITISH, (.14, .28, .72, .30),
       (.14, .62, .72, .10), decor=d)
_t("logo.wordmark.left", "column", L, GENERAL | {"wide"},
   (.04, .34, .80, .22), (.04, .58, .80, .10), align="left", p_valign="bottom")

BY_ID: Dict[str, Dict] = {t["id"]: t for t in TEMPLATES}
FAMILIES = sorted({t["family"] for t in TEMPLATES})

# old template names / LLM hints -> family (soft preference in auto pick)
FAMILY_ALIASES = {
    "top_headline": "top", "story_top": "top", "bottom_band": "bottom",
    "story_bottom": "bottom", "lower_third": "bottom",
    "center_stack": "center", "center_wide": "center",
    "wide_center": "center", "left_column": "column",
    "right_column": "column", "frame_border": "frame",
    "split_diagonal": "split", "badge_focus": "badge",
    "badge_stack": "badge", "wide_badge": "badge", "wide_left": "side",
    "wide_right": "side", "top_bottom": "top_bottom",
}


def is_badge(t: Dict) -> bool:
    return t.get("badge") is not None


def build(t: Dict, cw: int, ch: int, aspect: str) -> Dict:
    """Template (fractions) -> comp dict (canvas px) for the layout engine."""
    short = min(cw, ch)
    s = short / float(DEFAULT_CANVAS["h"])
    my = round(SAFE_MARGIN * s)
    mx = max(my, round(0.03 * cw))
    ins = round(t["inset"] * short)
    c = {"x": mx + ins, "y": my + ins,
         "w": cw - 2 * (mx + ins), "h": ch - 2 * (my + ins)}

    def px(f):
        x, y, w, h = f
        return {"x": int(c["x"] + x * c["w"]), "y": int(c["y"] + y * c["h"]),
                "w": int(w * c["w"]), "h": int(h * c["h"])}

    badge = is_badge(t)
    ts = t["type_scale"]
    if ts == WIDE_TYPE_SCALE and aspect != "wide":
        ts = 1.0          # the wide boost is only for short strips
    comp = {
        "name": t["id"], "template": t["id"], "family": t["family"],
        "align": t["primary"]["align"],
        "primary": {"box": px(t["primary"]["box"]),
                    "valign": t["primary"]["valign"],
                    "align": t["primary"]["align"],
                    "roles": ["headline", "subheadline"] if badge
                    else PRIMARY_ROLES},
        "secondary": {"box": px(t["secondary"]["box"]),
                      "valign": t["secondary"]["valign"],
                      "align": t["secondary"]["align"],
                      "roles": SECONDARY_ROLES},
        "emphasis_mode": "badge" if badge else "inline",
        "border": False,
        "variant": t["id"],
        "type_scale": ts,
        "inset_px": ins,
        "decor": t["decor"],
        "panel": t["panel"],
        "ribbon": t["ribbon"],
        "outline": t["outline"],
        "badge_shape": t["badge_shape"],
        # thumbnails stack big words: up to 3 headline lines
        "headline_lines": 3 if "thumbnail" in t["design_types"] else None,
        # extra room after the headline when a divider / bar goes there
        "headline_gap": 0.75 if any(d["type"] in ("divider", "accent_bar")
                                    for d in t["decor"]) else None,
    }
    if badge:
        comp["badge_box"] = px(t["badge"])
    return comp


def pool(design_type: str, aspect: str) -> List[Dict]:
    dt = design_type if design_type in DESIGN_TYPES else "poster"
    out = [t for t in TEMPLATES if dt in t["design_types"]
           and aspect in t["aspects"]]
    if not out:  # e.g. a wedding card on a 3:1 strip -> any template there
        out = [t for t in TEMPLATES if aspect in t["aspects"]]
    return out


def pick(design_type: str, aspect: str, rng: random.Random,
         hint: str = "auto", has_emphasis: bool = False) -> Dict:
    cands = pool(design_type, aspect)
    if has_emphasis:
        weights = [t["weight"] * (2.5 if is_badge(t) else 1.0)
                   for t in cands]
    else:
        # a badge template without an emphasis text wastes its badge slot
        no_badge = [t for t in cands if not is_badge(t)]
        cands = no_badge or cands
        weights = [t["weight"] for t in cands]
    fam = FAMILY_ALIASES.get(hint, hint)
    if fam in FAMILIES:
        weights = [w * (4.0 if t["family"] == fam else 1.0)
                   for w, t in zip(weights, cands)]
    return rng.choices(cands, weights=weights, k=1)[0]
