# -*- coding: utf-8 -*-
"""Shared constants, defaults and the final_json emitter.

The LLM input schema is intentionally simple (NO coordinates):

    {"language":"vietnamese|latin|...",
     "mood":"vintage|elegant|playful|minimal|bold|festive|...",
     "design_type":"poster|banner|invitation|logo|social",
     "composition_hint":"auto|<composition name>",
     "background_prompt":"...",
     "background_color":"#RRGGBB",
     "texts":[{"role":"headline|subheadline|emphasis|body|detail|note",
               "text":"...","priority":1}]}

final_json stays COMPATIBLE with the old flat text schema (text, role, font,
size_px, cx, cy, align, color, letter_spacing, shadow, outline_width,
outline_color) so the existing converter keeps working; new fields (box, lines,
line_height, scrim, shapes, canvas, composition, seed) are additive/optional.

`to_app_json()` is the single place that maps to the REAL app format. It is
kept isolated so it can be finalized once the app's design DTO is provided;
until then it passes the compatible schema through unchanged.
"""
from __future__ import annotations
from typing import Any, Dict, List

# Roles ordered by visual weight. Type scale is a ratio applied to a base size.
ROLES = ["headline", "subheadline", "emphasis", "body", "detail", "note"]

# Relative type scale (fraction of the canvas TYPE UNIT, see type_unit()).
# Layout may shrink to fit.
ROLE_SIZE_FRACTION = {
    "headline": 0.130,
    "subheadline": 0.055,
    "emphasis": 0.095,   # e.g. the "30%" discount number
    "body": 0.040,
    "detail": 0.034,
    "note": 0.026,
}

ROLE_LETTER_SPACING = {
    "headline": 0.02, "subheadline": 0.0, "emphasis": 0.01,
    "body": 0.0, "detail": 0.0, "note": 0.0,
}

# Preferred font style group per role, then per mood override. The font picker
# walks this order and takes the first group that has a glyph-covering font for
# the text's language.
ROLE_GROUP_PREF = {
    "headline": ["display", "decorative", "serif", "sans"],
    "subheadline": ["sans", "serif", "script"],
    "emphasis": ["display", "decorative", "sans"],
    "body": ["sans", "serif"],
    "detail": ["sans", "serif"],
    "note": ["sans", "serif"],
}

MOOD_GROUP_PREF = {
    "vintage": {"headline": ["serif", "display", "decorative"]},
    "elegant": {"headline": ["serif", "script", "display"],
                "subheadline": ["serif", "script", "sans"]},
    "playful": {"headline": ["decorative", "display", "script"],
                "subheadline": ["script", "sans"]},
    "festive": {"headline": ["decorative", "script", "display"]},
    "minimal": {"headline": ["sans", "display"]},
    "bold": {"headline": ["display", "sans"]},
}

DEFAULTS = {
    "language": "latin",
    "mood": "minimal",
    "design_type": "poster",
    "composition_hint": "auto",
    "background_prompt": "a smooth, calm abstract gradient backdrop with soft "
                         "studio light and plain empty negative space",
    "background_color": "#2B2B2B",
}

SAFE_MARGIN = 60          # px on the 1080 canvas
GRID_COLS = 12
DEFAULT_CANVAS = {"w": 1080, "h": 1080}
DEFAULT_BG = {"w": 1024, "h": 1024}


def type_unit(cw: int, ch: int) -> float:
    """Base size for the type scale: the canvas short side, grown a little
    for elongated canvases (9:16, 16:9) so text does not look tiny there.
    Equals the height on a square canvas."""
    short = float(min(cw, ch))
    return min(max((cw * ch) ** 0.5, short), 1.25 * short)


def role_size_px(role: str, unit: float) -> int:
    frac = ROLE_SIZE_FRACTION.get(role, ROLE_SIZE_FRACTION["body"])
    return max(12, round(frac * unit))


def group_pref(role: str, mood: str) -> List[str]:
    base = list(ROLE_GROUP_PREF.get(role, ["sans"]))
    over = MOOD_GROUP_PREF.get(mood, {}).get(role)
    if over:
        # mood preference first, then the role defaults as fallback
        return over + [g for g in base if g not in over]
    return base


def block_to_compat(block: Dict[str, Any]) -> Dict[str, Any]:
    """A layout block already carries old + new fields; select a clean dict."""
    keys = ["text", "role", "font", "size_px", "cx", "cy", "align", "color",
            "letter_spacing", "shadow", "outline_width", "outline_color",
            # additive/optional:
            "box", "lines", "line_height", "scrim"]
    return {k: block[k] for k in keys if k in block}


def to_app_json(layout: Dict[str, Any]) -> Dict[str, Any]:
    """Map an internal layout dict to the deliverable final_json.

    NOTE: kept isolated so it can be finalized against the real app design DTO.
    For now it emits the backward-compatible schema plus additive fields.
    """
    return {
        "language": layout.get("language", DEFAULTS["language"]),
        "background_color": layout.get("background_color",
                                       DEFAULTS["background_color"]),
        "canvas": layout.get("canvas", DEFAULT_CANVAS),
        "composition": layout.get("composition"),
        "seed": layout.get("seed"),
        "texts": [block_to_compat(b) for b in layout.get("blocks", [])],
        "shapes": layout.get("shapes", []),
    }
