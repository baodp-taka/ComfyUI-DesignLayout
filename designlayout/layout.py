# -*- coding: utf-8 -*-
"""The layout engine: turn a design_spec into placed text blocks + shapes.

Coordinates are on the CANVAS (default 1080x1080). The engine:
  * picks a composition (deterministic from seed),
  * groups texts into the primary / secondary bands,
  * measures every block with the real font (per-glyph fallback aware),
  * shrinks sizes so each band fits, stacks blocks with role-based gaps,
  * aligns lines inside each band,
  * emits emphasis as an inline block or a badge shape,
  * assigns provisional text colors from the background_color (the ZoneCheck
    node later refines them against the real rendered image).
"""
from __future__ import annotations
from typing import Dict, List, Optional

from . import compositions, measure
from .fonts import FontRegistry
from .schema import (ROLES, ROLE_LETTER_SPACING, role_size_px, type_unit,
                     DEFAULT_CANVAS)
from .colors import hex_to_rgb, rgb_to_hex, rel_luminance, ensure_contrast

GAP_FRACTION = 0.30       # vertical gap between blocks, x current size
BAND_FIT_ITERS = 4


class LayoutEngine:
    def __init__(self, registry: FontRegistry,
                 canvas: Optional[Dict] = None) -> None:
        self.reg = registry
        self.canvas = canvas or dict(DEFAULT_CANVAS)
        self.warnings: List[str] = []

    # ---- public -------------------------------------------------------------
    def layout(self, spec: Dict, seed: int = 0,
               force_composition: str = "") -> Dict:
        """`force_composition` (node widget) overrides the LLM hint and is
        built even if it does not suit the canvas aspect."""
        cw, ch = self.canvas["w"], self.canvas["h"]
        language = spec.get("language", "latin")
        mood = spec.get("mood", "minimal")
        dtype = spec.get("design_type", "poster")
        texts = [t for t in spec.get("texts", []) if str(t.get("text", "")).strip()]
        has_emph = any(t.get("role") == "emphasis" for t in texts)
        hint = force_composition or spec.get("composition_hint", "auto")
        comp = compositions.choose(cw, ch, seed, dtype, hint, has_emph,
                                   force=bool(force_composition))
        if hint not in ("", "auto") and comp["name"] != hint:
            self.warnings.append(f"composition {hint!r} does not suit "
                                 f"{comp['aspect']} canvas; using "
                                 f"{comp['name']!r}")
        self.unit = type_unit(cw, ch) * comp.get("type_scale", 1.0)

        bg_rgb = hex_to_rgb(spec.get("background_color", "#2B2B2B"))
        blocks: List[Dict] = []
        shapes: List[Dict] = []

        # split into bands by role
        prim_roles = comp["primary"]["roles"]
        badge_mode = comp["emphasis_mode"] == "badge"
        primary = [t for t in texts if t.get("role") in prim_roles
                   and not (badge_mode and t.get("role") == "emphasis")]
        secondary = [t for t in texts
                     if t.get("role") in comp["secondary"]["roles"]]
        placed_roles = set(prim_roles) | set(comp["secondary"]["roles"])
        leftover = [t for t in texts if t.get("role") not in placed_roles
                    or (t.get("role") == "emphasis" and badge_mode)]

        blocks += self._fill_band(comp["primary"], primary, language, mood,
                                  bg_rgb, comp["primary"]["align"])
        blocks += self._fill_band(comp["secondary"], secondary, language, mood,
                                  bg_rgb, comp["secondary"]["align"])

        # emphasis badge
        if badge_mode:
            emph = next((t for t in texts if t.get("role") == "emphasis"), None)
            if emph:
                b, s = self._make_badge(emph, comp, language, mood, bg_rgb)
                blocks.append(b)
                shapes.append(s)
        # any leftover roles that were not a badge -> drop into primary band top
        for t in leftover:
            if badge_mode and t.get("role") == "emphasis":
                continue
            self.warnings.append(f"role {t.get('role')} had no band; skipped")

        if comp.get("border"):
            shapes.append(self._border_shape(cw, ch, bg_rgb))

        return {
            "canvas": {"w": cw, "h": ch},
            "language": language,
            "mood": mood,
            "design_type": dtype,
            "composition": comp["name"],
            "aspect": comp["aspect"],
            "variant": comp.get("variant", ""),
            "seed": seed,
            "background_color": rgb_to_hex(bg_rgb),
            "blocks": blocks,
            "shapes": shapes,
            "zones": [{"role": b["role"], "box": b["box"]} for b in blocks],
        }

    # ---- band stacking ------------------------------------------------------
    def _measure_role(self, t: Dict, box: Dict, language: str, mood: str,
                      scale: float) -> Dict:
        role = t.get("role", "body")
        text = str(t["text"]).strip()
        target = max(12, int(role_size_px(role, self.unit) * scale))
        font_file, covered = self.reg.pick(text, language, role, mood)
        if not covered:
            self.warnings.append(f"{text!r}: no full-coverage font")
        ls = ROLE_LETTER_SPACING.get(role, 0.0)
        multiline = role in ("headline", "subheadline", "body")
        max_lines = 2 if role == "headline" else 3
        fit = measure.fit_block(self.reg, text, font_file, language,
                                max_w=box["w"], max_h=box["h"],
                                target_size=target, ls_frac=ls,
                                allow_multiline=multiline, max_lines=max_lines)
        fit.update({"role": role, "text": text, "font": font_file,
                    "letter_spacing": ls})
        return fit

    def _fill_band(self, band: Dict, items: List[Dict], language: str,
                   mood: str, bg_rgb, align: str) -> List[Dict]:
        if not items:
            return []
        box = band["box"]
        order = {r: i for i, r in enumerate(ROLES)}
        items = sorted(items, key=lambda t: order.get(t.get("role"), 99))

        scale = 1.0
        fits: List[Dict] = []
        for _ in range(BAND_FIT_ITERS):
            fits = [self._measure_role(t, box, language, mood, scale)
                    for t in items]
            total = sum(f["h"] for f in fits)
            total += sum(GAP_FRACTION * f["size_px"] for f in fits[:-1])
            if total <= box["h"] or scale < 0.5:
                break
            scale *= max(0.6, (box["h"] / total) * 0.98)

        total = sum(f["h"] for f in fits)
        total += sum(GAP_FRACTION * f["size_px"] for f in fits[:-1])
        valign = band["valign"]
        if valign == "top":
            cursor = box["y"]
        elif valign == "bottom":
            cursor = box["y"] + box["h"] - total
        else:
            cursor = box["y"] + (box["h"] - total) / 2.0

        blocks = []
        for i, f in enumerate(fits):
            bx = box["x"]
            bw = box["w"]
            by = cursor
            bh = f["h"]
            block = self._finalize(f, bx, by, bw, bh, align, bg_rgb)
            blocks.append(block)
            cursor += bh + GAP_FRACTION * f["size_px"]
        return blocks

    def _finalize(self, f: Dict, bx, by, bw, bh, align, bg_rgb) -> Dict:
        role = f["role"]
        color = self._role_color(role, bg_rgb)
        box = {"x": int(bx), "y": int(round(by)),
               "w": int(bw), "h": int(round(bh))}
        cx = box["x"] + box["w"] // 2
        cy = box["y"] + box["h"] // 2
        return {
            "role": role, "text": f["text"], "font": f["font"],
            "size_px": f["size_px"], "cx": cx, "cy": cy, "align": align,
            "color": rgb_to_hex(color),
            "letter_spacing": f["letter_spacing"],
            "shadow": role in ("headline", "subheadline", "emphasis"),
            "outline_width": 0, "outline_color": "#000000",
            "box": box, "lines": f["lines"],
            "line_height": f["line_height"],
        }

    def _role_color(self, role: str, bg_rgb):
        light_bg = rel_luminance(bg_rgb) > 0.5
        base = (25, 25, 25) if light_bg else (245, 245, 245)
        if role in ("subheadline", "emphasis"):
            accent = (200, 150, 80)  # warm accent; refined by ZoneCheck
            return ensure_contrast(accent, bg_rgb, 3.0)
        return ensure_contrast(base, bg_rgb, 4.5)

    # ---- badge + border -----------------------------------------------------
    def _make_badge(self, emph: Dict, comp: Dict, language, mood, bg_rgb):
        cw, ch = self.canvas["w"], self.canvas["h"]
        bb = comp.get("badge_box")
        if bb:
            r = int(min(bb["w"], bb["h"]) * 0.5)
            cx = bb["x"] + bb["w"] // 2
            cy = bb["y"] + bb["h"] // 2
        else:
            r = int(min(cw, ch) * 0.15)
            cx = int(cw * 0.72)
            cy = int(ch * 0.30)
        text = str(emph["text"]).strip()
        font_file, _ = self.reg.pick(text, language, "emphasis", mood)
        fit = measure.fit_block(self.reg, text, font_file, language,
                                max_w=r * 1.6, max_h=r * 1.2,
                                target_size=int(r * 0.9), ls_frac=0.0,
                                allow_multiline=True, max_lines=2)
        badge_fill = (190, 60, 50)
        txt_color = ensure_contrast((255, 255, 255), badge_fill, 3.0)
        box = {"x": cx - r, "y": cy - r, "w": 2 * r, "h": 2 * r}
        block = {
            "role": "emphasis", "text": text, "font": font_file,
            "size_px": fit["size_px"], "cx": cx, "cy": cy, "align": "center",
            "color": rgb_to_hex(txt_color), "letter_spacing": 0.0,
            "shadow": False, "outline_width": 0, "outline_color": "#000000",
            "box": box, "lines": fit["lines"], "line_height": fit["line_height"],
            "valign": "middle",
        }
        shape = {"type": "badge", "shape": "circle", "cx": cx, "cy": cy,
                 "r": r, "fill": rgb_to_hex(badge_fill)}
        return block, shape

    def _border_shape(self, cw, ch, bg_rgb):
        inset = int(min(cw, ch) * 0.04)
        stroke = (245, 245, 245) if rel_luminance(bg_rgb) < 0.5 \
            else (25, 25, 25)
        return {"type": "border", "x": inset, "y": inset,
                "w": cw - 2 * inset, "h": ch - 2 * inset,
                "stroke": rgb_to_hex(stroke), "width": max(2, int(cw * 0.004))}
