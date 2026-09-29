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

from . import compositions, measure, decor
from .fonts import FontRegistry
from .schema import (ROLES, ROLE_LETTER_SPACING, role_size_px, type_unit,
                     DEFAULT_CANVAS)
from .colors import (hex_to_rgb, rgb_to_hex, rel_luminance, ensure_contrast,
                     fit_contrast, fit_intent, neutral_on, required_contrast,
                     contrast_ratio, parse_hex, WHITE, BLACK)

ACCENT_ROLES = ("subheadline", "emphasis")
DEFAULT_ACCENT = (200, 150, 80)   # warm gold when the LLM gives no accent
DEFAULT_BADGE = (190, 60, 50)

GAP_FRACTION = 0.30       # vertical gap between blocks, x current size
FIT_ITERS = 12            # shared-scale shrink steps before the hard floor
SCALE_STEP = 0.9
# Readable size floors (fraction of the type unit). Text is shrunk down to
# these first; only if it still does not fit are the hard floors used, and
# only then is the least important line dropped (with a warning).
MIN_FRAC = {"headline": 0.050, "subheadline": 0.028, "emphasis": 0.040,
            "body": 0.024, "detail": 0.022, "note": 0.018}
HARD_MIN_FRAC = 0.015
MAX_LINES = {"headline": 2, "emphasis": 2, "subheadline": 3, "body": 3,
             "detail": 2, "note": 2}
# size must not increase down this chain (headline >= subheadline >= body
# >= ...); emphasis is capped by the headline only (it may out-size the rest)
HIERARCHY = ["headline", "subheadline", "body", "detail", "note"]
HIERARCHY_RATIO = 0.92


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
        # optional LLM color intent (hue kept, lightness fitted for contrast)
        self.text_intent = self._intent(spec.get("text_color"))
        self.accent_intent = self._intent(spec.get("accent_color"))
        blocks: List[Dict] = []
        shapes: List[Dict] = []

        # split into bands by role. In badge templates only the FIRST emphasis
        # goes in the badge; any further emphasis is stacked inline in the
        # primary band instead of being dropped.
        badge_mode = comp["emphasis_mode"] == "badge"
        emphases = [t for t in texts if t.get("role") == "emphasis"]
        badge_emph = emphases[0] if (badge_mode and emphases) else None
        prim_roles = set(comp["primary"]["roles"])
        if badge_mode:
            prim_roles.add("emphasis")
        primary: List[Dict] = []
        secondary: List[Dict] = []
        for t in texts:
            if t is badge_emph:
                continue
            if t.get("role") in prim_roles:
                primary.append(t)
            else:
                # secondary roles + anything unexpected -> details band
                secondary.append(t)

        # a panel (plate) changes what the text sits on -> fit colors to it
        pfill = decor.panel_fill(comp.get("panel"), self.accent_intent,
                                 DEFAULT_ACCENT)
        band_bgs = [bg_rgb, bg_rgb]
        if pfill:
            target = comp["panel"].get("target", "all")
            for i, name in enumerate(("primary", "secondary")):
                if target in ("all", name):
                    band_bgs[i] = decor.blend(pfill[0], bg_rgb, pfill[1])

        obstacles: List[Dict] = []
        badge = None
        if badge_emph is not None:
            badge = self._make_badge(badge_emph, comp, language, mood, bg_rgb)
            obstacles.append(badge[0]["box"])

        self._comp = comp
        blocks += self._fit_bands(comp, [primary, secondary], obstacles,
                                  language, mood, band_bgs)
        bands = [comp["primary"], comp["secondary"]]
        if pfill:
            p = decor.panel_shape(comp["panel"], blocks, bands, pfill[0],
                                  pfill[1], cw, ch)
            if p:
                shapes.append(p)
                plate = decor.blend(pfill[0], bg_rgb, pfill[1])
                for b in blocks:
                    if not b.get("on_panel"):
                        continue
                    if pfill[2] is not None:    # solid accent plate
                        b.update(color=rgb_to_hex(pfill[2]), intent_color=None,
                                 on_shape=True, shadow=False)
                    else:                       # light / dark plate
                        col, intent = self._role_color(b["role"], plate,
                                                       b["size_px"])
                        b["color"] = rgb_to_hex(col)
        if comp.get("ribbon"):
            sub = next((b for b in blocks if b["role"] == "subheadline"
                        and not b.get("on_panel")), None)
            if sub:
                shapes.append(decor.ribbon_shape(
                    sub, self.accent_intent or DEFAULT_ACCENT, cw, ch))
        if comp.get("outline"):
            for b in blocks:
                if b["role"] == "headline" and not b.get("on_shape"):
                    b["outline_width"] = max(2, round(comp["outline"] *
                                                      b["size_px"]))
                    b["outline_color"] = rgb_to_hex(
                        BLACK if rel_luminance(hex_to_rgb(b["color"])) > 0.4
                        else WHITE)
                    b["shadow"] = False
        if comp.get("decor"):
            avoid = [s for s in shapes if s["type"] in ("panel", "ribbon")]
            if badge:
                avoid.append(badge[0]["box"])
            shapes += decor.build_decor(
                comp["decor"], blocks, bands, self._limits(comp), cw, ch,
                self._decor_colors(blocks, bg_rgb), seed, avoid)
        if badge:
            blocks.append(badge[0])
            shapes.append(badge[1])

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
            "template": comp.get("template", comp["name"]),
            "family": comp.get("family"),
            "seed": seed,
            "background_color": rgb_to_hex(bg_rgb),
            "palette": {"text": parse_hex(spec.get("text_color")),
                        "accent": parse_hex(spec.get("accent_color"))},
            "blocks": blocks,
            "shapes": shapes,
            "zones": [{"role": b["role"], "box": b["box"]} for b in blocks],
        }

    # ---- band fitting -------------------------------------------------------
    def _floor(self, role: str, hard: bool) -> int:
        frac = HARD_MIN_FRAC if hard else MIN_FRAC.get(role, 0.022)
        return max(12, round(frac * self.unit))

    def _measure_role(self, t: Dict, width: int, language: str, mood: str,
                      scale: float, hard: bool,
                      target: Optional[int] = None) -> Dict:
        role = t.get("role", "body")
        text = str(t["text"]).strip()
        floor = self._floor(role, hard)
        if target is None:
            target = int(role_size_px(role, self.unit) * scale)
        target = max(floor, target)
        font_file, covered = self.reg.pick(text, language, role, mood)
        if not covered:
            self.warnings.append(f"{text!r}: no full-coverage font")
        ls = ROLE_LETTER_SPACING.get(role, 0.0)
        # height is solved per band, so only the width constrains a block
        fit = measure.fit_block(self.reg, text, font_file, language,
                                max_w=width, max_h=10 ** 9,
                                target_size=target, ls_frac=ls,
                                allow_multiline=True,
                                max_lines=self._max_lines(role, width),
                                min_size=floor)
        fit.update({"role": role, "text": text, "font": font_file,
                    "letter_spacing": ls, "_src": t, "_w": width})
        return fit

    def _max_lines(self, role: str, width: int) -> int:
        """Headlines get a 3rd line in narrow columns / thumbnails, where two
        lines would force a tiny size (designers stack big words)."""
        if role != "headline":
            return MAX_LINES.get(role, 2)
        comp = getattr(self, "_comp", None) or {}
        if comp.get("headline_lines"):
            return int(comp["headline_lines"])
        lim = self._limits(comp) if comp else None
        if lim and width < 0.72 * lim["w"]:
            return 3
        return MAX_LINES["headline"]

    def _gap_after(self, f: Dict) -> float:
        """Vertical gap after a block; templates with a divider / accent bar
        under the headline reserve more room there."""
        hg = (getattr(self, "_comp", None) or {}).get("headline_gap")
        frac = hg if (hg and f["role"] == "headline") else GAP_FRACTION
        return frac * f["size_px"]

    def _band_height(self, fits: List[Dict]) -> float:
        return (sum(f["h"] for f in fits) +
                sum(self._gap_after(f) for f in fits[:-1]))

    def _enforce_hierarchy(self, fits: List[List[Dict]], language: str,
                           mood: str, hard: bool) -> None:
        """Keep headline >= subheadline >= body >= detail >= note across BOTH
        bands, and emphasis <= headline.

        A long line may be shrunk on its own to fit the width (e.g. a long
        headline in a narrow column); this caps the less important roles so
        they never end up larger than a more important one.
        """
        def members(role):
            return [(bi, i) for bi, band in enumerate(fits)
                    for i, f in enumerate(band) if f["role"] == role]

        def cap_all(mem, cap):
            for bi, i in mem:
                f = fits[bi][i]
                if f["size_px"] > cap:
                    fits[bi][i] = self._measure_role(
                        f["_src"], f["_w"], language, mood, 1.0, hard,
                        target=cap)

        running = None
        head = None
        for role in HIERARCHY:
            mem = members(role)
            if not mem:
                continue
            if running is not None:
                cap_all(mem, int(running * HIERARCHY_RATIO))
            low = min(fits[bi][i]["size_px"] for bi, i in mem)
            running = low if running is None else min(running, low)
            if role == "headline":
                head = low
        if head is not None:
            cap_all(members("emphasis"), int(head * HIERARCHY_RATIO))

    def _limits(self, comp: Dict) -> Dict:
        c = compositions._content(self.canvas["w"], self.canvas["h"])
        ins = int(comp.get("inset_px", 0))
        if ins:  # template frame: keep text inside it
            c = {"x": c["x"] + ins, "y": c["y"] + ins,
                 "w": c["w"] - 2 * ins, "h": c["h"] - 2 * ins}
        if comp.get("border"):  # stay inside the frame
            inset = int(min(self.canvas["w"], self.canvas["h"]) * 0.06)
            c = {"x": c["x"] + inset, "y": c["y"] + inset,
                 "w": c["w"] - 2 * inset, "h": c["h"] - 2 * inset}
        return c

    @staticmethod
    def _room(box: Dict, others: List[Dict], limits: Dict,
              gap: int) -> tuple:
        """Vertical interval a band may occupy without touching `others`."""
        lo, hi = limits["y"], limits["y"] + limits["h"]
        bx0, bx1 = box["x"], box["x"] + box["w"]
        bcy = box["y"] + box["h"] / 2.0
        for o in others:
            if o["x"] + o["w"] <= bx0 or bx1 <= o["x"]:
                continue  # no horizontal overlap -> no conflict
            if o["y"] + o["h"] / 2.0 < bcy:
                lo = max(lo, o["y"] + o["h"] + gap)
            else:
                hi = min(hi, o["y"] - gap)
        return lo, hi

    @staticmethod
    def _resize(box: Dict, need: float, lo: float, hi: float,
                valign: str) -> Dict:
        y, h = box["y"], box["h"]
        if valign == "top":
            ny = y
        elif valign == "bottom":
            ny = y + h - need
        else:
            ny = y + (h - need) / 2.0
        ny = min(max(ny, lo), hi - need)
        return dict(box, y=int(round(ny)), h=int(round(need)) + 1)

    def _solve_boxes(self, comp: Dict, fits: List[List[Dict]],
                     obstacles: List[Dict]) -> Optional[List[Dict]]:
        """Final band boxes, or None if the text does not fit.

        Bands first shrink to their content (freeing room), then any band that
        needs more height grows into the free space next to it (never into the
        other band, the badge or outside the safe area).
        """
        bands = [comp["primary"], comp["secondary"]]
        limits = self._limits(comp)
        gap = compositions._gap(self.canvas["w"], self.canvas["h"], 20)
        panel = comp.get("panel") or {}
        if panel.get("target") in ("primary", "secondary"):
            # a plate behind ONE band: keep the other band out of its padding
            gap += 2 * round(0.035 * min(self.canvas["w"], self.canvas["h"]))
        boxes = [dict(b["box"]) for b in bands]
        needs = []
        for i, band in enumerate(bands):
            if any(f.get("overflow") for f in fits[i]):
                return None  # a line is too long for the width even at floor
            needs.append(self._band_height(fits[i]) if fits[i] else 0.0)
        for i, band in enumerate(bands):       # pass 1: shrink to content
            b = boxes[i]
            if fits[i] and needs[i] <= b["h"]:
                boxes[i] = self._resize(b, needs[i], b["y"], b["y"] + b["h"],
                                        band["valign"])
        for i, band in enumerate(bands):       # pass 2: grow where needed
            if not fits[i] or needs[i] <= boxes[i]["h"]:
                continue
            # push a neighbouring band away just enough (within its own free
            # room) before growing, e.g. details below a tall headline block
            for j in range(len(bands)):
                if j == i or not fits[j]:
                    continue
                bj, bi_ = boxes[j], boxes[i]
                if bj["x"] + bj["w"] <= bi_["x"] or \
                        bi_["x"] + bi_["w"] <= bj["x"]:
                    continue
                rest = [boxes[k] for k in range(len(bands))
                        if k not in (i, j) and fits[k]] + obstacles
                lo_j, hi_j = self._room(bj, rest, limits, gap)
                if bj["y"] + bj["h"] / 2.0 > bi_["y"] + bi_["h"] / 2.0:
                    want = bi_["y"] + needs[i] + gap          # j below i
                    ny = min(max(bj["y"], want), hi_j - bj["h"])
                else:
                    want = bi_["y"] + bi_["h"] - needs[i] - gap  # j above i
                    ny = max(min(bj["y"], want - bj["h"]), lo_j)
                boxes[j] = dict(bj, y=int(round(ny)))
            others = [boxes[j] for j in range(len(bands))
                      if j != i and fits[j]] + obstacles
            lo, hi = self._room(boxes[i], others, limits, gap)
            if needs[i] > hi - lo:
                return None
            boxes[i] = self._resize(boxes[i], needs[i], lo, hi,
                                    band["valign"])
        return boxes

    def _least_important(self, items: List[List[Dict]]):
        """(band, index) of the line to drop first; never the headline."""
        order = {r: i for i, r in enumerate(ROLES)}
        cands = [(bi, i) for bi, band in enumerate(items)
                 for i, t in enumerate(band) if t.get("role") != "headline"]
        if not cands:
            return None
        return max(cands, key=lambda c: (
            int(items[c[0]][c[1]].get("priority", 5) or 5),
            order.get(items[c[0]][c[1]].get("role"), 99)))

    def _fit_bands(self, comp: Dict, band_items: List[List[Dict]],
                   obstacles: List[Dict], language: str, mood: str,
                   band_bgs) -> List[Dict]:
        """Measure + place both bands with ONE shared scale.

        Order of fallbacks: shared scale down to the readable floors -> hard
        floors -> drop the least important line (warned). Never overlaps and
        never truncates text.
        """
        bands = [comp["primary"], comp["secondary"]]
        order = {r: i for i, r in enumerate(ROLES)}
        items = [sorted(its, key=lambda t: order.get(t.get("role"), 99))
                 for its in band_items]
        hard = False
        while True:
            scale = 1.0
            for _ in range(FIT_ITERS):
                fits = [[self._measure_role(t, band["box"]["w"], language,
                                            mood, scale, hard) for t in its]
                        for band, its in zip(bands, items)]
                self._enforce_hierarchy(fits, language, mood, hard)
                boxes = self._solve_boxes(comp, fits, obstacles)
                if boxes is not None:
                    if hard:
                        self.warnings.append(
                            "too much text: sizes went below the readable "
                            "floor")
                    return self._place(bands, fits, boxes, band_bgs)
                scale *= SCALE_STEP
            if not hard:
                hard = True
                continue
            victim = self._least_important(items)
            if victim is None:  # headline alone still does not fit: place it
                self.warnings.append("headline does not fit the canvas")
                boxes = [dict(b["box"]) for b in bands]
                return self._place(bands, fits, boxes, band_bgs)
            dropped = items[victim[0]].pop(victim[1])
            self.warnings.append(
                f"dropped {dropped.get('text')!r} ({dropped.get('role')}): "
                "not enough room")

    def _place(self, bands: List[Dict], fits: List[List[Dict]],
               boxes: List[Dict], band_bgs) -> List[Dict]:
        blocks = []
        for bi, (band, band_fits, box) in enumerate(zip(bands, fits, boxes)):
            bg_rgb = band_bgs[bi]
            if not band_fits:
                continue
            total = self._band_height(band_fits)
            valign = band["valign"]
            if valign == "top":
                cursor = box["y"]
            elif valign == "bottom":
                cursor = box["y"] + box["h"] - total
            else:
                cursor = box["y"] + (box["h"] - total) / 2.0
            for f in band_fits:
                blk = self._finalize(f, box["x"], cursor, box["w"], f["h"],
                                     band["align"], bg_rgb)
                blk["band"] = bi
                blocks.append(blk)
                cursor += f["h"] + self._gap_after(f)
        return blocks

    def _finalize(self, f: Dict, bx, by, bw, bh, align, bg_rgb) -> Dict:
        role = f["role"]
        color, intent = self._role_color(role, bg_rgb, f["size_px"])
        box = {"x": int(bx), "y": int(round(by)),
               "w": int(bw), "h": int(round(bh))}
        cx = box["x"] + box["w"] // 2
        cy = box["y"] + box["h"] // 2
        return {
            "role": role, "text": f["text"], "font": f["font"],
            "size_px": f["size_px"], "cx": cx, "cy": cy, "align": align,
            "color": rgb_to_hex(color),
            # the color the design WANTS (LLM hue); ZoneCheck re-fits it to the
            # real image. None = no intent -> neutral / palette-derived accent.
            "intent_color": rgb_to_hex(intent) if intent else None,
            "letter_spacing": f["letter_spacing"],
            "shadow": role in ("headline", "subheadline", "emphasis"),
            "outline_width": 0, "outline_color": "#000000",
            "box": box, "lines": f["lines"],
            "line_height": f["line_height"],
            "text_w": int(round(f.get("w", box["w"]))),
            # ink above the first / below the last line box (stacked accents)
            "pad_top": f.get("pad_top", 0),
            "pad_bottom": f.get("pad_bottom", 0),
        }

    def _decor_colors(self, blocks: List[Dict], bg_rgb) -> Dict:
        """Provisional decor colors: the headline's color as "text", the
        first accent-role color on the image as "accent"."""
        on_img = [b for b in blocks if not b.get("on_shape")
                  and not b.get("on_panel")]
        head = next((b for b in on_img if b["role"] == "headline"), None)
        acc = next((b for b in on_img if b["role"] in ACCENT_ROLES), None)
        text = hex_to_rgb(head["color"]) if head else neutral_on(bg_rgb)
        if acc:
            accent = hex_to_rgb(acc["color"])
        else:
            accent = fit_contrast(self.accent_intent or DEFAULT_ACCENT,
                                  bg_rgb, 3.0)
        return {"text": text, "accent": accent}

    @staticmethod
    def _intent(value):
        h = parse_hex(value)
        return hex_to_rgb(h) if h else None

    def _role_color(self, role: str, bg_rgb, size_px: int):
        """(provisional color, intent) against the planned background_color.

        Accent roles use the LLM accent_color, other roles its text_color;
        either keeps its hue and only has its lightness fitted for contrast.
        Without an intent: neutral text, and a default warm gold for accents
        (ZoneCheck may swap that for a color taken from the real image).
        """
        short = min(self.canvas["w"], self.canvas["h"])
        need = required_contrast(size_px, short)
        intent = self.accent_intent if role in ACCENT_ROLES \
            else self.text_intent
        if role in ACCENT_ROLES and intent is None:
            intent = self.text_intent  # at least the main text hue
        if intent is not None:
            return fit_intent(intent, bg_rgb, need), intent
        if role in ACCENT_ROLES:
            return fit_contrast(DEFAULT_ACCENT, bg_rgb, need), None
        return neutral_on(bg_rgb), None

    # ---- badge + border -----------------------------------------------------
    # text area inside each badge shape, as (max_w, max_h) fractions of r
    # (circle / star / diamond) or of the pill's (w, h)
    BADGE_TEXT_AREA = {"circle": (1.6, 1.2), "star": (1.3, 0.95),
                       "diamond": (1.0, 0.75), "pill": (0.82, 0.66)}

    def _make_badge(self, emph: Dict, comp: Dict, language, mood, bg_rgb):
        """Emphasis text on a solid badge: circle | star | diamond | pill."""
        cw, ch = self.canvas["w"], self.canvas["h"]
        shape_kind = comp.get("badge_shape", "circle")
        if shape_kind not in self.BADGE_TEXT_AREA:
            shape_kind = "circle"
        bb = comp.get("badge_box") or {
            "x": int(cw * 0.57), "y": int(ch * 0.15),
            "w": int(min(cw, ch) * 0.30), "h": int(min(cw, ch) * 0.30)}
        cx, cy = bb["x"] + bb["w"] // 2, bb["y"] + bb["h"] // 2
        if shape_kind == "pill":
            pw = int(bb["w"] * 0.94)
            ph = int(min(bb["h"], pw * 0.46))
            box = {"x": cx - pw // 2, "y": cy - ph // 2, "w": pw, "h": ph}
            fw, fh = self.BADGE_TEXT_AREA["pill"]
            max_w, max_h, target = pw * fw, ph * fh, int(ph * 0.55)
        else:
            r = int(min(bb["w"], bb["h"]) * 0.5)
            box = {"x": cx - r, "y": cy - r, "w": 2 * r, "h": 2 * r}
            fw, fh = self.BADGE_TEXT_AREA[shape_kind]
            max_w, max_h, target = r * fw, r * fh, int(r * 0.9 * fh / 1.2)
        text = str(emph["text"]).strip()
        font_file, _ = self.reg.pick(text, language, "emphasis", mood)
        fit = measure.fit_block(self.reg, text, font_file, language,
                                max_w=max_w, max_h=max_h,
                                target_size=max(12, target), ls_frac=0.0,
                                allow_multiline=True, max_lines=2)
        # badge = the accent color (LLM intent), fitted so white (preferred)
        # or near-black text reaches 4.5:1 on it
        badge_fill, txt_color = decor.fill_for_text(
            self.accent_intent or DEFAULT_BADGE)
        block = {
            "role": "emphasis", "text": text, "font": font_file,
            "size_px": fit["size_px"], "cx": cx, "cy": cy, "align": "center",
            "color": rgb_to_hex(txt_color), "letter_spacing": 0.0,
            "shadow": False, "outline_width": 0, "outline_color": "#000000",
            "box": box, "lines": fit["lines"], "line_height": fit["line_height"],
            "text_w": int(round(fit.get("w", box["w"]))),
            "pad_top": fit.get("pad_top", 0),
            "pad_bottom": fit.get("pad_bottom", 0),
            "valign": "middle",
            # sits on the badge shape, not on the image: ZoneCheck must not
            # recolor it against the background
            "on_shape": True,
            "intent_color": None,
        }
        if shape_kind == "pill":
            shape = dict(box, type="badge", shape="pill",
                         fill=rgb_to_hex(badge_fill))
        else:
            shape = {"type": "badge", "shape": shape_kind, "cx": cx,
                     "cy": cy, "r": box["w"] // 2,
                     "fill": rgb_to_hex(badge_fill)}
        return block, shape

    def _border_shape(self, cw, ch, bg_rgb):
        inset = int(min(cw, ch) * 0.04)
        stroke = (245, 245, 245) if rel_luminance(bg_rgb) < 0.5 \
            else (25, 25, 25)
        return {"type": "border", "x": inset, "y": inset,
                "w": cw - 2 * inset, "h": ch - 2 * inset,
                "stroke": rgb_to_hex(stroke), "width": max(2, int(cw * 0.004))}
