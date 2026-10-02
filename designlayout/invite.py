# -*- coding: utf-8 -*-
"""Invitation stack: stationery-style typography, drawn in code.

Wedding / event cards are not "a headline + a detail band": they are a centered
COLUMN of typed components, each with its own typographic voice --

    kicker    small spaced capitals ("TOGETHER WITH THEIR FAMILIES")
    names     big script names, a small script connector ("and" / "&")
              flanked by ornaments
    invite    small spaced capitals ("INVITE YOU TO THEIR WEDDING")
    date      MONTH | big DAY | YEAR between thin rules, weekday above,
              time below
    venue     spaced capitals name + smaller address
    note      a soft serif line ("Reception to follow")
    divider   thin line with a diamond / dot / leaf ornament
    icon      heart / sprig

laid out on a paper card (arch, rounded or ticket) over the AI background.
Everything is drawn here, so spelling and accents are always exact; fonts are
picked from the registry by personality (script + elegant serif).
"""
from __future__ import annotations

import colorsys
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .colors import contrast_ratio, fit_contrast, hex_to_rgb, rgb_to_hex
from .fonts import FontRegistry
from .text_fx import salient_colors, _hls, _from_hls

SS = 2                       # supersampling for shapes / text
CARD_KINDS = ("arch", "rounded", "ticket")

# size (fraction of the content width) and letter spacing (fraction of size)
STYLE = {
    "kicker": (0.034, 0.18, "caps"),
    "invite": (0.031, 0.14, "caps"),
    "name": (0.125, 0.0, "script"),
    "connector": (0.060, 0.0, "script"),
    "month": (0.036, 0.16, "caps"),
    "day": (0.085, 0.02, "serif"),
    "weekday": (0.028, 0.18, "caps"),
    "time": (0.028, 0.14, "caps"),
    "venue": (0.040, 0.12, "caps"),
    "address": (0.027, 0.12, "caps"),
    "note": (0.034, 0.02, "serif"),
}
GAP = 0.045                  # vertical rhythm, fraction of the content width
FILL = 0.88                  # share of the card height the column may use
MAX_SCALE = 1.45             # never grow type beyond this x the base sizes


# ---- fonts --------------------------------------------------------------------
def _pick(reg: FontRegistry, language: str, group: str, want: set, text: str,
          rng: random.Random, avoid: set = frozenset()) -> str:
    cands = []
    for e in reg.candidates(language, group):
        f = e["file"]
        if f in reg.excluded or not reg.covers(f, text):
            continue
        tags = reg.tags_of(f)
        if want & tags and not tags & avoid:
            cands.append(f)
    if not cands:
        cands = [e["file"] for e in reg.candidates(language, group)
                 if reg.covers(e["file"], text)]
    return rng.choice(sorted(cands)) if cands else ""


def choose_fonts(reg: FontRegistry, comps: List[Dict], language: str,
                 seed: int) -> Dict[str, str]:
    rng = random.Random(seed * 31 + 7)
    every = " ".join(_texts(comps))
    names = " ".join(str(c.get(k, "")) for c in comps if c["type"] == "names"
                     for k in ("first", "second", "connector"))
    script = _pick(reg, language, "script", {"elegant"}, names or "Aa", rng,
                   avoid={"bold", "playful", "thin"}) or \
        _pick(reg, language, "script", {"elegant"}, names or "Aa", rng,
              avoid={"bold", "playful"}) or \
        _pick(reg, language, "script", {"elegant", "handwritten"}, names or "Aa", rng)
    serif = _pick(reg, language, "serif", {"elegant"}, every.upper(), rng,
                  avoid={"bold"}) or \
        _pick(reg, language, "serif", {"elegant", "clean"}, every.upper(), rng) or \
        _pick(reg, language, "sans", {"clean"}, every.upper(), rng)
    # last resort: any face that renders every character (never an empty name)
    if not script:
        script = _any_font(reg, language, names or "Aa", ("script", "decorative", "serif"))
    if not serif:
        serif = _any_font(reg, language, every.upper(), ("serif", "sans", "other", "display"))
    return {"script": script, "caps": serif, "serif": serif}


def _any_font(reg: FontRegistry, language: str, text: str, groups) -> str:
    for g in groups:
        for e in reg.candidates(language, g):
            if e["file"] not in reg.excluded and reg.covers(e["file"], text):
                return e["file"]
    for e in reg.candidates(language):
        if reg.covers(e["file"], text):
            return e["file"]
    return reg.candidates(language)[0]["file"]


def _texts(comps):
    for c in comps:
        for k in ("text", "first", "second", "connector", "weekday", "month",
                  "day", "year", "time", "name", "address"):
            if c.get(k):
                yield str(c[k])


# ---- colors -------------------------------------------------------------------
def choose_colors(img: Image.Image, paper: Tuple[int, int, int]) -> Dict[str, str]:
    """Ink = a deep shade of the picture's own accent hue (plum on lilac, olive
    on sage...), metallic accent = gold leaning to the picture's warmth."""
    sal = salient_colors(img)
    hue = _hls(sal[0])[0] if sal else 0.08
    ink = _from_hls(hue, 0.26, 0.30)
    if contrast_ratio(hex_to_rgb(ink), paper) < 6:
        ink = _from_hls(hue, 0.18, 0.30)
    gold = "#A8844E"
    for c in sal:
        h, l, s = _hls(c)
        if 0.06 <= h <= 0.16 and s > 0.35:          # a real gold / ochre in it
            gold = _from_hls(h, 0.45, min(0.6, s))
            break
    return {"ink": ink, "accent": gold}


# ---- drawing primitives (SS-scaled) --------------------------------------------
class Canvas:
    def __init__(self, w: int, h: int, reg: FontRegistry):
        self.w, self.h, self.reg = w, h, reg
        self.img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self._fonts: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def font(self, file: str, size: float):
        key = (file, max(4, int(size * SS)))
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(self.reg.find_path(file), key[1])
        return self._fonts[key]


def _spaced_width(font, text: str, ls: float) -> float:
    if ls <= 0:
        return font.getlength(text)
    return sum(font.getlength(ch) for ch in text) + ls * (len(text) - 1)


def _draw_spaced(d, font, x: float, y: float, text: str, ls: float, fill):
    if ls <= 0:
        d.text((x, y), text, font=font, fill=fill, anchor="ls")
        return
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill, anchor="ls")
        x += font.getlength(ch) + ls


def _wrap(font, text: str, ls: float, max_w: float) -> List[str]:
    if "\n" in text:                 # breaks chosen by the art director / LLM
        flat = " ".join(text.split())
        if _spaced_width(font, flat, ls) <= max_w:
            return [flat]              # it fits: a hint is a preference, not a must
        out = []
        for seg in text.split("\n"):
            if seg.strip():
                out += _wrap(font, " ".join(seg.split()), ls, max_w)
        return out or [text]
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = wd if not cur else cur + " " + wd
        if _spaced_width(font, t, ls) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    if 2 <= len(lines) <= 4 and len(words) <= 16:
        bal = _balanced(font, words, len(lines), ls, max_w)
        if bal:
            return bal
    return lines or [text]


def _balanced(font, words, k, ls, max_w):
    """Same number of lines, but even: minimise the longest line and never
    leave one short word alone on a line (a name like "Hoa Sen" stays whole)."""
    from itertools import combinations
    n = len(words)
    if n <= k:
        return None
    cache = {}

    def w(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = _spaced_width(font, " ".join(words[i:j]), ls)
        return cache[(i, j)]
    best, best_score = None, None
    for cuts in combinations(range(1, n), k - 1):
        b = (0,) + cuts + (n,)
        widths = [w(b[i], b[i + 1]) for i in range(k)]
        if max(widths) > max_w:
            continue
        lone = sum(1 for i in range(k) if b[i + 1] - b[i] == 1)
        score = max(widths) + 0.25 * max_w * lone
        if best_score is None or score < best_score:
            best, best_score = b, score
    if best is None:
        return None
    return [" ".join(words[best[i]:best[i + 1]]) for i in range(k)]


# ---- components -----------------------------------------------------------------
_SS_FONTS: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _ss_font(file: str, size: float) -> ImageFont.FreeTypeFont:
    """Font at the supersampled size Canvas.font uses (same rounding)."""
    key = (file, max(4, int(size * SS)))
    if key not in _SS_FONTS:
        _SS_FONTS[key] = ImageFont.truetype(_REG.find_path(file), key[1])
    return _SS_FONTS[key]


class Comp:
    """measure(scale) -> height (px); draw(cv, cx, top, scale, colors)."""

    def __init__(self, c: Dict, fonts: Dict[str, str], width: float,
                 st: Optional[Dict] = None, names_cv=None):
        self.c, self.fonts, self.width = c, fonts, width
        self.wrap = width             # line-break width (narrower where decorations intrude)
        self.k = 1.0                  # this line only: smaller when it still touches a decoration
        self.st = st or STYLE
        self.names_cv = names_cv      # names drawn here (white) when they get an effect
        self.align = "center"         # "center": X is the column axis; "left": X is the left edge

    def _x0(self, X: float, w: float) -> float:
        """Left x of a line of width w drawn at X."""
        return X - w / 2 if self.align == "center" else X

    # -- date styles: "block" (MONTH | DAY | YEAR, the default), "line" (one
    #    caps line + time), "hero" (a big day number between month/year and
    #    weekday/time)
    def _date_style(self) -> str:
        st = self.c.get("date_style", "block")
        return "line" if self.align == "left" and st == "block" else st

    def _date_texts(self):
        c = self.c
        wk, day, mon, yr = (str(c.get(k, "") or "").strip() for k in ("weekday", "day", "month", "year"))
        vi = mon.lower().startswith("tháng")
        md = f"{day} {mon}" if vi else f"{mon} {day}"
        tm = str(c.get("time", "") or "").strip()
        return {"line": " · ".join(p for p in (wk, md.strip(), yr) if p).upper(), "time": tm.upper(),
                "top": " · ".join(p for p in (mon, yr) if p).upper(), "day": day,
                "bottom": " · ".join(p for p in (wk, tm) if p).upper()}

    def _date_fonts(self, s):
        u = self.width * s * self.k
        mk = self.st["month"]
        line = (self.fonts[mk[2]], mk[0] * 1.3 * u, mk[1] * mk[0] * 1.3 * u)
        top = (self.fonts[mk[2]], mk[0] * u, mk[1] * mk[0] * u)
        wk = self.st["weekday"]
        bottom = (self.fonts[wk[2]], wk[0] * u, wk[1] * wk[0] * u)
        tm = self.st["time"]
        time = (self.fonts[tm[2]], tm[0] * u, tm[1] * tm[0] * u)
        dk = self.st["day"]
        hero = (self.fonts[dk[2]], dk[0] * 2.1 * u, 0.0)
        return {"line": line, "top": top, "bottom": bottom, "time": time, "hero": hero}

    def _date_line_rows(self, s):
        t, f = self._date_texts(), self._date_fonts(s)
        file, px, ls = f["line"]
        lines = _wrap(_ss_font(file, px), t["line"], ls * SS, self.wrap * SS)
        return lines, f, t

    def _line(self, style, text, s):
        size, lsf, kind = self.st[style]
        px = size * self.width * s * self.k
        if kind == "caps":
            text = text.upper()
        return self.fonts[kind], px, lsf * px, text

    def _text_block(self, style, text, s, cv=None):
        file, px, ls, text = self._line(style, text, s)
        # measure and draw wrap at the SAME (supersampled) size, so they
        # always agree on the number of lines
        f = cv.font(file, px) if cv else _ss_font(file, px)
        k = SS
        lines = _wrap(f, text, ls * k, self.wrap * k)
        lh = px * (1.22 if self.st[style][2] == "script" else 1.45)
        return f, lines, ls * k, lh

    def _name_font(self, part, s):
        px = self.st["name"][0] * self.width * s * self.k
        f = ImageFont.truetype(_REG.find_path(self.fonts["script"]), max(4, int(px)))
        w = f.getlength(part)
        if w > self.wrap:                         # long name: shrink to fit
            px *= self.wrap / w
            f = ImageFont.truetype(_REG.find_path(self.fonts["script"]), max(4, int(px)))
        return px, f

    def _name_h(self, part, s):
        """Real ink height of a script name (swashes go far above / below)."""
        px, f = self._name_font(part, s)
        l, t, r, b = f.getbbox(part, anchor="ls")
        return (b - t) * 1.08

    def need_w(self, s: float) -> float:
        """Width (px) this component needs at scale s; wrapping lines and
        names (they shrink to fit) need none."""
        t, u = self.c["type"], self.width * s * self.k
        if t == "date" and self._date_style() == "line":
            return 0.0                                  # it wraps
        if t == "date" and self._date_style() == "hero":
            tx, fs = self._date_texts(), self._date_fonts(s)
            w = [ImageFont.truetype(_REG.find_path(fs["hero"][0]), max(4, int(fs["hero"][1]))).getlength(tx["day"])]
            for key in ("top", "bottom"):
                file, px, ls = fs[key]
                w.append(_spaced_width(ImageFont.truetype(_REG.find_path(file), max(4, int(px))), tx[key], ls))
            return max(w)
        if t == "date":
            def f(st):
                return ImageFont.truetype(_REG.find_path(self.fonts[self.st[st][2]]),
                                          max(4, int(self.st[st][0] * u)))
            mf, df = f("month"), f("day")
            ls = self.st["month"][1] * self.st["month"][0] * u
            words = [str(self.c.get("month", "")).upper(), str(self.c.get("year", "")).upper()]
            side = max(_spaced_width(mf, w, ls) for w in words)
            return df.getlength(str(self.c.get("day", ""))) + 2 * (u * 0.06 * 2.2 + side)
        if t == "venue":
            f = ImageFont.truetype(_REG.find_path(self.fonts["caps"]),
                                   max(4, int(self.st["venue"][0] * u)))
            name = str(self.c.get("name", "")).upper()
            longest = max(name.split(), key=len) if name else ""
            return _spaced_width(f, longest, 0) * 1.02   # name may wrap at spaces
        return 0.0

    def line_w(self, s: float) -> float:
        t, u = self.c["type"], self.width * s * self.k
        if t == "text":
            f, lines, ls, _ = self._text_block(self.c.get("style", "invite"), self.c["text"], s)
            return max(_spaced_width(f, ln, ls) for ln in lines) / SS
        if t == "names":
            return max(self._name_font(p, s)[1].getlength(p)
                       for p in (self.c.get("first"), self.c.get("second")) if p)
        if t == "venue":
            out = 0.0
            for key, txt in (("venue", self.c.get("name")), ("address", self.c.get("address"))):
                if not txt:
                    continue
                f = _ss_font(self.fonts["caps"], self.st[key][0] * u)
                ls = self.st[key][1] * self.st[key][0] * u * SS
                lines = _wrap(f, str(txt).upper(), ls, self.wrap * SS)
                out = max([out] + [_spaced_width(f, ln, ls) / SS for ln in lines])
            return out
        if t == "date" and self._date_style() == "line":
            lines, fs, _ = self._date_line_rows(s)
            file, px, ls = fs["line"]
            f = _ss_font(file, px)
            return max(_spaced_width(f, ln, ls * SS) for ln in lines) / SS
        if t == "date":
            return self.need_w(s)
        return 0.0

    def em(self, s: float) -> Optional[float]:
        """Type size (px) that sets the spacing around this component; None
        for a divider (it takes the spacing of its neighbours)."""
        t, u = self.c["type"], self.width * s * self.k
        key = {"text": self.c.get("style", "invite"), "names": "connector",
               "date": "weekday", "venue": "venue", "icon": "kicker"}.get(t)
        return self.st[key][0] * u if key else None

    def measure(self, s: float) -> float:
        t = self.c["type"]
        u = self.width * s * self.k
        if t == "text":
            _, lines, _, lh = self._text_block(self.c.get("style", "invite"), self.c["text"], s)
            return lh * len(lines)
        if t == "names":
            parts = [p for p in (self.c.get("first"), self.c.get("second")) if p]
            con = self.st["connector"][0] * u * 1.7 if len(parts) == 2 else 0
            return sum(self._name_h(p, s) for p in parts) + con
        if t == "date" and self._date_style() == "line":
            lines, fs, tx = self._date_line_rows(s)
            return fs["line"][1] * 1.55 * len(lines) + (fs["time"][1] * 1.6 if tx["time"] else 0)
        if t == "date" and self._date_style() == "hero":
            fs, tx = self._date_fonts(s), self._date_texts()
            return (fs["top"][1] * 1.7 + fs["hero"][1] * 1.08 +
                    (fs["bottom"][1] * 1.8 if tx["bottom"] else 0))
        if t == "date":
            return (self.st["weekday"][0] + self.st["day"][0] * 1.25 + self.st["time"][0]) * u * 1.55
        if t == "venue":
            f = _ss_font(self.fonts["caps"], self.st["venue"][0] * u)
            ls = self.st["venue"][1] * self.st["venue"][0] * u * SS
            nl = len(_wrap(f, str(self.c.get("name", "")).upper(), ls, self.wrap * SS))
            h = self.st["venue"][0] * u * 1.5 * nl
            if self.c.get("address"):
                af = _ss_font(self.fonts["caps"], self.st["address"][0] * u)
                als = self.st["address"][1] * self.st["address"][0] * u * SS
                na = len(_wrap(af, str(self.c["address"]).upper(), als, self.wrap * SS))
                h += self.st["address"][0] * u * 1.6 * na
            return h
        if t == "divider":
            return 0.035 * u
        if t == "icon":
            return 0.05 * u
        return 0

    # -- draw
    def draw(self, cv: Canvas, cx: float, top: float, s: float, col: Dict) -> None:
        t, u, d = self.c["type"], self.width * s * self.k, cv.d
        ink, acc = hex_to_rgb(col["ink"]) + (255,), hex_to_rgb(col["accent"]) + (255,)
        X, Y = cx * SS, top * SS
        if t == "text":
            style = self.c.get("style", "invite")
            f, lines, ls, lh = self._text_block(style, self.c["text"], s, cv)
            fill = acc if self.c.get("accent") else ink
            for i, ln in enumerate(lines):
                w = _spaced_width(f, ln, ls)
                _draw_spaced(d, f, self._x0(X, w), Y + (i + 0.82) * lh * SS, ln, ls, fill)
        elif t == "names":
            if self.names_cv is not None:
                d = self.names_cv.d
                ink = (255, 255, 255, 255)
            con = self.c.get("connector", "and")
            cf = cv.font(self.fonts["script"], self.st["connector"][0] * u)
            c_h = self.st["connector"][0] * u * 1.7 * SS
            y = Y
            seq = (self.c["first"], None, self.c["second"]) if self.c.get("second") \
                else (self.c["first"],)          # one name (birthday, a person)
            for part in seq:
                if part is None:
                    my = y + c_h / 2
                    # center the connector's INK on the ornament line
                    l, t_, r, b = cf.getbbox(con, anchor="ls")
                    span = u * 0.26 * SS
                    if self.align == "left":          # "and" at the edge, one sprig after it
                        cv.d.text((X - l, my - (t_ + b) / 2), con, font=cf, fill=acc, anchor="ls")
                        _laurel(cv.d, X + (r - l) + u * 0.035 * SS, my, 1, span, acc, u)
                        y += c_h
                        continue
                    cv.d.text((X - (l + r) / 2, my - (t_ + b) / 2), con, font=cf,
                              fill=acc, anchor="ls")
                    for sgn in (-1, 1):
                        x0 = X + sgn * ((r - l) / 2 + u * 0.035 * SS)
                        _laurel(cv.d, x0, my, sgn, span, acc, u)
                    y += c_h
                    continue
                px, _ = self._name_font(part, s)
                nf = cv.font(self.fonts["script"], px)
                l, t_, r, b = nf.getbbox(part, anchor="ls")
                h = (b - t_) * 1.08
                nx = X - (l + r) / 2 if self.align == "center" else X - l
                d.text((nx, y + h * 0.04 - t_), part, font=nf, fill=ink, anchor="ls")
                y += h
        elif t == "date" and self._date_style() == "line":
            lines, fs, tx = self._date_line_rows(s)
            file, px, ls = fs["line"]
            f = cv.font(file, px)
            y = Y
            for ln in lines:
                w = _spaced_width(f, ln, ls * SS)
                _draw_spaced(d, f, self._x0(X, w), y + px * SS * 1.1, ln, ls * SS, ink)
                y += px * 1.55 * SS
            if tx["time"]:
                file, px, ls = fs["time"]
                f = cv.font(file, px)
                w = _spaced_width(f, tx["time"], ls * SS)
                _draw_spaced(d, f, self._x0(X, w), y + px * SS * 1.2, tx["time"], ls * SS, acc)
        elif t == "date" and self._date_style() == "hero":
            fs, tx = self._date_fonts(s), self._date_texts()
            y = Y
            file, px, ls = fs["top"]
            f = cv.font(file, px)
            w = _spaced_width(f, tx["top"], ls * SS)
            _draw_spaced(d, f, self._x0(X, w), y + px * SS * 1.15, tx["top"], ls * SS, acc)
            y += px * 1.7 * SS
            file, px, _ = fs["hero"]
            f = cv.font(file, px)
            l, t_, r, b = f.getbbox(tx["day"], anchor="ls")
            hx = X - (l + r) / 2 if self.align == "center" else X - l
            d.text((hx, y - t_), tx["day"], font=f, fill=ink, anchor="ls")
            y += px * 1.08 * SS
            if tx["bottom"]:
                file, px, ls = fs["bottom"]
                f = cv.font(file, px)
                w = _spaced_width(f, tx["bottom"], ls * SS)
                _draw_spaced(d, f, self._x0(X, w), y + px * SS * 1.2, tx["bottom"], ls * SS, ink)
        elif t == "date":
            wk = self.c.get("weekday", "")
            ff = lambda st: cv.font(self.fonts[self.st[st][2]], self.st[st][0] * u)
            lsp = lambda st: self.st[st][1] * self.st[st][0] * u * SS
            y = Y
            if wk:
                f = ff("weekday"); w = _spaced_width(f, wk.upper(), lsp("weekday"))
                _draw_spaced(d, f, X - w / 2, y + self.st["weekday"][0] * u * SS * 1.0,
                             wk.upper(), lsp("weekday"), ink)
            y += self.st["weekday"][0] * u * SS * 1.55
            dayf = ff("day"); dh = self.st["day"][0] * u * SS
            day = str(self.c.get("day", ""))
            dw = dayf.getlength(day)
            base = y + dh * 1.0
            d.text((X - dw / 2, base), day, font=dayf, fill=ink, anchor="ls")
            gap = u * 0.06 * SS
            dl, dt, dr, db = dayf.getbbox(day, anchor="ls")
            for sgn in (-1, 1):
                rx = X + sgn * (dw / 2 + gap)
                d.line([(rx, base + dt - dh * 0.08), (rx, base + db + dh * 0.08)],
                       fill=acc, width=max(1, SS))
            mf = ff("month")
            for sgn, word in ((-1, self.c.get("month", "")), (1, self.c.get("year", ""))):
                word = str(word).upper()
                w = _spaced_width(mf, word, lsp("month"))
                rx = X + sgn * (dw / 2 + gap * 2.2)
                x0 = rx - w if sgn < 0 else rx
                _draw_spaced(d, mf, x0, base - dh * 0.28, word, lsp("month"), ink)
            y += dh * 1.25
            tm = self.c.get("time", "")
            if tm:
                f = ff("time"); w = _spaced_width(f, tm.upper(), lsp("time"))
                _draw_spaced(d, f, X - w / 2, y + self.st["time"][0] * u * SS * 1.1,
                             tm.upper(), lsp("time"), ink)
        elif t == "venue":
            f = cv.font(self.fonts["caps"], self.st["venue"][0] * u)
            ls = self.st["venue"][1] * self.st["venue"][0] * u * SS
            name = str(self.c.get("name", "")).upper()
            vh = self.st["venue"][0] * u * 1.5 * SS
            nlines = _wrap(f, name, ls, self.wrap * SS)
            for i, ln in enumerate(nlines):
                w = _spaced_width(f, ln, ls)
                _draw_spaced(d, f, self._x0(X, w), Y + vh * (0.72 + i), ln, ls, ink)
            Y += vh * (len(nlines) - 1)
            if self.c.get("address"):
                af = cv.font(self.fonts["caps"], self.st["address"][0] * u)
                als = self.st["address"][1] * self.st["address"][0] * u * SS
                lines = _wrap(af, str(self.c["address"]).upper(), als, self.wrap * SS)
                lh = self.st["address"][0] * u * 1.6 * SS
                for i, ln in enumerate(lines):
                    w = _spaced_width(af, ln, als)
                    _draw_spaced(d, af, self._x0(X, w), Y + vh + (i + 0.75) * lh, ln, als, ink)
        elif t == "divider":
            y = Y + 0.0175 * u * SS
            half = u * 0.17 * SS
            o = u * 0.009 * SS
            if self.align == "left":              # a short rule from the edge
                d.line([(X, y), (X + half * 1.4, y)], fill=acc, width=max(1, SS))
                return
            d.line([(X - half, y), (X - o * 2.2, y)], fill=acc, width=max(1, SS))
            d.line([(X + o * 2.2, y), (X + half, y)], fill=acc, width=max(1, SS))
            d.polygon([(X, y - o), (X + o, y), (X, y + o), (X - o, y)], fill=acc)
        elif t == "icon":
            r = 0.022 * u * SS
            cy = Y + 0.025 * u * SS
            ix = X if self.align == "center" else X + r
            if self.c.get("name", "heart") == "heart":
                _heart(d, ix, cy, r, acc)
            else:
                _sprig(d, ix, cy, r * 1.6, acc, u)


SIZE_GROUPS = {"names": ("name", "connector"), "small_caps": ("kicker", "invite"),
               "date": ("month", "day", "weekday", "time"), "venue": ("venue", "address"),
               "note": ("note",)}
EFFECTS = ("flat", "foil", "emboss", "shadow")


def style_table(sizes: Optional[Dict] = None, tracking: float = 1.0) -> Dict:
    """Base STYLE with per-group size multipliers and caps tracking."""
    st = dict(STYLE)
    for grp, keys in SIZE_GROUPS.items():
        m = float((sizes or {}).get(grp, 1.0))
        for k in keys:
            size, ls, kind = st[k]
            st[k] = (size * m, ls * (tracking if kind == "caps" else 1.0), kind)
    return st


def apply_name_effect(img: Image.Image, mask_rgba: Image.Image, effect: str,
                      ink: str, accent: str) -> Image.Image:
    """Names come as a white RGBA layer; give them the chosen finish."""
    W, H = img.size
    m = mask_rgba.split()[3].resize((W, H), Image.LANCZOS)
    A = np.asarray(m, np.float32) / 255
    out = img.convert("RGBA")

    def paint(base, col, alpha):
        arr = np.zeros((H, W, 4), np.float32)
        arr[..., :3] = col
        arr[..., 3] = np.clip(alpha, 0, 1) * 255
        return Image.alpha_composite(base, Image.fromarray(arr.astype(np.uint8), "RGBA"))

    ink_c = np.array(hex_to_rgb(ink), np.float32)
    if effect == "shadow":
        sh = np.asarray(m.filter(ImageFilter.GaussianBlur(max(2, H // 300))), np.float32) / 255
        sh = np.roll(sh, (max(1, H // 500), max(1, H // 700)), (0, 1))
        out = paint(out, np.zeros(3, np.float32), sh * 0.35)
        out = paint(out, ink_c, A)
    elif effect == "emboss":
        hi = np.roll(A, (-1, -1), (0, 1))
        lo = np.roll(A, (2, 2), (0, 1))
        out = paint(out, np.array([255, 255, 255], np.float32), np.clip(hi - A, 0, 1) * 0.9)
        out = paint(out, ink_c * 0.55, np.clip(lo - A, 0, 1) * 0.6)
        out = paint(out, ink_c, A)
    elif effect == "foil":
        h, l, s_ = _hls(accent)
        ys = np.where(A.max(1) > 0.1)[0]
        y0, y1 = (ys.min(), ys.max()) if len(ys) else (0, H)
        t = np.clip((np.arange(H) - y0) / max(1, y1 - y0), 0, 1)
        stops = [(0, _from_hls(h, min(.85, l + .25), s_)), (.42, accent),
                 (.55, _from_hls(h, max(.15, l - .18), s_)),
                 (1, _from_hls(h, min(.8, l + .12), s_))]
        cols = np.stack([np.array(hex_to_rgb(c), np.float32) for _, c in stops])
        grad = np.stack([np.interp(t, [p_ for p_, _ in stops], cols[:, k])
                         for k in range(3)], 1)
        col = np.broadcast_to(grad[:, None, :], (H, W, 3)).copy()
        col += np.random.default_rng(3).normal(0, 6, (H, W, 1)).astype(np.float32)
        out = paint(out, np.clip(col, 0, 255), A)
    else:
        out = paint(out, ink_c, A)
    return out


def _heart(d, cx, cy, r, fill):
    pts = []
    for i in range(60):
        t = 2 * math.pi * i / 60
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((cx + x * r / 16, cy + y * r / 16))
    d.polygon(pts, fill=fill)


def _laurel(d, x0, y, sgn, span, fill, u):
    """A thin stem with small leaves, growing outward from x0."""
    w = max(1, SS)
    x1 = x0 + sgn * span
    d.line([(x0, y), (x1, y)], fill=fill, width=w)
    leaf = u * 0.016 * SS
    n = 4
    for i in range(1, n + 1):
        lx = x0 + sgn * span * (i / (n + 1)) * 0.9
        for up in (-1, 1):
            tip = (lx + sgn * leaf * 1.3, y + up * leaf * 0.9)
            d.polygon([(lx, y), (lx + sgn * leaf * 0.7, y + up * leaf * 0.15), tip,
                       (lx + sgn * leaf * 0.35, y + up * leaf * 0.75)], fill=fill)
    d.ellipse([x1 - leaf * 0.35, y - leaf * 0.35, x1 + leaf * 0.35, y + leaf * 0.35], fill=fill)


def _sprig(d, cx, cy, r, fill, u):
    w = max(1, SS)
    d.line([(cx, cy + r), (cx, cy - r)], fill=fill, width=w)
    for i, t in enumerate((-0.6, -0.1, 0.4)):
        for side in (-1, 1):
            y = cy + t * r
            d.ellipse([cx + side * r * 0.15 - r * 0.25 + side * r * 0.25, y - r * 0.12,
                       cx + side * r * 0.15 + r * 0.25 + side * r * 0.25, y + r * 0.12], fill=fill)


# ---- card -----------------------------------------------------------------------
def card_mask(kind: str, w: int, h: int) -> Image.Image:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    if kind == "arch":
        r = w // 2
        d.rectangle([0, r, w, h], fill=255)
        d.ellipse([0, 0, w, 2 * r], fill=255)
    elif kind == "ticket":
        rr = int(w * 0.09)
        d.rounded_rectangle([0, 0, w, h], radius=int(w * 0.03), fill=255)
        for cx, cy in ((0, 0), (w, 0), (0, h), (w, h)):
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=0)
    else:
        d.rounded_rectangle([0, 0, w, h], radius=int(w * 0.05), fill=255)
    return m


def paper_color(img: Image.Image, box) -> Tuple[int, int, int]:
    a = np.asarray(img.convert("RGB").crop(box), np.float32).reshape(-1, 3).mean(0)
    return tuple(int(v) for v in a * 0.18 + np.array([250, 248, 244]) * 0.82)


def draw_card(base: Image.Image, kind: str, box, accent: str, opacity=0.95):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    m = card_mask(kind, w, h)
    paper = paper_color(base, box)
    out = base.convert("RGBA")
    sh = Image.new("L", base.size, 0)
    sh.paste(m, (x0, y0 + int(h * 0.01)))
    sh = sh.filter(ImageFilter.GaussianBlur(max(6, w // 40)))
    out = Image.alpha_composite(out, Image.merge("RGBA", [Image.new("L", base.size, 0)] * 3 +
                                                 [sh.point(lambda v: int(v * 0.22))]))
    layer = Image.new("RGBA", base.size, paper + (0,))
    full = Image.new("L", base.size, 0)
    full.paste(m, (x0, y0))
    layer.putalpha(full.point(lambda v: int(v * opacity)))
    out = Image.alpha_composite(out, layer)
    # thin metallic border, inset
    ins = max(8, w // 36)
    bw, bh = w - 2 * ins, h - 2 * ins
    if kind == "arch":
        bh = h - 2 * ins
    inner = card_mask(kind, bw, bh)
    edge = inner.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 40 else 0)
    bord = Image.new("L", base.size, 0)
    bord.paste(edge, (x0 + ins, y0 + ins))
    col = Image.new("RGBA", base.size, hex_to_rgb(accent) + (0,))
    col.putalpha(bord.point(lambda v: int(v * 0.85)))
    return Image.alpha_composite(out, col), paper


# ---- layout ---------------------------------------------------------------------
_REG: Optional[FontRegistry] = None


def _max_rect(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Largest axis-aligned rectangle of True cells (histogram method)."""
    h, w = mask.shape
    heights = np.zeros(w, int)
    best, rect = 0, (0, 0, 0, 0)
    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0)
        stack = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best:
                    best, rect = area, (sx, y - sh + 1, x, y + 1)
                start = sx
            stack.append((start, cur))
    return rect


def find_paper(img: Image.Image, margin: float = 0.07) -> Optional[Tuple[int, int, int, int]]:
    """The blank card / paper an AI background drew for the text, in canvas px.

    Paper = the most common LIGHT, smooth color (a white card on a cream table
    stays separate); the card is its biggest connected area. The text column
    is centered, so the box is the tallest centered band (55-90% of the card
    width) that stays on paper -- flowers on the card corners do not shrink it."""
    from scipy import ndimage
    W, H = img.size
    g = 4
    a = np.asarray(img.convert("RGB").resize((W // g, H // g), Image.BOX), np.float32)
    gray = a.mean(2) / 255
    gy, gx = np.gradient(ndimage.gaussian_filter(gray, 1))
    smooth = np.hypot(gx, gy) < 0.02
    light = gray > 0.70
    cand = smooth & light
    if cand.sum() < 0.1 * cand.size:
        return None
    q = (a[cand] // 8).astype(int)
    keys, counts = np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2], return_counts=True)
    top = keys[np.argmax(counts)]
    mode = np.array([top // 1024, (top // 32) % 32, top % 32], np.float32) * 8 + 4
    paper = smooth & (np.abs(a - mode).max(2) < 14)
    paper = ndimage.binary_opening(paper, iterations=1)
    lab, n = ndimage.label(paper)
    if n == 0:
        return None
    sizes = ndimage.sum(paper, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < 0.10 * paper.size:
        return None
    card = ndimage.binary_closing(lab == k, iterations=3)
    # card columns = where paper fills a good part of the column (a strip of
    # light table glued to the card must not stretch its bounds)
    colf = card.mean(0)
    xs = np.where(colf > 0.5 * colf.max())[0]
    cx0, cx1 = xs.min(), xs.max() + 1
    cw = cx1 - cx0
    ccx = (cx0 + cx1) / 2
    best = None
    for f in (0.90, 0.82, 0.74, 0.66, 0.58):
        x0, x1 = int(ccx - cw * f / 2), int(ccx + cw * f / 2)
        ok = card[:, x0:x1].mean(1) > 0.95
        run, start, br = 0, 0, (0, 0)
        for y, v in enumerate(ok):
            if v:
                if run == 0:
                    start = y
                run += 1
                if run > br[1] - br[0]:
                    br = (start, y + 1)
            else:
                run = 0
        area = (x1 - x0) * (br[1] - br[0]) * (1 + 0.6 * f)   # prefer wider
        if best is None or area > best[0]:
            best = (area, x0, br[0], x1, br[1])
    _, x0, y0, x1, y1 = best
    if (x1 - x0) * (y1 - y0) < 0.06 * paper.size:
        return None
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin * 0.5
    return (int((x0 + mw) * g), int((y0 + mh) * g), int((x1 - mw) * g), int((y1 - mh) * g))


MONTHS_VI = {i: f"Tháng {i}" for i in range(1, 13)}
MONTHS_EN = ["", "January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]


def clean_components(comps: List[Dict], language: str) -> List[Dict]:
    """Fix what a small LLM gets wrong in dates: month / day swapped, numeric
    months -> month names."""
    out = []
    for c in comps:
        c = dict(c)
        if c.get("type") == "date":
            m, d = str(c.get("month", "")).strip(), str(c.get("day", "")).strip()
            if m.isdigit() and d.isdigit() and int(m) > 12 >= int(d):
                m, d = d, m
            if m.isdigit() and 1 <= int(m) <= 12:
                m = MONTHS_VI[int(m)] if language == "vietnamese" else MONTHS_EN[int(m)]
            c["month"], c["day"] = m, d
        out.append(c)
    return out


def render(reg: FontRegistry, background: Image.Image, comps: List[Dict],
           language: str = "latin", seed: int = 0, card: str = "auto",
           size: Tuple[int, int] = None, content_box=None,
           overrides: Optional[Dict] = None) -> Tuple[Image.Image, Dict]:
    """(image, report). comps = list of component dicts (see module doc).

    card "none" + content_box: the background already has its own card / paper
    (drawn by the image model); the column is set inside content_box.
    overrides: {"ink", "accent", "scale" (x the fitted size), "shift_y"
    (fraction of the box height), "script", "caps"} -- edits from a reviewer."""
    global _REG
    _REG = reg
    ov = overrides or {}
    W, H = size or background.size
    bg = background.convert("RGB").resize((W, H), Image.LANCZOS)
    rng = random.Random(seed)
    language = normalize_language(reg, language, comps)
    comps = clean_components(comps, language)
    fonts = choose_fonts(reg, comps, language, seed)
    for k in ("script", "caps"):
        if ov.get(k):
            fonts[k] = ov[k]
    fonts["serif"] = fonts["caps"]
    if card == "none" or content_box is not None:
        kind = "none"
        box = content_box or find_paper(bg) or (int(W * .15), int(H * .1), int(W * .85), int(H * .9))
        img = bg.convert("RGBA")
        paper = tuple(int(v) for v in np.asarray(bg.crop(box), np.float32).reshape(-1, 3).mean(0))
        cw, ch = box[2] - box[0], box[3] - box[1]
        pad_x, top_pad, bot_pad = cw * 0.06, ch * 0.04, ch * 0.04
    else:
        kind = card if card in CARD_KINDS else rng.choice(CARD_KINDS)
        cw, ch = int(W * 0.70), int(H * 0.80)
        box = ((W - cw) // 2, (H - ch) // 2 + int(H * 0.01), (W - cw) // 2 + cw,
               (H - ch) // 2 + ch + int(H * 0.01))
        colors = choose_colors(bg, (250, 248, 244))
        img, paper = draw_card(bg, kind, box, colors["accent"])
        pad_x = cw * 0.12
        top_pad = ch * (0.20 if kind == "arch" else 0.08)
        bot_pad = ch * 0.07
    colors = choose_colors(bg, paper)
    for k in ("ink", "accent"):
        if ov.get(k):
            colors[k] = ov[k]
    inner_w = cw - 2 * pad_x
    inner_h = ch - top_pad - bot_pad
    st = style_table(ov.get("sizes"), float(ov.get("tracking", 1.0)))
    effect = ov.get("names_effect", "flat")
    names_cv = Canvas(W, H, reg) if effect in EFFECTS and effect != "flat" else None
    items = [Comp(c, fonts, inner_w, st, names_cv) for c in comps]
    fill = float(ov.get("fill", FILL))
    gap = GAP * inner_w

    def total_h(sc):
        return sum(it.measure(sc) for it in items) + gap * sc * (len(items) - 1)
    # the column fills ~88% of the card height: grow (up to MAX_SCALE) or shrink
    lo, hi = 0.3, MAX_SCALE
    for _ in range(30):
        mid = (lo + hi) / 2
        if total_h(mid) <= inner_h * fill and                 max([it.need_w(mid) for it in items] + [0]) <= inner_w:
            lo = mid
        else:
            hi = mid
    s = lo
    total = total_h(s)
    cv = Canvas(W, H, reg)
    cx = (box[0] + box[2]) / 2
    y = box[1] + top_pad + (inner_h - total) / 2 + float(ov.get("shift_y", 0)) * ch
    for it in items:
        it.draw(cv, cx, y, s, colors)
        y += it.measure(s) + gap * s
    layer = cv.img.resize((W, H), Image.LANCZOS)
    out = Image.alpha_composite(img, layer)
    if names_cv is not None:
        # foil must still read on the paper: keep the metal hue, deepen it
        foil = rgb_to_hex(fit_contrast(hex_to_rgb(colors["accent"]), paper, 3.2))             if effect == "foil" else colors["accent"]
        out = apply_name_effect(out, names_cv.img, effect, colors["ink"], foil)
    return out.convert("RGB"), {"card": kind, "box": list(box), "fonts": fonts,
                                "colors": colors, "scale": round(s, 3), "effect": effect}


# ---- several boxes (one per text group) -----------------------------------------
GROUP_OF = {"icon": "header", "names": "header", "text": "header", "date": "date",
            "divider": None, "venue": "place"}


def split_groups(comps: List[Dict]) -> Dict[str, List[Dict]]:
    """header (kicker, names, invite line) / date / place (venue, note).
    Dividers stick to the group that follows; a note after the venue goes to
    place, a kicker / invite line before the date to header."""
    groups = {"header": [], "date": [], "place": []}
    cur, pending = "header", []
    for c in comps:
        g = GROUP_OF.get(c.get("type"))
        if c.get("type") == "text" and c.get("style") == "note":
            g = "place"
        if g is None:
            pending.append(c)
            continue
        if g == "header" and cur != "header":
            g = cur                       # text after the date stays below
        cur = g
        groups[g] += pending + [c]
        pending = []
    groups[cur] += [p for p in pending if p.get("type") != "divider"]
    return {k: v for k, v in groups.items() if v}


# readable floors, fraction of the canvas short side (px = floor x short)
FLOOR = {"name": 0.065, "day": 0.050, "venue": 0.022, "kicker": 0.016,
         "invite": 0.016, "month": 0.017, "weekday": 0.015, "time": 0.015,
         "address": 0.015, "note": 0.018, "connector": 0.030}
PRIORITY = {"header": "name", "date": "day", "place": "venue"}
SECONDARY = ("kicker", "invite", "note", "weekday", "time", "month", "address")
MIN_KEEP_W, MIN_KEEP_H = 0.70, 0.60
GROUP_GAP_MIN, GROUP_GAP_MAX = 1.3, 2.4   # gap between groups, x the line gap inside
BAND_SLACK = 0.35                         # a group may leave its band by this x its height


def _styles_in(comps):
    keys = set()
    for c in comps:
        t = c.get("type")
        if t == "names":
            keys |= {"name", "connector"}
        elif t == "date":
            keys |= {"day", "month", "weekday", "time"}
        elif t == "venue":
            keys |= {"venue", "address"}
        elif t == "text":
            keys.add(c.get("style", "invite"))
    return keys


RHYTHM = 0.85                 # gap between two lines, x the smaller of their sizes
RELAX_GAIN = 1.15             # take the relaxed box when type grows by this
FLOW_MIN_W = 0.62             # flowing around decorations never narrows a line below this
NAME_MIN_K = 0.88             # the names: at most 12% smaller, a slight touch is accepted
LINE_MIN_K = 0.70             # a touching line shrinks to 70% of its size at most
TOUCH_TOL = 1.03              # a line counts as touching when wider than the free paper by this
MODE_MIN_W = {"wrap": 0.45, "both": 0.75, "shrink": 1.0, "keep": 1.0,
              "auto": 0.45}             # auto = wrap first, shrink only what still touches   # how far each fit mode wraps


def stack_gaps(items, s, cap):
    """Gap after each item but the last. Typographic rhythm: two small caps
    lines sit close, a big name keeps air around it; never above `cap`."""
    ems = [it.em(s) for it in items]
    out = []
    for i in range(len(items) - 1):
        a, b = ems[i], ems[i + 1]
        known = [e for e in (a, b) if e]
        e = min(known) if known else cap / RHYTHM
        if a is None or b is None:            # around a divider: a bit more air
            e *= 1.25
        out.append(min(cap, RHYTHM * e))
    return out


def stack_h(items, s, cap):
    return sum(it.measure(s) for it in items) + sum(stack_gaps(items, s, cap))


def _fit_group(comps, fonts, st, names_cv, box, fallback_box, short, allow_grow=True):
    """(scale, box, notes). Biggest scale that fits the box; then guarantee
    the readable floors: shrink the secondary lines first, then fall back to
    the wider unclipped box, and as a last resort accept the floor."""
    notes = []
    if fallback_box is None:
        fallbacks = []
    elif isinstance(fallback_box, list):
        fallbacks = list(fallback_box)
    else:
        fallbacks = [fallback_box]
    keys = _styles_in(comps)
    prio = next((PRIORITY[g] for g in PRIORITY if PRIORITY[g] in keys), None)

    def fit(bx):
        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        items = [Comp(c, fonts, bw * 0.94, st, names_cv) for c in comps]
        gap = GAP * bw * 0.94

        def total_h(sc):
            return stack_h(items, sc, gap * sc)
        lo, hi = 0.2, MAX_SCALE * 1.6
        for _ in range(30):
            mid = (lo + hi) / 2
            if total_h(mid) <= bh * 0.96 and \
                    max([it.need_w(mid) for it in items] + [0]) <= bw * 0.94:
                lo = mid
            else:
                hi = mid
        return lo, bw * 0.94

    def px(key, sc, u):
        return st[key][0] * u * sc

    def ok(sc, u):
        return all(px(k, sc, u) >= FLOOR[k] * short * 0.999 for k in keys if k in FLOOR)

    for attempt in range(12):
        sc, u = fit(box)
        if ok(sc, u):
            # floors reached -- but a slightly relaxed box (text may graze a
            # decoration; the flow step wraps around it) is taken when it
            # gives clearly bigger type
            for alt in (fallbacks if prio == "name" and allow_grow else []):   # names only
                alt = tuple(alt)
                if alt == tuple(box):
                    continue
                sc2, u2 = fit(alt)
                if sc2 * u2 >= sc * u * RELAX_GAIN and ok(sc2, u2):
                    notes.append("relaxed box for bigger type")
                    return sc2, alt, notes
            return sc, box, notes
        if prio and px(prio, sc, u) < FLOOR[prio] * short:
            # make room for the important line: secondary lines shrink first
            shrunk = False
            for k in SECONDARY:
                if k in keys and k in st:
                    size, ls, kind = st[k]
                    if size * u * sc * 0.88 >= FLOOR[k] * short:
                        st[k] = (size * 0.88, ls, kind)
                        shrunk = True
            if shrunk:
                notes.append("secondary lines shrunk")
                continue
        # relax step by step: the least decoration overlap that reaches the floors
        nxt = None
        while fallbacks:
            cand = tuple(fallbacks.pop(0))
            if cand != tuple(box):
                nxt = cand
                break
        if nxt is not None:
            box = nxt
            notes.append(f"relaxed box ({len(notes)})")
            continue
        break
    # last resort: the floor wins (the column may touch the decorations)
    sc, u = fit(box)
    need = max([FLOOR[k] * short / (st[k][0] * u) for k in keys if k in FLOOR] + [sc])
    if need > sc:
        notes.append(f"floor forced x{need / sc:.2f}")
    return max(sc, need), box, notes


def normalize_language(reg: FontRegistry, language: str, comps: List[Dict]) -> str:
    """LLMs say "english", "en", "vi"...: map to a language the font index knows."""
    from .parsing import looks_vietnamese
    lang = str(language or "").strip().lower()
    alias = {"vi": "vietnamese", "vn": "vietnamese", "tiếng việt": "vietnamese",
             "en": "latin", "english": "latin", "eng": "latin"}
    lang = alias.get(lang, lang)
    texts = [{"text": t} for t in _texts(comps)]
    if looks_vietnamese(texts):
        return "vietnamese"
    return lang if reg.candidates(lang) else "latin"


def render_boxes(reg: FontRegistry, background: Image.Image,
                 groups: Dict[str, Tuple[List[Dict], Tuple[int, int, int, int]]],
                 language: str = "latin", seed: int = 0,
                 overrides: Optional[Dict] = None,
                 fallback: Optional[Dict] = None,
                 mask: Optional[np.ndarray] = None) -> Tuple[Image.Image, Dict]:
    """Each group set as a centered column inside ITS box, with its own scale
    (the header box usually gets the biggest type). Fonts, colors and the
    name effect are shared so the card stays one design."""
    global _REG
    _REG = reg
    ov = overrides or {}
    bg = background.convert("RGB")
    W, H = bg.size
    every = [c for comps, _ in groups.values() for c in comps]
    language = normalize_language(reg, language, every)
    every = clean_components(every, language)
    fonts = choose_fonts(reg, every, language, seed)
    for k in ("script", "caps"):
        if ov.get(k):
            fonts[k] = ov[k]
    fonts["serif"] = fonts["caps"]
    boxes = [b for _, b in groups.values()]
    union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
             max(b[2] for b in boxes), max(b[3] for b in boxes))
    paper = tuple(int(v) for v in np.asarray(bg.crop(union), np.float32).reshape(-1, 3).mean(0))
    colors = choose_colors(bg, paper)
    for k in ("ink", "accent"):
        if ov.get(k):
            colors[k] = ov[k]
    st = style_table(ov.get("sizes"), float(ov.get("tracking", 1.0)))
    effect = ov.get("names_effect", "flat")
    names_cv = Canvas(W, H, reg) if effect in EFFECTS and effect != "flat" else None
    cv = Canvas(W, H, reg)
    report = {}
    short = min(W, H)
    # 1) fit every group in its own box (floors guaranteed)
    fitted = []
    for name, (comps, box) in groups.items():
        comps = clean_components(comps, language)
        gst = dict(st)                       # per-group: secondary lines may shrink
        # in the fit modes (shrink / wrap / both) the type never GROWS: wrapping
        # keeps the original size, shrinking only makes it smaller
        s_, box, notes = _fit_group(comps, fonts, gst, names_cv, box,
                                    (fallback or {}).get(name), short,
                                    allow_grow=not (ov.get("fit_mode") or ov.get("fit_by_group")))
        fitted.append({"name": name, "comps": comps, "st": gst, "s": s_,
                       "box": tuple(box), "notes": notes, "orig": tuple(box)})

    # 2) one typographic system: every secondary line (kicker, invite, time,
    #    address...) gets the SAME px size on the whole card (the smallest
    #    the groups reached, never under its floor)
    for k in SECONDARY + ("connector",):
        sizes = [f["st"][k][0] * f["box"][2] * 0.94 * f["s"] - f["st"][k][0] *
                 f["box"][0] * 0.94 * f["s"] for f in fitted if k in _styles_in(f["comps"])]
        if len(sizes) < 2:
            continue
        target = max(min(sizes), FLOOR.get(k, 0) * short)
        for f in fitted:
            if k in _styles_in(f["comps"]):
                u = (f["box"][2] - f["box"][0]) * 0.94 * f["s"]
                size, ls, kind = f["st"][k]
                f["st"][k] = (target / u, ls, kind)

    def build(f):
        bw = f["box"][2] - f["box"][0]
        items = [Comp(c, fonts, bw * 0.94, f["st"], names_cv) for c in f["comps"]]
        gap = GAP * bw * 0.94
        h = stack_h(items, f["s"], gap * f["s"])
        return items, gap, h

    for f in fitted:
        f["items"], f["gap"], f["h"] = build(f)

    # 3) one center line when the boxes allow it (no zig-zag columns)
    xs0 = max(f["box"][0] for f in fitted)
    xs1 = min(f["box"][2] for f in fitted)
    widest = max(f["box"][2] - f["box"][0] for f in fitted)
    stacked = all(f["box"][1] >= g["box"][3] or g["box"][1] >= f["box"][3]
                  for f in fitted for g in fitted if f is not g)
    shared_cx = (xs0 + xs1) / 2 if stacked and xs1 - xs0 >= 0.6 * widest else None
    mode = ov.get("fit_mode")                  # None | keep | auto | shrink | wrap | both
    by_group = ov.get("fit_by_group") or {}    # per group, e.g. from the AI: {"header": "wrap"}
    if by_group and not mode:
        mode = "keep"

    def mode_of(f):
        return by_group.get(f["name"], mode)
    for f in fitted:
        f["cx"] = shared_cx if shared_cx is not None else (f["box"][0] + f["box"][2]) / 2
        if mode and ov.get("axis") is not None:
            f["cx"] = float(ov["axis"])            # the column never shifts sideways
        if ov.get("align") == "left":              # flush left: X is the shared left edge
            f["cx"] = f["box"][0] + (f["box"][2] - f["box"][0]) * 0.03
            for it in f["items"]:
                it.align = "left"
            f["h"] = stack_h(f["items"], f["s"], f["gap"] * f["s"])

    def _fcx(f):
        """Where the blank width is measured: the middle of the text column
        (for a flush-left column that is not its left edge)."""
        if ov.get("align") == "left":
            return f["cx"] + (f["box"][2] - f["box"][0]) * 0.47
        return f["cx"]

    def place():
        # 4) even rhythm: the groups as one stack with EQUAL gaps between them,
        #    centered in the span of the boxes -- unless that pushes a group far
        #    out of its own (decoration-free) band
        order = sorted(fitted, key=lambda f: f["box"][1])
        if stacked and len(order) > 1:
            top, bot = order[0]["box"][1], order[-1]["box"][3]
            free = (bot - top) - sum(f["h"] for f in order)
            # the gap between groups is related to the line gap inside them
            # (capped), the rest of the free space goes ABOVE and BELOW the
            # whole block: one compact, centered column instead of holes
            inner = sorted((sorted(stack_gaps(f["items"], f["s"], f["gap"] * f["s"]))
                            or [f["gap"] * f["s"]])[-1] for f in order)[len(order) // 2]
            between = min(max(0.0, free / (len(order) + 1)),
                          GROUP_GAP_MAX * inner)
            between = max(between, min(GROUP_GAP_MIN * inner, max(0.0, free / (len(order) + 1))))
            block = sum(f["h"] for f in order) + between * (len(order) - 1)
            y = top + max(0.0, (bot - top) - block) / 2
            placed = []
            for f in order:
                placed.append(y)
                y += f["h"] + between
            if all(py >= f["box"][1] - BAND_SLACK * (f["box"][3] - f["box"][1]) and
                   py + f["h"] <= f["box"][3] + BAND_SLACK * (f["box"][3] - f["box"][1])
                   for py, f in zip(placed, order)):
                for py, f in zip(placed, order):
                    f["y"] = py
        for f in fitted:
            if "y" not in f:
                f["y"] = f["box"][1] + (f["box"][3] - f["box"][1] - f["h"]) / 2


    # 5) text flows around the decorations: a line whose height band is cut by
    #    a flower / ribbon is wrapped (or a name shrunk) to the blank width
    #    there; the type size stays the same
    if mask is None:
        mask = paper_mask(bg)
    flowed = 0
    for _ in range(2):
        for f in fitted:
            f.pop("y", None)
        place()
        for f in fitted:
            y = f["y"]
            base_w = (f["box"][2] - f["box"][0]) * 0.94
            gps = stack_gaps(f["items"], f["s"], f["gap"] * f["s"]) + [0]
            for it, gp in zip(f["items"], gps):
                h_ = it.measure(f["s"])
                if it.c.get("type") in ("text", "venue"):   # names cannot wrap
                    fw = free_width(mask, _fcx(f), y, y + h_) * 0.96
                    if fw == float("inf"):
                        fw = base_w
                    it.wrap = max(min(base_w, fw), base_w * MODE_MIN_W.get(mode_of(f), FLOW_MIN_W))
                    flowed += it.wrap < base_w * 0.99
                y += h_ + gp
            f["h"] = stack_h(f["items"], f["s"], f["gap"] * f["s"])
    for f in fitted:
        f.pop("y", None)
    place()
    if any(mode_of(f) in ("shrink", "both", "auto") for f in fitted):
        # a line still wider than the blank paper at its height gets smaller --
        # ONLY that line (a cramped kicker under the arch must not shrink the
        # names); "shrink" mode keeps its lines, so it shrinks every touching line
        for _ in range(6):
            changed = False
            for f in fitted:
                if mode_of(f) not in ("shrink", "both", "auto"):
                    continue
                y = f["y"]
                gps = stack_gaps(f["items"], f["s"], f["gap"] * f["s"]) + [0]
                for it, gp in zip(f["items"], gps):
                    h_ = it.measure(f["s"])
                    if it.c.get("type") in ("text", "venue", "names", "date"):
                        fw = free_width(mask, _fcx(f), y, y + h_) * 0.96
                        lw = it.line_w(f["s"])
                        kmin = NAME_MIN_K if it.c.get("type") == "names" else LINE_MIN_K
                        if lw > fw > 0 and fw != float("inf") and it.k > kmin:
                            it.k = max(kmin, it.k * max(fw / lw, 0.88))
                            changed = True
                    y += it.measure(f["s"]) + gp
                f["h"] = stack_h(f["items"], f["s"], f["gap"] * f["s"])
            for f in fitted:
                f.pop("y", None)
            place()
            if not changed:
                break
    nudge = ov.get("nudge") or {}
    step = 0.035 * (max(f["box"][3] for f in fitted) - min(f["box"][1] for f in fitted))         if fitted else 0
    for f in fitted:
        k = {"up": -1, "down": 1}.get(nudge.get(f["name"]), 0)
        if k:
            bh = f["box"][3] - f["box"][1]
            lo = f["box"][1] - 0.15 * bh
            hi = f["box"][3] + 0.15 * bh - f["h"]
            f["y"] = min(max(f["y"] + k * step, lo), max(lo, hi))
    for f in fitted:
        y = f["y"]
        gps = stack_gaps(f["items"], f["s"], f["gap"] * f["s"]) + [0]
        touch = []                             # lines wider than the blank paper there
        for it, gp in zip(f["items"], gps):
            it.draw(cv, f["cx"], y, f["s"], colors)
            h_ = it.measure(f["s"])
            if it.c.get("type") in ("text", "venue", "names", "date"):
                fw = free_width(mask, _fcx(f), y, y + h_) * 0.96
                lw = it.line_w(f["s"])
                if lw > fw * TOUCH_TOL:
                    touch.append({"type": it.c["type"], "over": round(lw / max(1.0, fw), 2)})
            y += h_ + gp
        report[f["name"]] = {"box": list(f["box"]), "scale": round(f["s"], 3),
                             "notes": f["notes"], "touch": touch}
    report["_layout"] = {"shared_center": shared_cx is not None,
                         "even_rhythm": all("y" in f for f in fitted),
                         "flowed_lines": int(flowed)}
    out = Image.alpha_composite(bg.convert("RGBA"), cv.img.resize((W, H), Image.LANCZOS))
    if names_cv is not None:
        foil = rgb_to_hex(fit_contrast(hex_to_rgb(colors["accent"]), paper, 3.2)) \
            if effect == "foil" else colors["accent"]
        out = apply_name_effect(out, names_cv.img, effect, colors["ink"], foil)
    return out.convert("RGB"), {"groups": report, "fonts": fonts, "colors": colors}


# ---- AI layout -> group boxes ----------------------------------------------------
LAYOUTS = ("stacked", "airy", "split_bottom")


def _paper_base(img: Image.Image, g: int):
    from scipy import ndimage
    W, H = img.size
    a = np.asarray(img.convert("RGB").resize((W // g, H // g), Image.BOX), np.float32)
    gray = a.mean(2) / 255
    gy, gx = np.gradient(ndimage.gaussian_filter(gray, 1))
    return a, gray, np.hypot(gx, gy) < 0.02


def paper_tones(img: Image.Image, g: int = 4, k: int = 3) -> List[np.ndarray]:
    """The k most common light, smooth colors (distinct ones), most common first."""
    a, gray, smooth = _paper_base(img, g)
    cand = smooth & (gray > 0.70)
    if cand.sum() < 0.05 * cand.size:
        return []
    q = (a[cand] // 8).astype(int)
    keys, counts = np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2], return_counts=True)
    out = []
    for key in keys[np.argsort(-counts)][:200]:
        t = np.array([key // 1024, (key // 32) % 32, key % 32], np.float32) * 8 + 4
        if all(np.abs(t - o).max() >= 34 for o in out):
            out.append(t)
        if len(out) == k:
            break
    return out


def paper_mask(img: Image.Image, g: int = 4, tone=None) -> np.ndarray:
    """Bool mask (1/g resolution) of the blank paper: by default the most common
    light, smooth color of the picture (same test as find_paper); `tone` = the
    card's own paper color when it is known (a pink card on a white table)."""
    from scipy import ndimage
    a, gray, smooth = _paper_base(img, g)
    if tone is None:
        tones = paper_tones(img, g, 1)
        if not tones:
            return np.ones_like(gray, bool)
        tone = tones[0]
    mode = np.asarray(tone, np.float32)
    # lit / photographed cards shade from bright to dim: judge by lightness +
    # low saturation + smoothness, and a generous distance to the paper tone
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / np.maximum(1.0, mx)
    near = np.abs(a - mode).max(2) < 34
    # tinted paper (peach under warm light...): the limit follows the paper tone
    mode_sat = float((mode.max() - mode.min()) / max(1.0, mode.max()))
    return ndimage.binary_closing(smooth & near & (sat < max(0.25, mode_sat + 0.12))
                                  & (gray > 0.55), iterations=2)


def fit_to_paper(box, mask: np.ndarray, g: int = 4, need: float = 0.88):
    """Shrink a box until (almost) all of it is blank paper: trim top / bottom
    rows that run into flowers or the arch, then narrow it symmetrically."""
    x0, y0, x1, y1 = [int(v / g) for v in box]
    h, w = mask.shape
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    min_w, min_h = (x1 - x0) * MIN_KEEP_W, (y1 - y0) * MIN_KEEP_H
    for _ in range(80):
        if x1 - x0 <= min_w or y1 - y0 <= min_h:
            break                     # a ribbon / stem crossing: text may touch it
        rows = mask[y0:y1, x0:x1].mean(1)
        if rows.min() >= need:
            break
        bad = rows < need
        if bad[0]:
            y0 += 1
        elif bad[-1]:
            y1 -= 1
        else:                         # obstruction in the middle rows: narrow
            dx = max(1, (x1 - x0) // 40)  # from the side the decoration is on
            sub = mask[y0:y1, x0:x1]
            q = max(1, (x1 - x0) // 4)
            left, right = sub[:, :q].mean(), sub[:, -q:].mean()
            if abs(left - right) < 0.03:
                x0, x1 = x0 + dx, x1 - dx
            elif left < right:
                x0 += 2 * dx
            else:
                x1 -= 2 * dx
    return (x0 * g, y0 * g, x1 * g, y1 * g)


def free_width(mask: np.ndarray, cx: float, y0: float, y1: float, g: int = 4) -> float:
    """Width (px) of blank paper centered on cx over rows y0..y1: the text of a
    centered line fits if it is narrower. Decorations at the edges cut it."""
    h, w = mask.shape
    r0, r1 = max(0, int(y0 / g)), min(h, int(np.ceil(y1 / g)))
    c = int(min(w - 1, max(0, cx / g)))
    if r1 <= r0:
        return float(w * g)
    band = mask[r0:r1]
    # a column counts as free when nearly all its rows in the band are paper
    free = band.mean(0) >= 0.9
    if not free[c]:
        # nothing measurable at the axis (a word sits on a decoration, or the
        # paper was not recognised): only "touching" if paper exists nearby
        near = band[:, max(0, c - w // 6):c + w // 6 + 1].mean()
        return 0.0 if near > 0.25 else float("inf")
    left = c
    while left > 0 and free[left - 1]:
        left -= 1
    right = c
    while right < w - 1 and free[right + 1]:
        right += 1
    return float(2 * min(c - left, right - c) * g)


def plan_boxes(card, layout: str, shares: Dict[str, float], groups,
               gap: float = 0.035) -> Dict[str, Tuple[int, int, int, int]]:
    """Split the card box into one box per text group, from the AI's layout
    choice + height shares (normalised). Geometry stays valid by construction."""
    x0, y0, x1, y1 = card
    W, H = x1 - x0, y1 - y0
    names = [g for g in ("header", "date", "place") if g in groups]
    sh = {g: max(0.05, float(shares.get(g, 1.0 / len(names)))) for g in names}
    if layout == "airy":                       # a calm margin above and below
        y0, y1 = y0 + int(H * 0.08), y1 - int(H * 0.08)
        H = y1 - y0
    boxes = {}
    if layout == "split_bottom" and {"date", "place"} <= set(names):
        tot = sh.get("header", 0.5) + max(sh["date"], sh["place"])
        hh = int(H * sh.get("header", 0.5) / tot)
        g = int(H * gap)
        if "header" in names:
            boxes["header"] = (x0, y0, x1, y0 + hh - g // 2)
        mid = (x0 + x1) // 2
        boxes["date"] = (x0, y0 + hh + g // 2, mid - g, y1)
        boxes["place"] = (mid + g, y0 + hh + g // 2, x1, y1)
        return boxes
    tot = sum(sh.values())
    g = int(H * gap)
    y = y0
    for i, n in enumerate(names):
        hgt = int((H - g * (len(names) - 1)) * sh[n] / tot)
        boxes[n] = (x0, y, x1, y + hgt)
        y += hgt + g
    return boxes


# ---- tilted / skewed cards -------------------------------------------------------
TILT_MIN_DEG = 2.0            # below this a card counts as straight
QUAD_MIN_FILL = 0.83          # paper share inside the quad: lower = not one clean card
QUAD_SIDE_TOL = 0.06          # contour points this close (x card size) belong to a side


def _order_corners(c: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL."""
    s, d = c.sum(1), c[:, 1] - c[:, 0]
    return np.array([c[np.argmin(s)], c[np.argmin(d)], c[np.argmax(s)], c[np.argmax(d)]], np.float64)


def _intersect(l1, l2):
    (p, u), (q, v) = l1, l2
    a = np.array([[u[0], -v[0]], [u[1], -v[1]]])
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    t = np.linalg.solve(a, q - p)[0]
    return p + t * u


def card_quad(img: Image.Image, with_tone: bool = False):
    """The 4 corners (TL, TR, BR, BL, canvas px) of the blank paper card, or None.

    Each side is a line fitted to the paper outline along the middle of that
    side, so a flower over a corner does not move the corner, and a card seen
    in perspective (a trapezoid) is found as it is. A paper area that is not
    one clean card (it ran into a white tablecloth) gives None. The most
    common light tone may be the table, so a few paper tones are tried and
    the biggest clean card wins. with_tone -> (corners, paper tone)."""
    from scipy import ndimage
    g = 4
    best = None                       # (cover, corners, tone): the cleanest card wins
    for tone in paper_tones(img, g):
        m = paper_mask(img, g, tone)
        # same-colored things touching the card (pink lotus on a pink card) are
        # cut off by an opening; sides come from the outline's min-area rect or
        # from its longest straight stretches
        for it in (0, 3, 6, 10, 15, 20):
            mm = ndimage.binary_opening(m, iterations=it) if it else m
            for method in ("rect", "segments"):
                q = _quad_of(mm, g, method)
                if q is not None and (best is None or q[1] > best[0]):
                    best = (q[1], q[0], tone)
    if best is None:
        return (None, None) if with_tone else None
    return (best[1], best[2]) if with_tone else best[1]


QUAD_MAX_SIDE_DEG = 20.0      # a side further than this from its axis: not a card
QUAD_MAX_PERSP_DEG = 3.0      # opposite sides further from parallel: a decoration pulled one side


def _side_angles(quad):
    tl, tr, br, bl = quad
    return (np.degrees(np.arctan2(*(tr - tl)[::-1])), np.degrees(np.arctan2(*(br - bl)[::-1])),
            np.degrees(np.arctan2((bl - tl)[0], (bl - tl)[1])), np.degrees(np.arctan2((br - tr)[0], (br - tr)[1])))


def _fit_side(pts, p, u, sel):
    import cv2
    if sel.sum() < 12:
        return p, u
    vx, vy, x0, y0 = cv2.fitLine(pts[sel].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)


def _sides_by_rect(cnt, pts):
    """Sides = the outline along the middle of each minimum-area-rect side."""
    import cv2
    rect = cv2.minAreaRect(cnt)
    box = _order_corners(cv2.boxPoints(rect).astype(np.float64))
    tol = QUAD_SIDE_TOL * min(rect[1])
    lines = []
    for i in range(4):                               # top, right, bottom, left
        p, q = box[i], box[(i + 1) % 4]
        if np.linalg.norm(q - p) < 1:
            return None
        u = (q - p) / np.linalg.norm(q - p)
        rel = pts - p
        t = rel @ u / np.linalg.norm(q - p)
        dist = np.abs(rel @ np.array([-u[1], u[0]]))
        lines.append(_fit_side(pts, p, u, (t > 0.15) & (t < 0.85) & (dist < tol)))
    return lines


def _sides_by_segments(cnt, pts, size):
    """Sides = the longest straight stretches of the outline (a card edge is
    straight, a flower stuck to it is not): two far-apart near-horizontal
    ones and two far-apart near-vertical ones."""
    import cv2
    poly = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True).reshape(-1, 2).astype(np.float64)
    segs = {"h": [], "v": []}
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        L = np.linalg.norm(q - p)
        if L < 0.25 * size:
            continue
        ang = np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])) % 180
        if min(ang, 180 - ang) < QUAD_MAX_SIDE_DEG:
            segs["h"].append((L, p, q))
        elif abs(ang - 90) < QUAD_MAX_SIDE_DEG:
            segs["v"].append((L, p, q))
    pair = {}
    for kind, axis in (("h", 1), ("v", 0)):
        c = sorted(segs[kind], key=lambda x: -x[0])
        if not c:
            return None
        m0 = (c[0][1] + c[0][2]) / 2
        far = [x for x in c[1:] if abs((x[1] + x[2])[axis] / 2 - m0[axis]) > 0.4 * size]
        if not far:
            return None
        pair[kind] = sorted([c[0], far[0]], key=lambda x: (x[1] + x[2])[axis])
    tol = 0.02 * size
    lines = []
    for L, p, q in (pair["h"][0], pair["v"][1], pair["h"][1], pair["v"][0]):   # top, right, bottom, left
        u = (q - p) / L
        rel = pts - p
        t = rel @ u / L
        dist = np.abs(rel @ np.array([-u[1], u[0]]))
        lines.append(_fit_side(pts, p, u, (t > -0.3) & (t < 1.3) & (dist < tol)))
    return lines


def _quad_ok(quad, comp):
    """A clean card: convex, mostly paper inside, the main part of the paper
    area, inside the picture, sides near the axes, not too skewed."""
    import cv2
    q32 = quad.astype(np.float32)
    area = cv2.contourArea(q32)
    if area <= 0 or not cv2.isContourConvex(q32):
        return 0.0
    h, w = comp.shape
    if (quad[:, 0] < -0.1 * w).any() or (quad[:, 0] > 1.1 * w).any() or \
            (quad[:, 1] < -0.1 * h).any() or (quad[:, 1] > 1.1 * h).any():
        return 0.0
    t, b, l, r = _side_angles(quad)
    if max(abs(t), abs(b), abs(l), abs(r)) > QUAD_MAX_SIDE_DEG or \
            abs(t - b) > QUAD_MAX_PERSP_DEG or abs(l - r) > QUAD_MAX_PERSP_DEG:
        return 0.0
    inside = np.zeros_like(comp)
    cv2.fillConvexPoly(inside, np.round(quad).astype(np.int32), 1)
    cover = (comp & inside).sum() / max(1, inside.sum())
    if cover < QUAD_MIN_FILL or area < 0.45 * comp.sum():
        return 0.0
    return float(cover)


def _quad_of(m: np.ndarray, g: int, method: str = "rect"):
    """(corners in canvas px, paper share inside) of the biggest paper area, if it is a clean card."""
    import cv2
    from scipy import ndimage
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < 0.08 * m.size:
        return None
    comp = ndimage.binary_fill_holes(lab == k).astype(np.uint8)
    # cut by the picture frame on 2+ sides: a table or a wall, not a card
    if sum(bool(e.any()) for e in (comp[0], comp[-1], comp[:, 0], comp[:, -1])) >= 2:
        return None
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2).astype(np.float64)
    if method == "rect":
        lines = _sides_by_rect(cnt, pts)
    else:
        lines = _sides_by_segments(cnt, pts, np.sqrt(comp.sum()))
    if lines is None:
        return None
    top, right, bottom, left = lines
    corners = [_intersect(left, top), _intersect(top, right), _intersect(right, bottom),
               _intersect(bottom, left)]
    if any(c is None for c in corners):
        return None
    quad = _order_corners(np.array(corners))
    cover = _quad_ok(quad, comp)
    return (quad * g, cover) if cover else None


def trim_card(box, mask: np.ndarray, g: int = 4, row_need: float = 0.9, row_band: float = 0.4,
              col_need: float = 0.7, col_band: float = 0.6, max_cut: float = 0.35):
    """The card's usable area: cut rows (top / bottom) whose middle is not
    clean paper -- dinosaurs standing in front of the card's foot, a ribbon
    over its top -- and columns that are mostly covered. Corner decorations
    stay out of the middle band and are handled by fit_to_paper."""
    x0, y0, x1, y1 = [int(round(v / g)) for v in box]
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return box
    bw, bh = x1 - x0, y1 - y0
    mx = int(bw * (1 - row_band) / 2)
    my = int(bh * (1 - col_band) / 2)
    rows = mask[y0:y1, x0 + mx:x1 - mx].mean(1)
    cols = mask[y0 + my:y1 - my, x0:x1].mean(0)

    def cut(v, need):
        lim = int(len(v) * max_cut)
        a = next((i for i in range(lim) if v[i] >= need), lim)
        b = next((i for i in range(lim) if v[len(v) - 1 - i] >= need), lim)
        return a, b
    t, b = cut(rows, row_need)
    l, r = cut(cols, col_need)
    return ((x0 + l) * g, (y0 + t) * g, (x1 - r) * g, (y1 - b) * g)


def quad_skew(quad: np.ndarray) -> float:
    """Largest angle (deg) between a card side and the canvas axes."""
    return float(max(abs(v) for v in _side_angles(quad)))


def flatten_card(img: Image.Image, quad: np.ndarray):
    """Warp the picture so the card becomes an upright rectangle (same size
    canvas, card centered where it was). -> (flat image, H, card box)."""
    import cv2
    tl, tr, br, bl = quad
    cw = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    ch = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    cx, cy = quad.mean(0)
    dst = np.array([[cx - cw / 2, cy - ch / 2], [cx + cw / 2, cy - ch / 2],
                    [cx + cw / 2, cy + ch / 2], [cx - cw / 2, cy + ch / 2]])
    Hm = cv2.getPerspectiveTransform(quad.astype(np.float32), dst.astype(np.float32))
    W, Hh = img.size
    flat = cv2.warpPerspective(np.asarray(img.convert("RGB")), Hm, (W, Hh), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    box = tuple(int(round(v)) for v in (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))
    return Image.fromarray(flat), Hm, box


def unflatten_text(out_flat: Image.Image, flat_bg: Image.Image, bg: Image.Image,
                   Hm: np.ndarray, quad: np.ndarray) -> Image.Image:
    """Put ONLY what the render changed (text, ornaments, their shadows and
    highlights) back on the original picture, in the card's own perspective.
    The change is carried as a signed difference, so soft edges, emboss and
    foil keep their exact tone and the background is never resampled."""
    import cv2
    W, Hh = bg.size
    delta = np.asarray(out_flat, np.float32) - np.asarray(flat_bg, np.float32)
    back = cv2.warpPerspective(delta, np.linalg.inv(Hm), (W, Hh), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = np.zeros((Hh, W), np.uint8)
    cv2.fillConvexPoly(inside, np.round(quad).astype(np.int32), 1)
    back *= inside[..., None]
    return Image.fromarray(np.clip(np.asarray(bg, np.float32) + back, 0, 255).astype(np.uint8))


def card_tilt(img: Image.Image) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
    """(angle in degrees, (cx, cy, w, h)) of the blank paper card from its
    corners. Positive angle = clockwise."""
    q = card_quad(img)
    if q is None:
        return 0.0, None
    tl, tr, br, bl = q
    ang = np.degrees((np.arctan2(*(tr - tl)[::-1]) + np.arctan2(*(br - bl)[::-1])) / 2)
    cx, cy = q.mean(0)
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return float(ang), (float(cx), float(cy), float(w), float(h))


def render_on_card(render_fn, bg: Image.Image):
    """Lay the text out on the card made flat and upright, then put ONLY the
    text back in the card's angle and perspective.
    render_fn(flat_bg, card_box or None, paper tone or None) -> (image, report)."""
    quad, tone = card_quad(bg, with_tone=True)
    if quad is None or quad_skew(quad) < TILT_MIN_DEG:
        out, rep = render_fn(bg, None, tone)
        rep["_skew"] = 0.0
        return out, rep
    flat, Hm, box = flatten_card(bg, quad)
    out_f, rep = render_fn(flat, box, tone)
    rep["_skew"] = round(quad_skew(quad), 1)
    return unflatten_text(out_f, flat, bg, Hm, quad), rep
