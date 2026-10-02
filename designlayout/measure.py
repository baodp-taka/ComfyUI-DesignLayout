# -*- coding: utf-8 -*-
"""Real-font text measuring, wrapping and fit-to-box.

All measuring uses the actual font (Pillow ImageFont.getbbox / getlength) with
letter spacing, and supports per-glyph font fallback: a string is split into
"runs" of consecutive characters that share a covering font, so a headline can
mix a display font with a fallback font for a few missing accents without
breaking measurement.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
from typing import Dict, List, Optional, Tuple
from PIL import ImageFont

from .fonts import FontRegistry

Run = Tuple[str, str]  # (font_file, substring)


@lru_cache(maxsize=256)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def get_font(reg: FontRegistry, file: str, size: int) -> ImageFont.FreeTypeFont:
    path = reg.find_path(file)
    if not path:
        raise FileNotFoundError(f"font not found in fonts_dir: {file}")
    return _font(path, int(size))


def build_runs(reg: FontRegistry, text: str, primary: str,
               language: str) -> List[Run]:
    """Split `text` into runs; each run uses one font that covers its chars."""
    runs: List[Run] = []
    for ch in text:
        if ch == " " or reg.covers(primary, ch):
            font_file = primary
        else:
            font_file = reg.fallback_font_for_chars(ch, language) or primary
        if runs and runs[-1][0] == font_file:
            runs[-1] = (font_file, runs[-1][1] + ch)
        else:
            runs.append((font_file, ch))
    return runs


def line_width(reg: FontRegistry, text: str, primary: str, language: str,
               size: int, ls_px: float) -> float:
    if not text:
        return 0.0
    runs = build_runs(reg, text, primary, language)
    total = 0.0
    for file, sub in runs:
        f = get_font(reg, file, size)
        total += f.getlength(sub)
    total += ls_px * max(0, len(text) - 1)
    return total


def line_metrics(reg: FontRegistry, primary: str,
                 size: int) -> Tuple[int, int]:
    """(ascent, descent) of the primary font at `size`."""
    f = get_font(reg, primary, size)
    return f.getmetrics()  # (ascent, descent)


SEPARATORS = ("·", "•", "|", "-", "–", "—", "/")


def _phrase_break(reg: FontRegistry, line: str, primary: str, language: str,
                  size: int, ls_px: float, max_w: float) -> Tuple[str, str]:
    """Split an overflowing line at the last phrase separator ("·", "|", a
    trailing comma...) if that still leaves >= a quarter line, so "17:00 · Chủ
    Nhật 12/10" breaks as "17:00 ·" / "Chủ Nhật 12/10" instead of mid-date.
    The separator stays at the end of the first line (text is never lost)."""
    toks = line.split(" ")
    for j in range(len(toks) - 2, 0, -1):
        t = toks[j]
        if t in SEPARATORS or t.endswith(",") or t.endswith(";"):
            head = " ".join(toks[:j + 1])
            if line_width(reg, head, primary, language, size, ls_px) \
                    >= 0.25 * max_w:
                return head, " ".join(toks[j + 1:])
    return line, ""


def hint_cuts(text: str):
    """Word indices after which the writer marked a line break ("\n"), or
    None. "Tiệc Sinh Nhật\nBé Bảo Ngọc" -> {3}: never "Nhật Bé"."""
    if "\n" not in text:
        return None
    cuts, n = set(), 0
    parts = [p.split() for p in text.split("\n")]
    for p in parts[:-1]:
        n += len(p)
        if p:
            cuts.add(n)
    return cuts or None


def flat_text(text: str) -> str:
    return " ".join(text.split())


def _is_sep(tok: str) -> bool:
    return tok in SEPARATORS or tok.endswith(",") or tok.endswith(";")


def _balance(reg: FontRegistry, words: List[str], k: int, primary: str,
             language: str, size: int, ls_px: float,
             max_w: float, allowed=None) -> Optional[List[str]]:
    """Re-break `words` into the SAME number of lines k, evenly.

    Like CSS `text-wrap: balance`: minimise the longest line and penalise
    a lone word on a line. Text with phrase separators may only break right
    after one. Returns None if nothing fits max_w.
    """
    n = len(words)
    if n <= k or n > 30:
        return None
    width_cache = {}

    def w(i, j):
        if (i, j) not in width_cache:
            width_cache[(i, j)] = line_width(reg, " ".join(words[i:j]),
                                             primary, language, size, ls_px)
        return width_cache[(i, j)]

    has_sep = any(_is_sep(t) for t in words[:-1])
    best, best_score = None, None
    for cuts in combinations(range(1, n), k - 1):
        if allowed is not None:
            if not all(c in allowed for c in cuts):
                continue          # break only where the writer marked "\n"
        elif has_sep and not all(_is_sep(words[c - 1]) for c in cuts):
            # phrase text ("17:00 · Chủ Nhật 12/10"): only break after a
            # separator; if that cannot fit, return None so the caller
            # shrinks the size instead of cutting through a date
            continue
        bounds = (0,) + cuts + (n,)
        widths = [w(bounds[i], bounds[i + 1]) for i in range(k)]
        if max(widths) > max_w:
            continue
        score = max(widths)
        score += 0.10 * max_w * sum(1 for i in range(k)
                                    if bounds[i + 1] - bounds[i] == 1)
        if best_score is None or score < best_score:
            best, best_score = bounds, score
    if best is None:
        return None
    return [" ".join(words[best[i]:best[i + 1]]) for i in range(k)]


def wrap_words(reg: FontRegistry, text: str, primary: str, language: str,
               size: int, ls_px: float, max_w: float,
               max_lines: int) -> List[str]:
    """Greedy word wrap at whitespace; avoids a lone word on the last line.

    Always returns ALL words: the result may have more than `max_lines` lines,
    which the caller treats as "does not fit" (and shrinks the size) instead
    of silently dropping the tail of the text.
    """
    allowed = hint_cuts(text)
    words = text.split()
    if not words:
        return [text]
    if allowed is not None:
        # explicit breaks: the fewest lines that fit, cutting only at hints
        for k in range(1, min(max_lines, len(allowed) + 1) + 1):
            if k == 1:
                flat = " ".join(words)
                if line_width(reg, flat, primary, language, size, ls_px) <= max_w:
                    return [flat]
                continue
            bal = _balance(reg, words, k, primary, language, size, ls_px,
                           max_w, allowed)
            if bal:
                return bal
        # the marked lines do not fit at this size: return them anyway so the
        # caller SHRINKS the size instead of breaking inside a phrase / name
        segs = [" ".join(p.split()) for p in text.split(chr(10)) if p.split()]
        if len(segs) <= max_lines:
            return segs
    lines: List[str] = []
    cur = ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if line_width(reg, trial, primary, language, size, ls_px) <= max_w \
                or not cur:
            cur = trial
        else:
            head, rest = _phrase_break(reg, cur, primary, language, size,
                                       ls_px, max_w)
            lines.append(head)
            cur = (rest + " " + w) if rest else w
    if cur:
        lines.append(cur)
    if 2 <= len(lines) <= 4:
        bal = _balance(reg, words, len(lines), primary, language, size, ls_px,
                       max_w)
        if bal:
            return bal
    # avoid orphan: single short word alone on the last line
    if len(lines) >= 2 and len(lines[-1].split()) == 1:
        prev = lines[-2].split()
        if len(prev) > 1:
            moved = prev.pop()
            merged = lines[-1]
            lines[-2] = " ".join(prev)
            lines[-1] = moved + " " + merged
    return lines


def ink_pads(reg: FontRegistry, lines: List[str], primary: str, size: int,
             lh: float) -> Tuple[int, int]:
    """(pad_top, pad_bottom): ink that sticks out of the line boxes.

    Stacked Vietnamese diacritics (Ẩ, Ễ, Ỗ...) rise above the font ascender,
    so the first line's ink can extend above the block box and collide with
    the block above. Measure the real glyph bbox and pad the block for it.
    """
    if not lines:
        return 0, 0
    f = get_font(reg, primary, size)
    top = f.getbbox(lines[0], anchor="la")[1]       # < 0 = above ascender
    bottom = f.getbbox(lines[-1], anchor="la")[3]   # relative to line top
    return max(0, int(round(-top))), max(0, int(round(bottom - lh)))


EXTRA_LINE_COST = 0.82          # score = size x this ^ (lines - 1)
FEWER_LINES_MAX_SHRINK = 0.78   # look for fewer lines down to this x first fit


def fit_block(reg: FontRegistry, text: str, primary: str, language: str,
              max_w: float, max_h: float, target_size: int, ls_frac: float,
              allow_multiline: bool = True, max_lines: int = 3,
              line_height: float = 1.15, min_size: int = 14) -> Dict:
    """Shrink size / wrap so the text fits (max_w, max_h). Returns a layout.

    Strategy: at the target size, try single line; if too wide and multiline is
    allowed, wrap; if still too tall/wide, shrink the size and retry. Returns
    dict(lines, size_px, w, h, line_height, ls_px, runs_primary=primary).
    """
    size = int(target_size)
    hinted = text
    text = flat_text(text)
    # a short multi-word line that ALMOST fits reads better slightly smaller
    # on one line than broken in two ("Minh & Lan" vs "Minh / & Lan")
    if allow_multiline and len(text.split()) > 1:
        s1 = size
        while s1 >= max(min_size, int(size * 0.85)):
            ls1 = ls_frac * s1
            w1 = line_width(reg, text, primary, language, s1, ls1)
            if w1 <= max_w:
                asc, desc = line_metrics(reg, primary, s1)
                lh = (asc + desc) * line_height
                pt, pb = ink_pads(reg, [text], primary, s1, lh)
                if pt + lh + pb <= max_h:
                    return {"lines": [text], "size_px": s1, "w": w1,
                            "h": pt + lh + pb, "line_height": line_height,
                            "ls_px": ls1, "ascent": asc, "descent": desc,
                            "pad_top": pt, "pad_bottom": pb}
            s1 -= max(1, int(s1 * 0.02))
    first, best, best_score = None, None, None
    while size >= min_size:
        ls_px = ls_frac * size
        if allow_multiline:
            lines = wrap_words(reg, hinted, primary, language, size, ls_px,
                               max_w, max_lines)
        else:
            lines = [text]
        widths = [line_width(reg, ln, primary, language, size, ls_px)
                  for ln in lines]
        asc, desc = line_metrics(reg, primary, size)
        lh = (asc + desc) * line_height
        pad_top, pad_bottom = ink_pads(reg, lines, primary, size, lh)
        block_h = pad_top + lh * len(lines) + pad_bottom
        block_w = max(widths) if widths else 0
        if block_w <= max_w and block_h <= max_h and len(lines) <= max_lines:
            fit = {
                "lines": lines, "size_px": size, "w": block_w, "h": block_h,
                "line_height": line_height, "ls_px": ls_px,
                "ascent": asc, "descent": desc,
                "pad_top": pad_top, "pad_bottom": pad_bottom,
            }
            # each extra line costs like ~18% of size: 2 lines at 0.85x beat 3
            score = size * (EXTRA_LINE_COST ** (len(lines) - 1))
            if best_score is None or score > best_score:
                best, best_score = fit, score
            if first is None:
                first = size
            if len(lines) == 1 or size < first * FEWER_LINES_MAX_SHRINK:
                return best
        elif first is not None and size < first * FEWER_LINES_MAX_SHRINK:
            return best
        size = int(size * 0.94) - 1
    if best is not None:
        return best
    # fell through: return smallest attempt (clamped) so we never crash
    size = min_size
    ls_px = ls_frac * size
    lines = wrap_words(reg, text, primary, language, size, ls_px, max_w,
                       max_lines) if allow_multiline else [text]
    widths = [line_width(reg, ln, primary, language, size, ls_px)
              for ln in lines]
    asc, desc = line_metrics(reg, primary, size)
    lh = (asc + desc) * line_height
    pad_top, pad_bottom = ink_pads(reg, lines, primary, size, lh)
    return {"lines": lines, "size_px": size, "w": max(widths) if widths else 0,
            "h": pad_top + lh * len(lines) + pad_bottom,
            "line_height": line_height, "ls_px": ls_px,
            "ascent": asc, "descent": desc,
            "pad_top": pad_top, "pad_bottom": pad_bottom, "overflow": True}
