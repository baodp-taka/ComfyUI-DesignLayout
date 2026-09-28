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
from typing import Dict, List, Tuple
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


def wrap_words(reg: FontRegistry, text: str, primary: str, language: str,
               size: int, ls_px: float, max_w: float,
               max_lines: int) -> List[str]:
    """Greedy word wrap at whitespace; avoids a lone word on the last line."""
    words = text.split()
    if not words:
        return [text]
    lines: List[str] = []
    cur = ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if line_width(reg, trial, primary, language, size, ls_px) <= max_w \
                or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    # avoid orphan: single short word alone on the last line
    if len(lines) >= 2 and len(lines[-1].split()) == 1:
        prev = lines[-2].split()
        if len(prev) > 1:
            moved = prev.pop()
            merged = lines[-1]
            lines[-2] = " ".join(prev)
            lines[-1] = moved + " " + merged
    return lines


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
    while size >= min_size:
        ls_px = ls_frac * size
        if allow_multiline:
            lines = wrap_words(reg, text, primary, language, size, ls_px,
                               max_w, max_lines)
        else:
            lines = [text]
        widths = [line_width(reg, ln, primary, language, size, ls_px)
                  for ln in lines]
        asc, desc = line_metrics(reg, primary, size)
        lh = (asc + desc) * line_height
        block_h = lh * len(lines)
        block_w = max(widths) if widths else 0
        if block_w <= max_w and block_h <= max_h:
            return {
                "lines": lines, "size_px": size, "w": block_w, "h": block_h,
                "line_height": line_height, "ls_px": ls_px,
                "ascent": asc, "descent": desc,
            }
        size = int(size * 0.94) - 1
    # fell through: return smallest attempt (clamped) so we never crash
    size = min_size
    ls_px = ls_frac * size
    lines = wrap_words(reg, text, primary, language, size, ls_px, max_w,
                       max_lines) if allow_multiline else [text]
    widths = [line_width(reg, ln, primary, language, size, ls_px)
              for ln in lines]
    asc, desc = line_metrics(reg, primary, size)
    return {"lines": lines, "size_px": size, "w": max(widths) if widths else 0,
            "h": (asc + desc) * line_height * len(lines),
            "line_height": line_height, "ls_px": ls_px,
            "ascent": asc, "descent": desc, "overflow": True}
