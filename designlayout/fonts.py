# -*- coding: utf-8 -*-
"""Font registry: language/style filtering + live glyph-coverage fallback.

fonts.json is a fast index (style group + languages). Real glyph coverage is
ALWAYS re-verified against the actual string via the font cmap, so a font is
never used for a character it cannot render.
"""
from __future__ import annotations
import os
import json
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Set
from fontTools.ttLib import TTFont, TTCollection

from .schema import group_pref

_HERE = os.path.dirname(__file__)
_DEFAULT_JSON = os.path.join(os.path.dirname(_HERE), "fonts.json")


class FontRegistry:
    """Loads fonts.json and resolves real font files inside a fonts_dir."""

    def __init__(self, fonts_dir: str,
                 fonts_json: Optional[str] = None) -> None:
        self.fonts_dir = fonts_dir
        self.json_path = fonts_json or _DEFAULT_JSON
        self.warnings: List[str] = []
        with open(self.json_path, encoding="utf-8") as f:
            data = json.load(f)
        self.entries: List[Dict] = [e for e in data.get("fonts", [])
                                    if "error" not in e]
        self.by_file: Dict[str, Dict] = {e["file"]: e for e in self.entries}
        self._path_cache: Dict[str, Optional[str]] = {}
        self._cmap_cache: Dict[str, frozenset] = {}

    # ---- file / glyph resolution -------------------------------------------
    def find_path(self, file: str) -> Optional[str]:
        if file in self._path_cache:
            return self._path_cache[file]
        found = None
        direct = os.path.join(self.fonts_dir, file)
        if os.path.isfile(direct):
            found = direct
        else:
            for dp, _, files in os.walk(self.fonts_dir):
                if file in files:
                    found = os.path.join(dp, file)
                    break
        self._path_cache[file] = found
        return found

    def cmap(self, file: str) -> frozenset:
        if file in self._cmap_cache:
            return self._cmap_cache[file]
        path = self.find_path(file)
        cps: frozenset = frozenset()
        if path:
            try:
                if path.lower().endswith(".ttc"):
                    ft = TTCollection(path).fonts[0]
                else:
                    ft = TTFont(path, fontNumber=0)
                cps = frozenset(ft.getBestCmap().keys())
            except Exception as e:  # pragma: no cover - corrupt font
                self.warnings.append(f"cmap failed for {file}: {e}")
        self._cmap_cache[file] = cps
        return cps

    def covers(self, file: str, text: str) -> bool:
        cps = self.cmap(file)
        if not cps:
            return False
        return all(ch in " \t\n" or ord(ch) in cps for ch in text)

    # ---- selection ----------------------------------------------------------
    def candidates(self, language: str,
                   group: Optional[str] = None) -> List[Dict]:
        out = []
        for e in self.entries:
            if language not in e.get("languages", []):
                continue
            if group and e.get("group") != group:
                continue
            if self.find_path(e["file"]) is None:
                continue  # not present in this fonts_dir
            out.append(e)
        return out

    def is_strict(self, language: str) -> bool:
        """Scripts with their own letters (vietnamese, cyrillic, greek...) must
        use a font that supports the WHOLE language, never a Latin font that
        happens to cover the few characters of one string."""
        return language not in ("latin", "") and bool(self.candidates(language))

    def pick(self, text: str, language: str, role: str, mood: str,
             exclude: Optional[Set[str]] = None) -> Tuple[str, bool]:
        """Return (font_file, covered). Guarantees a usable file.

        Order: preferred style groups for (role, mood) whose font covers the
        text -> any font of the language that covers -> any font that covers ->
        first available font (covered=False, caller warns / uses fallback).
        For a strict language (see is_strict) only fonts that support the
        language are ever returned.
        """
        exclude = exclude or set()
        strict = self.is_strict(language)

        def first_covering(entries):
            for e in entries:
                if e["file"] in exclude:
                    continue
                if self.covers(e["file"], text):
                    return e["file"]
            return None

        # 1) style-group preference within the language
        for grp in group_pref(role, mood):
            hit = first_covering(self.candidates(language, grp))
            if hit:
                return hit, True
        # 2) any font of this language that covers
        hit = first_covering(self.candidates(language))
        if hit:
            return hit, True
        pool = (self.candidates(language) if strict else
                [e for e in self.entries if self.find_path(e["file"])])
        # 3) any font at all that covers (cross-language, e.g. symbols)
        if not strict:
            hit = first_covering(pool)
            if hit:
                return hit, True
        # 4) give up: return first available; caller handles per-glyph fallback
        for e in pool:
            self.warnings.append(
                f"no font covers {text!r} ({language}); using {e['file']}")
            return e["file"], False
        raise RuntimeError("no usable font found in fonts_dir")

    def fallback_font_for_chars(self, chars: str,
                                language: str) -> Optional[str]:
        """A font that covers `chars` (for per-glyph fallback in measure).

        Strict languages look in their own fonts first and only borrow from
        other fonts for non-letters (symbols, punctuation)."""
        for e in self.candidates(language):
            if self.covers(e["file"], chars):
                return e["file"]
        if self.is_strict(language) and any(c.isalpha() for c in chars):
            return None
        for e in self.entries:
            if self.find_path(e["file"]) and self.covers(e["file"], chars):
                return e["file"]
        return None
