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
_DEFAULT_TAGS = os.path.join(os.path.dirname(_HERE), "font_tags.json")

# Personalities that fit each mood (font_tags.json). A font qualifies when it
# has at least one of these tags.
MOOD_TAGS = {
    "vintage": {"vintage", "retro", "bold", "blackletter", "grunge",
                "elegant", "clean"},
    "elegant": {"elegant", "clean", "thin", "fancy"},
    "playful": {"playful", "handwritten", "bold", "festive", "retro"},
    "minimal": {"clean", "thin", "outline"},
    "bold": {"bold", "tech", "retro", "outline", "grunge", "clean"},
    "festive": {"festive", "playful", "bold", "retro", "handwritten",
                "elegant"},
}
DEFAULT_MOOD_TAGS = {"clean", "bold"}
# personalities that clash with a mood even when another tag matches
MOOD_BANNED = {"bold": {"thin"}, "vintage": {"thin"}, "festive": {"thin"}}
# never used for small text: hard to read at 30-40 px
SMALL_TEXT_BANNED = {"thin", "fancy", "grunge", "outline", "horror",
                     "blackletter"}
SMALL_ROLES = {"body", "detail", "note"}
# preference by position in group_pref(role, mood): 1st group x3, 2nd x2 ...
GROUP_RANK_WEIGHT = (3.0, 2.0, 1.0)
# below this many mood-fitting fonts, off-mood fonts join at a lower weight
MIN_MOOD_POOL = 4
OFF_MOOD_WEIGHT = 0.35


# parsed coverage per font file, shared by every registry in the process
# (parsing all outlines takes seconds; fonts do not change while running)
_CMAP_BY_PATH: Dict[str, frozenset] = {}


def _blank_glyphs(ft) -> Set[str]:
    """Glyph names that draw nothing (no contours / empty charstring)."""
    blank: Set[str] = set()
    try:
        if "glyf" in ft:
            glyf = ft["glyf"]
            for name in ft.getGlyphOrder():
                g = glyf[name]
                if g.numberOfContours == 0:
                    blank.add(name)
        elif "CFF " in ft:
            cff = ft["CFF "].cff
            cs = cff[cff.fontNames[0]].CharStrings
            for name in cs.keys():
                if cs[name].calcBounds(cs) is None:
                    blank.add(name)
    except Exception:  # pragma: no cover - odd font: trust the cmap
        return set()
    return blank


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
        self._load_tags(_DEFAULT_TAGS)

    def _load_tags(self, path: str) -> None:
        """Curated personalities / exclusions (optional file)."""
        self.tag_defaults: Dict[str, List[str]] = {}
        self.font_tags: Dict[str, Set[str]] = {}
        self.excluded: Set[str] = set()
        self.no_digits: Set[str] = set()
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                t = json.load(f)
        except Exception as e:  # pragma: no cover - broken file
            self.warnings.append(f"font_tags.json unreadable: {e}")
            return
        self.tag_defaults = t.get("group_defaults", {})
        self.font_tags = {k: set(v) for k, v in t.get("tags", {}).items()}
        self.excluded = set(t.get("exclude", []))
        self.no_digits = set(t.get("no_digits", []))

    def tags_of(self, file: str) -> Set[str]:
        if file in self.font_tags:
            return self.font_tags[file]
        grp = self.by_file.get(file, {}).get("group", "uncategorized")
        return set(self.tag_defaults.get(grp, ["clean"]))

    def usable_for(self, file: str, text: str, role: str) -> bool:
        """Curated exclusions + digit safety + small-text legibility."""
        if file in self.excluded:
            return False
        if file in self.no_digits and any(ch.isdigit() for ch in text):
            return False
        if role in SMALL_ROLES and self.tags_of(file) & SMALL_TEXT_BANNED:
            return False
        return True

    def choices(self, text: str, language: str, role: str,
                mood: str) -> List[Tuple[str, float]]:
        """Weighted fonts that suit (role, mood) AND can render `text`.

        Replaces "the first covering font": every font of the preferred style
        groups qualifies, weighted by the group's rank, filtered by the
        curated mood personalities; when fewer than MIN_MOOD_POOL fonts fit
        the mood (e.g. only 5 Vietnamese fonts exist), off-mood fonts join at
        OFF_MOOD_WEIGHT so the roles can still use different faces."""
        allowed = MOOD_TAGS.get(mood, DEFAULT_MOOD_TAGS)
        banned = MOOD_BANNED.get(mood, set())
        # rank only groups that HAVE fonts for this language (an empty
        # "display" group must not push "sans" out of the top 3)
        groups = [g for g in group_pref(role, mood)
                  if self.candidates(language, g)][:len(GROUP_RANK_WEIGHT)]
        fit: List[Tuple[str, float]] = []
        rest: List[Tuple[str, float]] = []
        for rank, grp in enumerate(groups):
            for e in self.candidates(language, grp):
                f = e["file"]
                if not self.usable_for(f, text, role) or \
                        not self.covers(f, text):
                    continue
                tags = self.tags_of(f)
                w = GROUP_RANK_WEIGHT[rank]
                if tags & allowed and not tags & banned:
                    fit.append((f, w))
                else:
                    rest.append((f, w * OFF_MOOD_WEIGHT))
        # a tiny mood pool (5 Vietnamese fonts -> often 1 fits) would give
        # every role the same face: widen it with the off-mood fonts
        if len(fit) < MIN_MOOD_POOL:
            return fit + rest
        return fit

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
        if path and path in _CMAP_BY_PATH:
            cps = _CMAP_BY_PATH[path]
        elif path:
            try:
                if path.lower().endswith(".ttc"):
                    ft = TTCollection(path).fonts[0]
                else:
                    ft = TTFont(path, fontNumber=0)
                best = ft.getBestCmap()
                blank = _blank_glyphs(ft)
                # a mapped-but-empty glyph renders as nothing ("12/10   5 PM"
                # with the dots gone), so it does not count as covered
                cps = frozenset(cp for cp, g in best.items()
                                if g not in blank or chr(cp).isspace())
                _CMAP_BY_PATH[path] = cps
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
