# -*- coding: utf-8 -*-
"""Robust parsing of the LLM output + background-prompt cleaning (Node 1).

Never raises: on total failure it returns a safe fallback spec plus a warning,
so the ComfyUI graph keeps running.
"""
from __future__ import annotations
import re
import json
from typing import Dict, List, Tuple

from .schema import DEFAULTS, ROLES

# words that make the image model draw letters -> swap for a neutral noun
_SWAP = ["poster", "flyer", "banner", "card", "invitation", "sign", "signage",
         "menu", "label", "logo", "cover", "headline", "title", "billboard",
         "brochure", "advertisement", "storefront", "quote", "slogan",
         "caption"]
# words to strip together with 0-2 neighbouring words (phrase, not sentence)
_STRIP = ["text", "texts", "letter", "letters", "lettering", "word", "words",
          "typography", "font", "fonts", "writing", "written", "calligraphy"]

_SWAP_RE = re.compile(r"\b(" + "|".join(_SWAP) + r")s?\b", re.I)
_STRIP_RE = re.compile(
    r"(?:\b(?:with|and|featuring|including|showing|of|no)\b\s+)?"
    r"(?:\w+\s+){0,2}\b(?:" + "|".join(_STRIP) + r")\b(?:\s+\w+){0,2}", re.I)


def clean_background_prompt(s: str) -> str:
    """Remove text-triggering phrases only (keep the rest of the sentence)."""
    s = s or ""
    s = _SWAP_RE.sub("scene", s)
    s = _STRIP_RE.sub("", s)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s*,+", ", ", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,.").strip()
    if s:
        s = s[0].upper() + s[1:]
        if not s.endswith("."):
            s += "."
    return s


# letters that only Vietnamese uses among Latin scripts (đ, ă, ơ, ư and the
# dot-below / hook-above tone marks)
_VI_CHARS = set("đĐăĂơƠưƯ"
                "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
                "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ")


def looks_vietnamese(texts: List[Dict]) -> bool:
    return any(ch in _VI_CHARS for t in texts for ch in t.get("text", ""))


def _extract_object(raw: str) -> str:
    """Return the first balanced {...} object, ignoring markdown fences."""
    raw = re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "")
    start = raw.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return raw[start:i + 1]
    return raw[start:]


def _repair(text: str) -> str:
    text = re.sub(r",\s*([}\]])", r"\1", text)      # trailing commas
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text


def _coerce_texts(items) -> List[Dict]:
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        txt = str(it.get("text", "")).strip()
        if not txt:
            continue
        role = it.get("role", "body")
        if role not in ROLES:
            role = "body"
        out.append({"role": role, "text": txt,
                    "priority": int(it.get("priority", 5) or 5)})
    return out


def parse_llm_output(raw: str) -> Tuple[Dict, str, List[str]]:
    """Return (design_spec, background_prompt_clean, warnings)."""
    warnings: List[str] = []
    obj = _extract_object(raw or "")
    data = None
    for attempt in (obj, _repair(obj)):
        if not attempt:
            continue
        try:
            data = json.loads(attempt)
            break
        except Exception as e:  # noqa: BLE001
            last = str(e)
    if not isinstance(data, dict):
        warnings.append(f"JSON parse failed ({last if obj else 'no object'}); "
                        "using fallback spec")
        data = {}

    spec = {
        "language": str(data.get("language") or DEFAULTS["language"]),
        "mood": str(data.get("mood") or DEFAULTS["mood"]),
        "design_type": str(data.get("design_type") or DEFAULTS["design_type"]),
        "composition_hint": str(data.get("composition_hint") or "auto"),
        "background_prompt": str(data.get("background_prompt")
                                 or DEFAULTS["background_prompt"]),
        "background_color": str(data.get("background_color")
                                or DEFAULTS["background_color"]),
        "texts": _coerce_texts(data.get("texts")),
    }
    if not spec["texts"]:
        warnings.append("no valid texts in LLM output")
    if spec["language"] != "vietnamese" and looks_vietnamese(spec["texts"]):
        warnings.append(f"language {spec['language']!r} but texts are "
                        "Vietnamese; using 'vietnamese'")
        spec["language"] = "vietnamese"
    bg_clean = clean_background_prompt(spec["background_prompt"])
    return spec, bg_clean, warnings
