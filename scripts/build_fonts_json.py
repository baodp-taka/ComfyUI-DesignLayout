# -*- coding: utf-8 -*-
"""Scan font folder(s) and emit fonts.json (style group + language coverage).

Run once whenever the font set changes:

    python scripts/build_fonts_json.py \
        --root "C:/TAKA/PosterMaker/asset_resource/src/main/assets/fonts" \
        --root "C:/TAKA/AI-Logo-Maker/on_demand_asset_pack/src/main/assets/fonts" \
        --out fonts.json

At runtime the node only reads fonts.json; a font's actual glyph coverage is
re-checked live per string (see designlayout/fonts.py), so fonts.json is just a
fast index for style + language filtering.
"""
import os
import re
import json
import glob
import argparse
from fontTools.ttLib import TTFont, TTCollection

# A font "supports" a language only if it covers EVERY probe char.
PROBES = {
    "latin": "AZaz0.,!?&%-",
    "latin_ext": "éèêëàâäñüöçßœ",
    "vietnamese": ("ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
                   "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"),
    "cjk_han": "你好中文汉字愛",
    "japanese_kana": "あいうえおカタカナ",
    "korean_hangul": "한국어글밥",
    "cyrillic": "ПриветДЖЯб",
    "greek": "ΩαβΓπΣ",
    "arabic": "ابجدهو",
    "thai": "กขคงจฉ",
    "hebrew": "אבגדהו",
}

AILOGO_CAT = {
    "decorative": "decorative", "script": "script", "serif": "serif",
    "san serif": "sans", "sans serif": "sans", "other": "other",
}
NAME_HINTS = [
    ("script", "script"), ("hand", "script"), ("brush", "script"),
    ("marker", "script"), ("signature", "script"), ("calli", "script"),
    ("vibes", "script"), ("lobster", "script"), ("pacifico", "script"),
    ("slab", "serif"), ("didot", "serif"), ("serif", "serif"),
    ("gothic", "sans"), ("sans", "sans"), ("grotesk", "sans"),
    ("roboto", "sans"), ("proxima", "sans"), ("museo sans", "sans"),
    ("bebas", "display"), ("condensed", "display"), ("captain", "display"),
    ("black", "display"), ("display", "display"), ("liberator", "display"),
    ("ostrich", "display"), ("venera", "display"),
    ("monoton", "decorative"), ("amatic", "decorative"),
]


def read_font(path):
    if path.lower().endswith(".ttc"):
        f = TTCollection(path).fonts[0]
    else:
        f = TTFont(path, fontNumber=0)
    cps = set(f.getBestCmap().keys())
    family = ""
    try:
        for rec in f["name"].names:
            if rec.nameID == 1:
                family = rec.toUnicode()
                break
    except Exception:
        pass
    return cps, family


def langs_for(cps):
    return [lang for lang, probe in PROBES.items()
            if all(ord(ch) in cps for ch in probe)]


def category(path):
    parent = os.path.basename(os.path.dirname(path)).lower()
    if parent in AILOGO_CAT:
        return AILOGO_CAT[parent]
    low = os.path.basename(path).lower()
    for kw, cat in NAME_HINTS:
        if kw in low:
            return cat
    return "uncategorized"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True,
                    help="font folder (repeatable)")
    ap.add_argument("--out", default="fonts.json")
    args = ap.parse_args()

    by_file = {}
    for root in args.root:
        for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if not path.lower().endswith((".ttf", ".otf", ".ttc")):
                continue
            name = os.path.basename(path)
            if name in by_file:
                # duplicate filename: first root wins, except that a copy in
                # a category folder (Serif/, Script/...) beats an
                # uncategorized one so the style group is not lost
                prev = by_file[name]
                if not (prev.get("group") == "uncategorized"
                        and category(path) != "uncategorized"):
                    continue
            try:
                cps, family = read_font(path)
            except Exception as e:
                by_file[name] = {"file": name, "error": str(e)[:60]}
                continue
            langs = langs_for(cps)
            by_file[name] = {
                "file": name,
                "family": family or os.path.splitext(name)[0],
                "group": category(path),
                "languages": langs,
            }
    fonts = [e for e in by_file.values() if "error" not in e]
    fonts.sort(key=lambda e: (e["group"], e["file"].lower()))
    out = {
        "version": 1,
        "style_groups": ["display", "serif", "sans", "script",
                         "decorative", "other", "uncategorized"],
        "fonts": fonts,
    }
    dest = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.out)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    from collections import Counter
    print("fonts:", len(fonts), "| errors:",
          len(by_file) - len(fonts))
    print("by group:", dict(Counter(e["group"] for e in fonts)))
    langc = Counter()
    for e in fonts:
        for lg in e["languages"]:
            langc[lg] += 1
    print("by language:", dict(langc))
    print("wrote:", dest)


if __name__ == "__main__":
    main()
