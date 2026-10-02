# -*- coding: utf-8 -*-
"""Invitation card, text first.

The plan JSON (written by an LLM in ONE call) -> code sets the text on a paper
card in the middle of the canvas -> Qwen-Image-Edit inpaints the scene around
it with a soft mask -> the glyphs are put back pixel-exact.

    prepare(reg, plan, request, seed, w, h)
        -> image  : the card (paper colour + text) on a mottled theme backdrop
           mask   : repaint strength (0 = keep, 1 = repaint) for
                    SetLatentNoiseMask + DifferentialDiffusion
           alpha  : the glyphs, for composite()
           prompt : the scene prompt for TextEncodeQwenImageEditPlus
           report : what was chosen / fixed
    composite(generated, image, alpha) -> the final card

Mask (per pixel): the text and a thin halo are locked; inside the card the
strength ramps from 0.35 near the text to 0.75, the card's edge band is 0.9
(decorations may cross it), outside the card 1.
"""
from __future__ import annotations

import colorsys
import datetime as _dt
import difflib
import json
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from . import invite
from .colors import fit_contrast, hex_to_rgb, rgb_to_hex
from .fonts import FontRegistry

# ---- geometry (fractions of the canvas width W unless said otherwise) --------
CARD_BOX = (.17, .17, .83, .85)   # where the text column sits (x0, y0, x1, y1 of W, H)
HALO = 0.045                      # card margin around the text
CARD_MARGIN = 0.04                # card never closer than this to the picture edge
LOCK = 0.012                      # band hugging the text that is never repainted
RAMP = 0.10                       # inside the card: 0.35 -> 0.75 over this distance
EDGE = 0.025                      # the card's edge band, repainted at 0.9
INSIDE_NEAR, INSIDE_FAR, EDGE_STRENGTH = 0.35, 0.75, 0.90

# ---- what the code may choose from (the LLM ranks them) ----------------------
COMPOSITIONS = {
    "classic":       {"align": "center", "date": "block", "order": ("header", "date", "place")},
    "modern_line":   {"align": "center", "date": "line", "order": ("header", "date", "place")},
    "hero_date":     {"align": "center", "date": "hero", "order": ("header", "date", "place")},
    "save_the_date": {"align": "center", "date": "hero", "order": ("date", "header", "place")},
    "editorial":     {"align": "left", "date": "line", "order": ("header", "date", "place")},
}
SHAPES = ("arch", "rounded", "ticket")
TOP_COMPOSITIONS, TOP_SHAPES = 3, 2   # the seed picks inside the LLM's top N

SCENE_SUFFIX = ("The paper card stands in the middle of the picture. Keep the text on the paper card "
                "exactly as it is; finish the invitation card around it. Do not add any other text, "
                "letters, numbers or symbols anywhere, and no other card.")
DEFAULT_PAPER = (250, 248, 243)

# sentences of an OLD worked example in the prompt: never let them through
EXAMPLE_LINES = ["Together with their families", "Invite you to celebrate their wedding",
                 "Dinner and dancing to follow", "The Old Mill", "12 River Lane, Bath"]


# ---- plan parsing / fixing ---------------------------------------------------
def parse_plan(plan) -> Dict:
    """A dict, or the LLM's text (code fences / chatter around the JSON ok)."""
    if isinstance(plan, dict):
        return json.loads(json.dumps(plan))
    t = re.sub(r"```[a-z]*", "", str(plan or "")).replace("```", "")
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in the plan")
    return json.loads(t[a:b + 1])


WEEKDAYS_VI_RE = r"(thứ\s*(hai|ba|tư|năm|sáu|bảy)|chủ\s*nhật)"
WEEKDAYS_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


def vi_date_from_request(request: str) -> Dict[str, str]:
    """Weekday, day, month, year and time read from a Vietnamese request
    (d/m/yyyy is day/month/year). A missing weekday is computed from the date."""
    out: Dict[str, str] = {}
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b", request)
    if m:
        out["day"], out["month"] = str(int(m.group(1))), f"Tháng {int(m.group(2))}"
        if m.group(3):
            out["year"] = m.group(3)
    w = re.search(WEEKDAYS_VI_RE, request, re.I)
    if w:
        out["weekday"] = " ".join(x.capitalize() for x in w.group(1).split())
    elif m and m.group(3):
        try:
            d = _dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            out["weekday"] = WEEKDAYS_VI[d.weekday()]
        except ValueError:
            pass
    t = re.search(r"\b(\d{1,2})[:h](\d{2})\b", request)
    if t:
        out["time"] = f"{int(t.group(1))}:{t.group(2)}"
    return out


def fix_plan(plan: Dict, request: str = "") -> Dict:
    """What code can fix without asking the LLM again: Vietnamese dates come
    from the request; a kicker that only repeats the names is dropped."""
    comps = plan.get("components") or []
    if plan.get("language") == "vietnamese" or re.search(WEEKDAYS_VI_RE, request, re.I):
        facts = vi_date_from_request(request)
        for c in comps:
            if c.get("type") == "date":
                c.update(facts)
    names = " ".join(str(c.get(f, "")) for c in comps if c.get("type") == "names"
                     for f in ("first", "second")).lower().split()
    out = []
    for c in comps:
        if c.get("type") == "text" and c.get("style") == "kicker":
            w = str(c.get("text", "")).lower().split()
            if w and sum(x.strip(",.!") in names for x in w) >= max(2, len(w) // 2):
                continue
        out.append(c)
    plan["components"] = out
    return plan


def _norm(t) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(t).lower()).split())


def drop_copied(comps: List[Dict], request: str) -> Tuple[List[Dict], List[str]]:
    """Drop lines copied from the prompt's example or left as a <placeholder>."""
    req = _norm(request)
    out, dropped = [], []
    for c in comps:
        vals = [str(v) for v in c.values() if isinstance(v, str)]
        if any("<" in v and ">" in v for v in vals):
            dropped.append(str(c)[:60])
            continue
        txt = c.get("text") if c.get("type") == "text" else None
        if txt:
            nt = _norm(txt)
            if nt not in req and any(difflib.SequenceMatcher(None, nt, _norm(e)).ratio() >= 0.8
                                     for e in EXAMPLE_LINES):
                dropped.append(txt)
                continue
        out.append(c)
    return out, dropped


# ---- natural line breaks (code; a 2B LLM ignores "|" marks) -----------------
BREAK_FIELDS = {"text": ("text",), "venue": ("name", "address")}
BREAK_MIN_WORDS = 6
BREAK_BEFORE = {"để", "và", "của", "cho", "tại", "với", "trong", "cùng", "vào", "khi", "từ", "đến", "mà",
                "for", "of", "to", "and", "with", "at", "in", "on", "as", "from", "our", "your", "this"}


def code_split(text: str) -> str:
    """Two balanced lines at the most natural word boundary: after a comma or
    before a connector, never between two Capitalized words (a name / place),
    never leaving one word alone. The layout only uses it when the line does
    not fit on one line."""
    w = text.split()
    total = len(text)
    natural, plain = [], []
    for i in range(2, len(w) - 1):
        left = " ".join(w[:i])
        if w[i - 1][:1].isupper() and w[i][:1].isupper() and not w[i - 1].endswith(","):
            continue
        cost = abs(len(left) - (total - len(left) - 1)) / total
        if w[i - 1].endswith((",", ";", ":")):
            natural.append((cost - 0.15, i))
        elif w[i].lower().strip(",.!") in BREAK_BEFORE:
            natural.append((cost, i))
        else:
            plain.append((cost, i))
    natural = [c for c in natural if c[0] <= 0.5]     # unless the lines get very uneven
    pick = min(natural or plain or [(0, None)])[1]
    if pick is None:
        return text
    return " ".join(w[:pick]) + "\n" + " ".join(w[pick:])


def apply_breaks(comps: List[Dict]) -> Tuple[List[Dict], List[str]]:
    out, log = [], []
    for c in comps:
        c = dict(c)
        for f, v in list(c.items()):
            if not isinstance(v, str):
                continue
            breakable = f in BREAK_FIELDS.get(c.get("type"), ())
            if "|" in v:
                parts = [q.strip() for q in v.split("|") if q.strip()]
                ok = breakable and len(parts) >= 2 and all(len(q.split()) >= 2 for q in parts)
                c[f] = "\n".join(parts) if ok else " ".join(parts)
            elif breakable and len(v.split()) >= BREAK_MIN_WORDS:
                c[f] = code_split(v)
                if "\n" in c[f]:
                    log.append(c[f].replace("\n", " / "))
        out.append(c)
    return out, log


# ---- layout choice / paper colour -------------------------------------------
def ranked(raw, valid) -> List[str]:
    out = [v for v in (raw if isinstance(raw, list) else []) if v in valid]
    return list(dict.fromkeys(out + [v for v in valid if v not in out]))


def pick_layout(plan: Dict, rng: random.Random, force: str = "") -> Tuple[str, str]:
    """(composition, card shape): the seed picks inside the LLM's ranking top."""
    comps = ranked(plan.get("composition_rank"), list(COMPOSITIONS))
    shapes = ranked(plan.get("card_rank"), list(SHAPES))
    comp = force if force in COMPOSITIONS else rng.choice(comps[:TOP_COMPOSITIONS])
    return comp, rng.choice(shapes[:TOP_SHAPES])


def paper_of(plan: Dict) -> Tuple[int, int, int]:
    """The LLM's paper colour kept printable: a light paper becomes a soft
    pastel (no neon), a deep one a rich dark tone."""
    c = str(plan.get("paper") or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", c):
        return DEFAULT_PAPER
    r, g, b = (int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if l >= 0.45:
        l, s = max(l, 0.88), min(s, 0.45)
    else:
        l, s = min(l, 0.22), min(s, 0.65)
    return tuple(int(round(v * 255)) for v in colorsys.hls_to_rgb(h, l, s))


def text_colors(plan: Dict, paper) -> Dict[str, str]:
    """ink / accent from the plan, always readable on THIS paper; a missing one
    gets a default for a light or a deep paper."""
    deep = sum(paper) / 3 < 110
    defaults = {"ink": "#F3EDE2" if deep else "#2B2420", "accent": "#C9A227"}
    out = {}
    for k, need in (("ink", 6.0), ("accent", 3.0 if deep else 2.2)):
        c = plan.get(k)
        c = c if isinstance(c, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", c.strip()) else defaults[k]
        out[k] = rgb_to_hex(fit_contrast(hex_to_rgb(c.strip()), paper, need))
    return out


def order_boxes(box, order, layout, groups):
    """plan_boxes with the groups in any order (save-the-date puts the date first)."""
    names = [g for g in order if g in groups]
    if tuple(names) == tuple(g for g in ("header", "date", "place") if g in groups):
        return invite.plan_boxes(box, layout, {"header": .5, "date": .22, "place": .28}, groups)
    share = {"header": .5, "date": .26, "place": .28}
    x0, y0, x1, y1 = box
    if layout == "airy":
        y0, y1 = y0 + int((y1 - y0) * 0.08), y1 - int((y1 - y0) * 0.08)
    g = int((y1 - y0) * 0.035)
    tot = sum(share[n] for n in names)
    out, y = {}, y0
    for n in names:
        h = int((y1 - y0 - g * (len(names) - 1)) * share[n] / tot)
        out[n] = (x0, y, x1, y + h)
        y += h + g
    return out


def scene_prompt(raw: Optional[str]) -> str:
    """The plan's scene, minus any clause about a blank card (code places the
    card), plus where the card stands and the keep-the-text rule."""
    s = re.sub(r"[^,.;]*\bblank\b[^,.;]*[,.;]?", "", raw or "")
    s = re.sub(r"\s+", " ", s).strip(" ,.;")
    return f"{s}. {SCENE_SUFFIX}" if s else SCENE_SUFFIX


# ---- rendering ----------------------------------------------------------------
def render_text(reg: FontRegistry, comps: List[Dict], language: str, seed: int, rng: random.Random,
                composition: str, paper, colors: Dict[str, str], w: int, h: int):
    """The text set on a canvas of the paper colour. -> (image, card box, layout)."""
    cp = COMPOSITIONS[composition]
    comps = [dict(c, date_style=cp["date"]) if c.get("type") == "date" else c for c in comps]
    groups = invite.split_groups(comps)
    b = CARD_BOX
    box = (int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h))
    sub = {g: groups[g] for g in ("header", "date", "place") if groups.get(g)}
    layout = rng.choice(["stacked", "airy"])
    boxes = order_boxes(box, cp["order"], layout, sub)
    ov = {"fit_mode": "auto", "align": cp["align"], "names_effect": rng.choice(["flat", "foil"]), **colors}
    if cp["align"] != "left":
        ov["axis"] = (box[0] + box[2]) / 2
    canvas = Image.new("RGB", (w, h), tuple(paper))
    out, _ = invite.render_boxes(reg, canvas, {g: (sub[g], boxes[g]) for g in sub}, language, seed, ov,
                                 mask=np.ones((h // 4, w // 4), bool))
    return out, box, {"layout": layout, "names_effect": ov["names_effect"]}


def build_inputs(text_img: Image.Image, paper, shape: str, backdrop, seed: int):
    """-> (input image, repaint strength 'L', glyph alpha 'L')."""
    from scipy import ndimage
    w, h = text_img.size
    d = np.abs(np.asarray(text_img, np.int16) - np.array(paper, np.int16)).max(2)
    ink = d > 10
    ys, xs = np.where(ink)
    if not len(ys):
        raise ValueError("no text was rendered")
    hx, m = int(HALO * w), int(CARD_MARGIN * w)
    x0, x1, y0, y1 = xs.min() - hx, xs.max() + hx, ys.min() - hx, ys.max() + hx
    if shape == "arch":                       # the arch curve must clear the first line
        y0 -= int(0.30 * (x1 - x0))
    elif shape == "ticket":                   # the corner notches must clear the corner lines
        x0, x1 = x0 - int(0.03 * w), x1 + int(0.03 * w)
        y0, y1 = y0 - int(0.03 * w), y1 + int(0.03 * w)
    x0, y0, x1, y1 = max(m, x0), max(m, y0), min(w - m, x1), min(h - m, y1)
    card = Image.new("L", (w, h), 0)
    card.paste(invite.card_mask(shape, int(x1 - x0), int(y1 - y0)), (int(x0), int(y0)))
    card_a = np.asarray(card, np.float32)[..., None] / 255
    # backdrop: theme tone, MOTTLED -- a flat colour came back as a flat band
    nrng = np.random.default_rng(seed)
    blot = nrng.normal(0, 1, (max(2, h // 32), max(2, w // 32), 3)).astype(np.float32)
    blot = np.asarray(Image.fromarray(((blot * 40) + 128).clip(0, 255).astype(np.uint8))
                      .resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(0.04 * w)), np.float32)
    img = (np.ones((h, w, 3), np.float32) * np.array(backdrop, np.float32) + (blot - 128) * 1.2).clip(0, 255)
    img = img * (1 - card_a) + np.asarray(text_img, np.float32) * card_a
    inp = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    # strength
    g = 4
    small = np.asarray(Image.fromarray(ink.astype(np.uint8) * 255).resize((w // g, h // g), Image.BOX)) > 0
    block = ndimage.binary_dilation(small, iterations=max(1, int(0.015 * w / g)))
    dist = ndimage.distance_transform_edt(~block) * g
    inside = INSIDE_NEAR + (INSIDE_FAR - INSIDE_NEAR) * np.clip(dist / (RAMP * w), 0, 1)
    card_small = card.resize((w // g, h // g))
    edge = ndimage.distance_transform_edt(np.asarray(card_small, np.uint8) > 127) * g
    inside = np.where(edge < EDGE * w, np.maximum(inside, EDGE_STRENGTH), inside)
    # the band hugging the text wins over everything (a heart in a ticket's
    # corner sat inside the notch's edge band)
    inside[dist < LOCK * w] = 0.0
    cs = np.asarray(card.resize((w // g, h // g), Image.BILINEAR), np.float32) / 255
    strength = inside * cs + (1 - cs)
    st = Image.fromarray((strength * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    st = st.filter(ImageFilter.GaussianBlur(0.006 * w))
    alpha = Image.fromarray((np.clip(d / 255.0 * 3, 0, 1) * 255).astype(np.uint8))
    return inp, st, alpha


def prepare(reg: FontRegistry, plan, request: str, seed: int = 0, w: int = 896, h: int = 1344,
            composition: str = "") -> Dict:
    plan = fix_plan(parse_plan(plan), request)
    rng = random.Random(seed)
    comp, shape = pick_layout(plan, rng, composition)
    paper = paper_of(plan)
    colors = text_colors(plan, paper)
    clean, dropped = drop_copied(plan.get("components") or [], request)
    comps, breaks = apply_breaks(clean)
    if not any(c.get("type") == "names" for c in comps):
        raise ValueError("the plan has no names")
    text_img, box, info = render_text(reg, comps, plan.get("language", "latin"), seed, rng,
                                      comp, paper, colors, w, h)
    acc = hex_to_rgb(colors["accent"])
    backdrop = tuple(int(0.55 * c + 0.45 * 150) for c in acc)
    inp, mask, alpha = build_inputs(text_img, paper, shape, backdrop, seed)
    report = {"composition": comp, "card": shape, "paper": rgb_to_hex(paper), **colors, **info,
              "line_breaks": breaks, "dropped": dropped}
    return {"image": inp, "mask": mask, "alpha": alpha,
            "prompt": scene_prompt(plan.get("background_prompt")), "report": report}


def composite(generated: Image.Image, image: Image.Image, alpha: Image.Image) -> Image.Image:
    """The glyphs back, pixel-exact, over whatever the model painted."""
    gen = generated.convert("RGB")
    if gen.size != image.size:
        gen = gen.resize(image.size, Image.LANCZOS)
    a = np.asarray(alpha.convert("L"), np.float32)[..., None] / 255
    out = np.asarray(gen, np.float32) * (1 - a) + np.asarray(image.convert("RGB"), np.float32) * a
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
