# -*- coding: utf-8 -*-
"""Offline end-to-end sample renderer (no GPU / ComfyUI needed).

For each sample LLM output: parse -> layout -> guide/mask/augmented prompt ->
debug_preview + composite_preview (over a fake background). Writes images to
samples/ for visual inspection.

Usage:
    python scripts/render_samples.py \
        --fonts-dir "C:/TAKA/PosterMaker/asset_resource/src/main/assets/fonts"
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from designlayout.fonts import FontRegistry           # noqa: E402
from designlayout.layout import LayoutEngine          # noqa: E402
from designlayout import parsing, guide, render_preview, zone_check  # noqa: E402
from designlayout.colors import hex_to_rgb            # noqa: E402

SAMPLES = [
    ('cafe_vn', '''{"language":"vietnamese","mood":"vintage","design_type":"poster",
      "composition_hint":"auto","background_color":"#3B2618",
      "background_prompt":"A cozy vintage coffee shop interior with warm wood and a copper espresso machine, soft golden light. Do not draw any text or letters.",
      "texts":[{"role":"headline","text":"MỘC COFFEE","priority":1},
      {"role":"subheadline","text":"KHAI TRƯƠNG · GIẢM 30% TUẦN ĐẦU","priority":2},
      {"role":"detail","text":"Thứ Bảy 12/10 · 123 Bạch Đằng, Đà Nẵng","priority":3}]}'''),
    ('sale_vn', '''{"language":"vietnamese","mood":"bold","design_type":"social",
      "background_color":"#111820",
      "background_prompt":"A dynamic abstract backdrop with soft neon gradients and empty space.",
      "texts":[{"role":"headline","text":"SIÊU SALE","priority":1},
      {"role":"emphasis","text":"50%","priority":1},
      {"role":"subheadline","text":"Duy nhất cuối tuần","priority":2},
      {"role":"note","text":"Áp dụng toàn bộ sản phẩm","priority":4}]}'''),
    ('wedding_en', '''{"language":"latin","mood":"elegant","design_type":"invitation",
      "background_color":"#F3EEE6",
      "background_prompt":"A delicate watercolor floral scene, soft blush and sage, lots of empty space.",
      "texts":[{"role":"subheadline","text":"TOGETHER WITH THEIR FAMILIES","priority":3},
      {"role":"headline","text":"Anna & David","priority":1},
      {"role":"body","text":"request the pleasure of your company","priority":2},
      {"role":"detail","text":"Saturday, October 12 · 5 PM","priority":3}]}'''),
    ('birthday_en', '''{"language":"latin","mood":"playful","design_type":"poster",
      "background_color":"#2A1A3A",
      "background_prompt":"A festive scene with confetti and balloons, plain empty middle.",
      "texts":[{"role":"headline","text":"HAPPY BIRTHDAY","priority":1},
      {"role":"subheadline","text":"Come celebrate with us","priority":2},
      {"role":"detail","text":"Sun 20th · 3 PM · Rooftop Bar","priority":3}]}'''),
    ('event_banner', '''{"language":"latin","mood":"minimal","design_type":"banner",
      "background_color":"#0E2A2A",
      "background_prompt":"A clean modern gradient scene, calm empty right side.",
      "texts":[{"role":"headline","text":"TECH SUMMIT 2026","priority":1},
      {"role":"subheadline","text":"The future of AI, live on stage","priority":2},
      {"role":"detail","text":"Nov 3-5 · Da Nang Convention Center","priority":3}]}'''),
    ('opening_broken', '''noise ```json {"language":"vietnamese","mood":"festive","design_type":"poster",
      "background_color":"#7A1020","background_prompt":"A red festive scene with lanterns and text banners everywhere.",
      "texts":[{"role":"headline","text":"KHAI TRƯƠNG","priority":1},
      {"role":"emphasis","text":"MỪNG","priority":1},
      {"role":"detail","text":"Ưu đãi vàng cho 100 khách đầu tiên",}]} trailing junk'''),
]


def fake_bg(w, h, hexcolor, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array(hex_to_rgb(hexcolor), dtype=np.float32)
    top = np.clip(base + 24, 0, 255)
    bot = np.clip(base - 24, 0, 255)
    grad = np.linspace(0, 1, h)[:, None, None]
    img = top[None, None, :] * (1 - grad) + bot[None, None, :] * grad
    img = np.repeat(img, w, axis=1)
    img += rng.normal(0, 6, size=img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    os.makedirs(out, exist_ok=True)

    reg = FontRegistry(args.fonts_dir)
    eng = LayoutEngine(reg)
    index = []
    for i, (name, raw) in enumerate(SAMPLES, 1):
        spec, bg_clean, warns = parsing.parse_llm_output(raw)
        lay = eng.layout(spec, seed=args.seed + i)
        g_img, mask = guide.build_guide(lay, 1024, 1024, "flat")
        aug = guide.augmented_prompt(lay, bg_clean)
        bg = fake_bg(1024, 1024, spec["background_color"], seed=i)
        comp = render_preview.composite_preview(bg, lay, reg)
        dbg = render_preview.debug_preview(lay, reg)
        final_lay, report = zone_check.check(bg, lay)
        final_img = render_preview.composite_preview(bg, final_lay, reg)
        stem = f"{i:02d}_{name}"
        g_img.save(os.path.join(out, stem + "_guide.webp"), quality=90)
        dbg.save(os.path.join(out, stem + "_debug.webp"), quality=90)
        comp.save(os.path.join(out, stem + "_composite.webp"), quality=90)
        final_img.save(os.path.join(out, stem + "_final.webp"), quality=90)
        allw = warns + eng.warnings + reg.warnings
        index.append((stem, lay["composition"], aug[:80], allw))
        eng.warnings = []
        print(f"[{stem}] comp={lay['composition']:14} blocks={len(lay['blocks'])}"
              f" shapes={len(lay['shapes'])} warns={len(allw)}")
    with open(os.path.join(out, "index.txt"), "w", encoding="utf-8") as f:
        for stem, comp, aug, warns in index:
            f.write(f"{stem}\t{comp}\n  aug: {aug}\n  warns: {warns}\n")
    print("samples ->", out)


if __name__ == "__main__":
    main()
