# -*- coding: utf-8 -*-
"""ComfyUI node: DesignLayoutEngine (Node 2)."""
from __future__ import annotations
import os
import json

from ..designlayout import guide as guide_mod
from ..designlayout import render_preview, compositions
from ..designlayout.fonts import FontRegistry
from ..designlayout.layout import LayoutEngine
from ..designlayout.tensors import pil_to_tensor, mask_to_tensor

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_FONTS = os.path.join(_PKG_ROOT, "fonts")


def _resolve_fonts_dir(fonts_dir: str) -> str:
    if fonts_dir and os.path.isdir(fonts_dir):
        return fonts_dir
    rel = os.path.join(_PKG_ROOT, fonts_dir) if fonts_dir else _DEFAULT_FONTS
    return rel if os.path.isdir(rel) else (fonts_dir or _DEFAULT_FONTS)


class DesignLayoutEngine:
    """Compute the layout, guide image, mask and augmented background prompt.

    canvas_width/height is the final design size (9:16, 4:5, 16:9, 3:1
    banners... ratio within 1:4 .. 4:1); templates are picked for that
    aspect. bg_width/height = 0 derives the background (latent) size from the
    canvas aspect at ~1 MP; wire the bg_width/bg_height outputs into
    EmptySD3LatentImage.
    """

    CATEGORY = "DesignLayout"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "IMAGE", "INT", "INT",
                    "STRING")
    RETURN_NAMES = ("guide_image", "text_mask", "layout_json",
                    "background_prompt_augmented", "debug_preview",
                    "bg_width", "bg_height", "warnings")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        comps = ["auto"] + compositions.list_names()
        return {
            "required": {
                "design_spec": ("STRING", {"multiline": True, "default": ""}),
                "fonts_dir": ("STRING", {"default": "fonts"}),
                # ratio must stay within 1:4 .. 4:1
                "canvas_width": ("INT", {"default": 1080, "min": 256,
                                         "max": 8192}),
                "canvas_height": ("INT", {"default": 1080, "min": 256,
                                          "max": 8192}),
                # 0 = auto (canvas aspect, ~1 MP, multiple of 16)
                "bg_width": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "bg_height": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1}),
                "composition": (comps,),
                "guide_mode": (["flat", "soft_gradient", "none"],),
                "feather_px": ("INT", {"default": 24, "min": 0, "max": 200}),
                "zone_padding": ("INT", {"default": 18, "min": 0, "max": 120}),
            },
            "optional": {
                "background_prompt": ("STRING", {"multiline": True,
                                                 "default": ""}),
            },
        }

    def run(self, design_spec, fonts_dir, canvas_width, canvas_height,
            bg_width, bg_height, seed, composition, guide_mode, feather_px,
            zone_padding, background_prompt=""):
        spec = json.loads(design_spec) if design_spec.strip() else {}
        force = composition if composition and composition != "auto" else ""
        reg = FontRegistry(_resolve_fonts_dir(fonts_dir))
        eng = LayoutEngine(reg, {"w": canvas_width, "h": canvas_height})
        layout = eng.layout(spec, seed=seed, force_composition=force)
        if not (bg_width and bg_height):
            bg_width, bg_height = guide_mod.auto_bg_size(canvas_width,
                                                         canvas_height)

        g_img, mask = guide_mod.build_guide(
            layout, bg_width, bg_height, guide_mode, feather_px, zone_padding)
        base_bp = background_prompt or spec.get("background_prompt", "")
        aug = guide_mod.augmented_prompt(layout, base_bp)
        debug = render_preview.debug_preview(layout, reg)

        warnings = list(dict.fromkeys(eng.warnings + reg.warnings))

        return (pil_to_tensor(g_img), mask_to_tensor(mask),
                json.dumps(layout, ensure_ascii=False), aug,
                pil_to_tensor(debug), bg_width, bg_height,
                "\n".join(warnings))
