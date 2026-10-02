# -*- coding: utf-8 -*-
"""ComfyUI node: DesignLayoutFromImage (image-first flow).

Runs AFTER the background is generated: scores the templates against the real
image, keeps the calmest placement, defocuses the scene behind each line and
fits the text colors -- one node instead of LayoutEngine + ZoneCheck.
"""
from __future__ import annotations
import json

from ..designlayout import image_fit, zone_check, render_preview, compositions
from ..designlayout import text_fx
from ..designlayout.colors import parse_hex
from ..designlayout.schema import to_app_json
from ..designlayout.fonts import FontRegistry
from ..designlayout.tensors import pil_from_any, pil_to_tensor
from .layout_node import _resolve_fonts_dir


class DesignLayoutFromImage:
    """Place the text where the generated background is calm."""

    CATEGORY = "DesignLayout"
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("final_json", "composite_preview", "background_treated",
                    "layout_json", "report")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        comps = ["auto"] + compositions.list_names()
        return {
            "required": {
                "background": ("IMAGE",),
                "design_spec": ("STRING", {"multiline": True, "default": ""}),
                "fonts_dir": ("STRING", {"default": "fonts"}),
                "canvas_width": ("INT", {"default": 1080, "min": 256,
                                         "max": 8192}),
                "canvas_height": ("INT", {"default": 1080, "min": 256,
                                          "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1}),
                # auto = score templates on the image; a name forces one
                "composition": (comps,),
                # 0 = every suitable template
                "candidates": ("INT", {"default": 0, "min": 0, "max": 300}),
                # defocus behind each line (0 = off, 1 = default, 2 = strong)
                "defocus": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0,
                                      "step": 0.05}),
                "min_contrast": ("FLOAT", {"default": 4.5, "min": 1.0,
                                           "max": 21.0, "step": 0.1}),
            },
            "optional": {
                # text look: none = flat text (as before), auto = preset +
                # colors picked from mood / LLM colors / the image, per seed
                "text_effect": (text_fx.EFFECT_CHOICES, {"default": "none"}),
            },
        }

    def run(self, background, design_spec, fonts_dir, canvas_width,
            canvas_height, seed, composition, candidates, defocus,
            min_contrast, text_effect="none"):
        spec = json.loads(design_spec) if design_spec.strip() else {}
        bg = pil_from_any(background)
        reg = FontRegistry(_resolve_fonts_dir(fonts_dir))
        canvas = {"w": canvas_width, "h": canvas_height}
        force = composition if composition and composition != "auto" else ""
        layout, report, warns = image_fit.fit_layout(
            reg, spec, bg, canvas, seed=seed, candidates=candidates,
            force=force)
        treated = image_fit.defocus_behind_text(reg, layout, bg, defocus,
                                                report)
        # the scene behind the text is calm now: fit colors, no gray scrims
        final, zr = zone_check.check(treated, layout,
                                     min_contrast=min_contrast,
                                     allow_scrim=False)
        composite, style = text_fx.render_with_effects(
            reg, final, treated, spec.get("mood", final.get("mood", "")),
            [c for c in (parse_hex(spec.get("text_color")),
                         parse_hex(spec.get("accent_color"))) if c],
            seed, text_effect)
        if composite is None:
            composite = render_preview.composite_preview(treated, final, reg)
        else:
            final["text_effect"] = style
            zr.append(f"text effect: {style['preset']} "
                      f"{style['main']} / {style['accent']}")
        report = report + zr + [f"warning: {w}" for w in
                                dict.fromkeys(warns + reg.warnings)]
        return (json.dumps(to_app_json(final), ensure_ascii=False),
                pil_to_tensor(composite), pil_to_tensor(treated),
                json.dumps(final, ensure_ascii=False), "\n".join(report))
