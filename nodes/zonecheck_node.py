# -*- coding: utf-8 -*-
"""ComfyUI node: DesignZoneCheck (Node 3)."""
from __future__ import annotations
import json

from ..designlayout import zone_check, render_preview
from ..designlayout.schema import to_app_json
from ..designlayout.fonts import FontRegistry
from ..designlayout.tensors import pil_from_any, pil_to_tensor
from .layout_node import _resolve_fonts_dir


class DesignZoneCheck:
    """Refine text colors / shadows against the real generated background."""

    CATEGORY = "DesignLayout"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("final_json", "composite_preview", "report")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "layout_json": ("STRING", {"multiline": True, "default": ""}),
                "fonts_dir": ("STRING", {"default": "fonts"}),
                "busy_threshold": ("FLOAT", {"default": 0.18, "min": 0.0,
                                             "max": 1.0, "step": 0.01}),
                "min_contrast": ("FLOAT", {"default": 4.5, "min": 1.0,
                                           "max": 21.0, "step": 0.1}),
            }
        }

    def run(self, background, layout_json, fonts_dir, busy_threshold,
            min_contrast):
        layout = json.loads(layout_json) if layout_json.strip() else {}
        bg = pil_from_any(background)
        reg = FontRegistry(_resolve_fonts_dir(fonts_dir))
        updated, report = zone_check.check(bg, layout, busy_threshold,
                                           min_contrast)
        composite = render_preview.composite_preview(bg, updated, reg)
        final = to_app_json(updated)
        return (json.dumps(final, ensure_ascii=False),
                pil_to_tensor(composite), "\n".join(report))
