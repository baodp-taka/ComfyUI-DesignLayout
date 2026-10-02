# -*- coding: utf-8 -*-
"""ComfyUI nodes: invitation card, text first.

    InviteCardPrepare   plan JSON + request -> card image, repaint mask,
                        glyph mask, scene prompt (for Qwen-Image-Edit inpaint)
    InviteCardComposite the inpainted image + the card -> glyphs put back

Wiring (API format):
    Prepare.image  -> VAEEncode.pixels, TextEncodeQwenImageEditPlus.image1
    Prepare.prompt -> TextEncodeQwenImageEditPlus.prompt
    Prepare.mask   -> SetLatentNoiseMask.mask      (model through DifferentialDiffusion)
    VAEDecode      -> Composite.generated; Prepare.image / glyph_mask -> Composite
"""
from __future__ import annotations
import json

from ..designlayout import invite_card
from ..designlayout.fonts import FontRegistry
from ..designlayout.tensors import mask_to_tensor, pil_from_any, pil_to_tensor
from .layout_node import _resolve_fonts_dir


class InviteCardPrepare:
    CATEGORY = "DesignLayout"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "glyph_mask", "prompt", "report")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # the LLM's reply (JSON, code fences / chatter around it ok)
                "plan_json": ("STRING", {"multiline": True, "default": ""}),
                # the user's request: the truth for Vietnamese dates, copied-line check
                "request": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1}),
                "width": ("INT", {"default": 896, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1344, "min": 256, "max": 4096, "step": 16}),
                "fonts_dir": ("STRING", {"default": "fonts"}),
            },
            "optional": {
                # empty = the seed picks in the LLM's top 3
                "composition": (["auto"] + list(invite_card.COMPOSITIONS), {"default": "auto"}),
            },
        }

    def run(self, plan_json, request, seed, width, height, fonts_dir, composition="auto"):
        reg = FontRegistry(_resolve_fonts_dir(fonts_dir))
        out = invite_card.prepare(reg, plan_json, request, seed, width, height,
                                  "" if composition == "auto" else composition)
        return (pil_to_tensor(out["image"]), mask_to_tensor(out["mask"]),
                mask_to_tensor(out["alpha"]), out["prompt"],
                json.dumps(out["report"], ensure_ascii=False))


class InviteCardComposite:
    CATEGORY = "DesignLayout"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"generated": ("IMAGE",), "card_image": ("IMAGE",),
                             "glyph_mask": ("MASK",)}}

    def run(self, generated, card_image, glyph_mask):
        from PIL import Image
        import numpy as np
        a = glyph_mask.detach().cpu().numpy() if hasattr(glyph_mask, "detach") else np.asarray(glyph_mask)
        a = a[0] if a.ndim == 3 else a
        alpha = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L")
        out = invite_card.composite(pil_from_any(generated), pil_from_any(card_image), alpha)
        return (pil_to_tensor(out),)
