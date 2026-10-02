# -*- coding: utf-8 -*-
"""ComfyUI-DesignLayout: code-driven design layout for text-free AI backgrounds.

Registers six nodes:
    DesignJSONParse        parse + clean the LLM output
    DesignLayoutEngine     compute layout, guide image, mask, augmented prompt
    DesignZoneCheck        refine colors / shadows against the real render
    DesignLayoutFromImage  image-first: pick the layout on the generated
                           background, defocus behind text, fit colors
    InviteCardPrepare      invitation, text first: plan JSON -> card image,
                           repaint mask, glyph mask, scene prompt
    InviteCardComposite    put the glyphs back on the inpainted card
"""
# Node classes import torch (via tensors) and use relative imports, so they are
# only importable inside ComfyUI. Guard the import so the package metadata can
# still be imported by offline tests / tools without ComfyUI or torch present.
try:
    from .nodes.parse_node import DesignJSONParse
    from .nodes.layout_node import DesignLayoutEngine
    from .nodes.zonecheck_node import DesignZoneCheck
    from .nodes.fit_node import DesignLayoutFromImage
    from .nodes.invite_node import InviteCardPrepare, InviteCardComposite

    NODE_CLASS_MAPPINGS = {
        "DesignJSONParse": DesignJSONParse,
        "DesignLayoutEngine": DesignLayoutEngine,
        "DesignZoneCheck": DesignZoneCheck,
        "DesignLayoutFromImage": DesignLayoutFromImage,
        "InviteCardPrepare": InviteCardPrepare,
        "InviteCardComposite": InviteCardComposite,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "DesignJSONParse": "Design JSON Parse",
        "DesignLayoutEngine": "Design Layout Engine",
        "DesignZoneCheck": "Design Zone Check",
        "DesignLayoutFromImage": "Design Layout From Image",
        "InviteCardPrepare": "Invite Card Prepare",
        "InviteCardComposite": "Invite Card Composite",
    }
except Exception:  # pragma: no cover - offline / non-ComfyUI import
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
