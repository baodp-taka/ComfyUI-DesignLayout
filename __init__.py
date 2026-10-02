# -*- coding: utf-8 -*-
"""ComfyUI-DesignLayout: code-driven design layout for text-free AI backgrounds.

Registers:
    DesignJSONParse        parse + clean the LLM output
    DesignLayoutEngine     compute layout, guide image, mask, augmented prompt
    DesignZoneCheck        refine colors / shadows against the real render
    DesignLayoutFromImage  image-first: pick the layout on the generated
                           background, defocus behind text, fit colors
    InviteCardPrepare      invitation, text first: plan JSON -> card image,
                           repaint mask, glyph mask, scene prompt
    InviteCardComposite    put the glyphs back on the inpainted card
    SaveTextFile           return text (e.g. the LLM's plan JSON) from a
                           RunPod worker-comfyui job
"""
import importlib

# Each node is imported on its own: one that cannot load (a missing library,
# torch outside ComfyUI in offline tests) does not take the others down, and
# the reason is printed -- the build's quick test fails on "not loaded".
_NODES = [
    ("nodes.parse_node", "DesignJSONParse", "Design JSON Parse"),
    ("nodes.layout_node", "DesignLayoutEngine", "Design Layout Engine"),
    ("nodes.zonecheck_node", "DesignZoneCheck", "Design Zone Check"),
    ("nodes.fit_node", "DesignLayoutFromImage", "Design Layout From Image"),
    ("nodes.invite_node", "InviteCardPrepare", "Invite Card Prepare"),
    ("nodes.invite_node", "InviteCardComposite", "Invite Card Composite"),
    ("nodes.text_node", "SaveTextFile", "Save Text File"),
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _module, _cls, _title in _NODES:
    try:
        NODE_CLASS_MAPPINGS[_cls] = getattr(importlib.import_module(f".{_module}", __name__), _cls)
        NODE_DISPLAY_NAME_MAPPINGS[_cls] = _title
    except Exception as _e:  # pragma: no cover - depends on the environment
        print(f"[ComfyUI-DesignLayout] {_cls} not loaded: {_e!r}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
