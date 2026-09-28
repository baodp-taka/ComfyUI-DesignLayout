# -*- coding: utf-8 -*-
"""ComfyUI-DesignLayout: code-driven design layout for text-free AI backgrounds.

Registers three nodes:
    DesignJSONParse     parse + clean the LLM output
    DesignLayoutEngine  compute layout, guide image, mask, augmented prompt
    DesignZoneCheck     refine colors / shadows against the real render
"""
# Node classes import torch (via tensors) and use relative imports, so they are
# only importable inside ComfyUI. Guard the import so the package metadata can
# still be imported by offline tests / tools without ComfyUI or torch present.
try:
    from .nodes.parse_node import DesignJSONParse
    from .nodes.layout_node import DesignLayoutEngine
    from .nodes.zonecheck_node import DesignZoneCheck

    NODE_CLASS_MAPPINGS = {
        "DesignJSONParse": DesignJSONParse,
        "DesignLayoutEngine": DesignLayoutEngine,
        "DesignZoneCheck": DesignZoneCheck,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "DesignJSONParse": "Design JSON Parse",
        "DesignLayoutEngine": "Design Layout Engine",
        "DesignZoneCheck": "Design Zone Check",
    }
except Exception:  # pragma: no cover - offline / non-ComfyUI import
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
