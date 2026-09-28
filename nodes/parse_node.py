# -*- coding: utf-8 -*-
"""ComfyUI node: DesignJSONParse (Node 1)."""
from __future__ import annotations
import json

from ..designlayout import parsing


class DesignJSONParse:
    """Robustly parse the LLM output into a normalized design spec.

    Replaces the chain of Regex nodes: it validates the JSON, fills defaults,
    and cleans the background prompt (removing text-triggering phrases only).
    """

    CATEGORY = "DesignLayout"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("design_spec", "background_prompt", "warnings")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_output": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def run(self, llm_output: str):
        spec, bg_clean, warnings = parsing.parse_llm_output(llm_output)
        spec_json = json.dumps(spec, ensure_ascii=False)
        return (spec_json, bg_clean, "\n".join(warnings))
