# -*- coding: utf-8 -*-
"""ComfyUI node: SaveTextFile -- return TEXT from a RunPod worker-comfyui job.

worker-comfyui only collects the "images" entries of output nodes: it fetches
each listed file through ComfyUI's /view and returns it base64 in
output.images[].data. /view serves any file of the output folder (a .json comes
back with its own content type), so writing the text to a file and listing it
under "images" returns the text with the job COMPLETED (no error-message hack).
"""
from __future__ import annotations
import os


class SaveTextFile:
    CATEGORY = "DesignLayout"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "text"}),
                "extension": (["json", "txt"], {"default": "json"}),
            },
        }

    def save(self, text, filename_prefix="text", extension="json"):
        import folder_paths
        out_dir = folder_paths.get_output_directory()
        full_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        filename = f"{name}_{counter:05}_.{extension}"
        with open(os.path.join(full_dir, filename), "w", encoding="utf-8") as f:
            f.write(text if isinstance(text, str) else str(text))
        return {"ui": {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}]},
                "result": ()}
