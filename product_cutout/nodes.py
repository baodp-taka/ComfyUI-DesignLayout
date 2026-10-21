# -*- coding: utf-8 -*-
"""ComfyUI nodes on BiRefNet (ZhengPeng7/BiRefNet, MIT):

- ProductCutout: product photo -> clean reference (AI Background). cutout.py
  puts the product on white and crops it tight; feeds "Picture 1" of 2511.
- RemoveBackground: generated picture -> transparent RGBA (logo / sticker).
  transparent.py cleans the edges and draws the optional white outline.

GPU memory follows ComfyUI's own upscale-model pattern: before each run the
model asks ComfyUI to free the VRAM it needs (ComfyUI unloads its own models
if necessary), moves to the GPU, runs, then goes back to the CPU and the CUDA
cache is emptied. Between runs BiRefNet holds no VRAM, so ComfyUI keeps full
control of loading / unloading its models (2511, text encoder, VAE).
"""
from __future__ import annotations
import os
import threading
import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import folder_paths

# Checked at load time: a missing library makes the node "not loaded", which
# fails the Docker build's quick test instead of the first real job.
import einops  # noqa: F401
import kornia  # noqa: F401
import timm  # noqa: F401
import transformers  # noqa: F401

from .cutout import DEFAULT_MIN_PART, DEFAULT_PADDING, cutout
from .transparent import transparent

MODEL_DIR = os.environ.get("BIREFNET_MODEL_DIR") or os.path.join(
    folder_paths.models_dir, "birefnet", "BiRefNet")
SIZE = 1024  # BiRefNet's training size
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_MEMORY = 3 * 1024 ** 3  # VRAM asked from ComfyUI before a run (fp16, 1024px)

_model = None
_lock = threading.Lock()


def _load():
    """Model on the CPU (RAM), loaded once; fp16 when a GPU will run it."""
    global _model
    with _lock:
        if _model is None:
            from transformers import AutoModelForImageSegmentation
            m = AutoModelForImageSegmentation.from_pretrained(
                MODEL_DIR, trust_remote_code=True, local_files_only=True)
            m.eval()
            if mm.get_torch_device().type == "cuda":
                m.half()
            _model = m
    return _model


def foreground_alpha(rgb: np.ndarray) -> np.ndarray:
    """rgb float [H, W, 3] 0..1 -> alpha float [H, W] 0..1."""
    model = _load()
    device = mm.get_torch_device()
    dtype = next(model.parameters()).dtype
    h, w = rgb.shape[:2]
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
    x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear",
                      align_corners=False)
    x = ((x - _MEAN) / _STD).to(device, dtype)
    mm.free_memory(_MEMORY, device)
    model.to(device)
    try:
        with torch.no_grad():
            pred = model(x)[-1].sigmoid().float()
    finally:
        # never keep VRAM between runs: back to the CPU, empty the cache
        model.to(mm.unet_offload_device())
        mm.soft_empty_cache()
    pred = F.interpolate(pred.cpu(), size=(h, w), mode="bilinear",
                         align_corners=False)
    return pred[0, 0].clamp(0, 1).numpy()


class ProductCutout:
    CATEGORY = "image/product"
    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "cropped")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "padding": ("FLOAT", {"default": DEFAULT_PADDING, "min": 0.0,
                                      "max": 0.5, "step": 0.01}),
                "min_part": ("FLOAT", {"default": DEFAULT_MIN_PART,
                                       "min": 0.01, "max": 1.0,
                                       "step": 0.01}),
            },
        }

    def run(self, image, padding=DEFAULT_PADDING, min_part=DEFAULT_MIN_PART):
        # IMAGE = float [B, H, W, 3] 0..1; the product photo is frame 0
        rgb = image[0].detach().cpu().numpy().astype(np.float32)[..., :3]
        out, mask, cropped = cutout(rgb, foreground_alpha(rgb), padding,
                                    min_part)
        return (torch.from_numpy(out)[None], torch.from_numpy(mask)[None],
                cropped)


class RemoveBackground:
    """Picture -> RGBA IMAGE (4 channels): SaveImage writes a transparent PNG."""
    CATEGORY = "image/product"
    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "plain_background")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                # white sticker outline, fraction of the long side; 0 = logo
                "outline": ("FLOAT", {"default": 0.0, "min": 0.0,
                                      "max": 0.1, "step": 0.005}),
            },
        }

    def run(self, image, outline=0.0):
        rgb = image[0].detach().cpu().numpy().astype(np.float32)[..., :3]
        out, alpha, plain = transparent(rgb, foreground_alpha(rgb), outline)
        rgba = np.dstack([out, alpha])
        return (torch.from_numpy(rgba)[None], torch.from_numpy(alpha)[None],
                plain)
