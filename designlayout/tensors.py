# -*- coding: utf-8 -*-
"""PIL <-> torch tensor helpers for ComfyUI I/O.

ComfyUI IMAGE = float tensor [B, H, W, C] in 0..1 (RGB).
ComfyUI MASK  = float tensor [B, H, W] in 0..1.

torch is imported lazily so the pure-Python modules and the offline tests can
run without torch installed. `to_image_tensor` / `to_mask_tensor` require torch
(they are only called from the ComfyUI node wrappers).
"""
from __future__ import annotations
from typing import Any
import numpy as np
from PIL import Image


def _torch():
    import torch  # local import: only nodes need it
    return torch


def pil_to_tensor(img: Image.Image) -> Any:
    """PIL RGB(A) -> IMAGE tensor [1, H, W, 3] float 0..1."""
    torch = _torch()
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def tensor_to_pil(t: Any) -> Image.Image:
    """IMAGE tensor [B,H,W,C] (or [H,W,C]) -> first-frame PIL RGB."""
    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def mask_to_tensor(mask: Image.Image) -> Any:
    """PIL 'L' -> MASK tensor [1, H, W] float 0..1."""
    torch = _torch()
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def np_to_image_tensor(arr: np.ndarray) -> Any:
    """HxWx3 uint8/float ndarray -> IMAGE tensor [1,H,W,3]."""
    torch = _torch()
    a = arr.astype(np.float32)
    if a.max() > 1.0:
        a = a / 255.0
    return torch.from_numpy(a)[None, ...]


def pil_from_any(x: Any) -> Image.Image:
    """Accept a PIL image or an IMAGE tensor; return a PIL RGB image."""
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return tensor_to_pil(x)
