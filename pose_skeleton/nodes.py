# -*- coding: utf-8 -*-
"""ComfyUI node: PoseSkeleton -- pose photo -> OpenPose-style skeleton.

Feeds Qwen-Image-Edit 2511 as "Picture 3" of the AI-model try-on: the model
takes the pose from the skeleton without copying the face or clothes of the
person in the pose photo. See pose.py.
"""
from __future__ import annotations
import numpy as np
import torch

# Checked at load time: a missing library makes the node "not loaded", which
# fails the Docker build's quick test instead of the first real job.
import cv2  # noqa: F401
import mediapipe  # noqa: F401

from .pose import DEFAULT_MIN_VISIBILITY, pose_skeleton


class PoseSkeleton:
    CATEGORY = "image/pose"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("skeleton",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_visibility": ("FLOAT", {"default": DEFAULT_MIN_VISIBILITY,
                                             "min": 0.0, "max": 1.0,
                                             "step": 0.01}),
            },
        }

    def run(self, image, min_visibility=DEFAULT_MIN_VISIBILITY):
        # IMAGE = float [B, H, W, 3] in 0..1; one skeleton per frame
        frames = image.detach().cpu().numpy()
        out = []
        for frame in frames:
            rgb = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            sk = pose_skeleton(rgb, min_visibility)
            out.append(torch.from_numpy(sk.astype(np.float32) / 255.0))
        return (torch.stack(out),)
