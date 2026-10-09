# -*- coding: utf-8 -*-
"""Pose photo -> OpenPose-style skeleton image (for Qwen-Image-Edit 2511).

MediaPipe Pose Landmarker (heavy) finds 33 body landmarks; they are mapped to
the OpenPose COCO-18 layout and drawn with the usual OpenPose colours on black,
plus thin wrist->thumb/index/pinky lines. The stick width grows with the image
(DWPose draws 4px at any size, which 2511 then reads as a real dark figure).

Low-visibility points are kept down to `min_visibility` (0.05 by default):
MediaPipe often gives ~0.1 to limbs hidden by wide sleeves while their position
is still right.

mediapipe and cv2 are imported lazily so the drawing can be tested offline.
"""
from __future__ import annotations
import atexit
import os
from typing import List, Optional, Sequence, Tuple
import numpy as np

Point = Tuple[float, float, float]  # x px, y px, visibility

# COCO-18 from MediaPipe indices (even = person's right).
# 0 nose, 1 neck (mid shoulders), 2 RSho, 3 REl, 4 RWr, 5 LSho, 6 LEl, 7 LWr,
# 8 RHip, 9 RKnee, 10 RAnk, 11 LHip, 12 LKnee, 13 LAnk, 14 REye, 15 LEye,
# 16 REar, 17 LEar
_COCO_FROM_MP = [0, None, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27,
                 5, 2, 8, 7]
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
         (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
         (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
          (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
          (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
          (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
          (255, 0, 170), (255, 0, 85)]  # RGB
# wrist -> (thumb, index, pinky), MediaPipe indices: right then left hand
HANDS = [(16, (22, 20, 18)), (15, (21, 19, 17))]
_FINGER_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

DEFAULT_MIN_VISIBILITY = 0.05
MODEL_FILE = "pose_landmarker_heavy.task"


def to_coco18(mp_points: Sequence[Point]) -> List[Point]:
    """33 MediaPipe points (px) -> 18 COCO points; neck = mid shoulders."""
    out: List[Point] = []
    for idx in _COCO_FROM_MP:
        if idx is None:
            a, b = mp_points[11], mp_points[12]
            out.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2])))
        else:
            out.append(mp_points[idx])
    return out


def stick_width(w: int, h: int) -> int:
    """~1% of the short side (8px at 832px), never under 4px."""
    return max(4, min(w, h) // 100)


def draw_skeleton(mp_points: Sequence[Point], w: int, h: int,
                  min_visibility: float = DEFAULT_MIN_VISIBILITY) -> np.ndarray:
    """33 MediaPipe points (px) -> HxWx3 uint8 RGB skeleton on black."""
    import cv2
    k = to_coco18(mp_points)
    canvas = np.zeros((h, w, 3), np.uint8)
    sw = stick_width(w, h)
    for i, (a, b) in enumerate(LIMBS):
        if min(k[a][2], k[b][2]) < min_visibility:
            continue
        pa, pb = np.array(k[a][:2], float), np.array(k[b][:2], float)
        length = float(np.linalg.norm(pa - pb))
        angle = float(np.degrees(np.arctan2(pb[1] - pa[1], pb[0] - pa[0])))
        poly = cv2.ellipse2Poly(tuple(int(v) for v in (pa + pb) / 2),
                                (int(length / 2), sw), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly,
                           tuple(int(c * 0.6) for c in COLORS[i]))
    for i, (x, y, v) in enumerate(k):
        if v >= min_visibility:
            cv2.circle(canvas, (int(x), int(y)), sw + 2, COLORS[i], -1)
    hand_w = max(2, sw // 2)
    for wrist, fingers in HANDS:
        for j, f in enumerate(fingers):
            a, b = mp_points[wrist], mp_points[f]
            if min(a[2], b[2]) < min_visibility:
                continue
            cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     _FINGER_COLORS[j], hand_w)
            cv2.circle(canvas, (int(b[0]), int(b[1])), hand_w + 1,
                       (255, 255, 255), -1)
    return canvas


_DETECTORS: dict = {}


@atexit.register
def _close_detectors():  # MediaPipe's own __del__ fails at interpreter exit
    for det in _DETECTORS.values():
        try:
            det.close()
        except Exception:
            pass
    _DETECTORS.clear()


def find_model(path: Optional[str] = None) -> str:
    """Model file: explicit path -> $POSE_LANDMARKER_MODEL ->
    <ComfyUI models>/mediapipe/ -> next to this package."""
    candidates = [path, os.environ.get("POSE_LANDMARKER_MODEL")]
    try:
        import folder_paths
        candidates.append(os.path.join(folder_paths.models_dir, "mediapipe",
                                       MODEL_FILE))
    except Exception:  # offline / tests: no ComfyUI
        pass
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "models", MODEL_FILE))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"{MODEL_FILE} not found (set POSE_LANDMARKER_MODEL or put it in "
        "models/mediapipe/)")


def detect(rgb: np.ndarray, model_path: Optional[str] = None
           ) -> Optional[List[Point]]:
    """HxWx3 uint8 RGB -> 33 landmarks in px of the first person, or None."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    path = find_model(model_path)
    det = _DETECTORS.get(path)
    if det is None:
        det = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.IMAGE))
        _DETECTORS[path] = det
    h, w = rgb.shape[:2]
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb)))
    if not res.pose_landmarks:
        return None
    return [(p.x * w, p.y * h, p.visibility) for p in res.pose_landmarks[0]]


def pose_skeleton(rgb: np.ndarray, min_visibility: float = DEFAULT_MIN_VISIBILITY,
                  model_path: Optional[str] = None) -> np.ndarray:
    """Pose photo (HxWx3 uint8 RGB) -> skeleton image of the same size.
    Raises ValueError when no person is found (the job then fails cleanly)."""
    points = detect(rgb, model_path)
    if points is None:
        raise ValueError("PoseSkeleton: no person found in the pose image")
    h, w = rgb.shape[:2]
    return draw_skeleton(points, w, h, min_visibility)
