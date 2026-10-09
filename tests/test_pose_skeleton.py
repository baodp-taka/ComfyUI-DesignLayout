# -*- coding: utf-8 -*-
"""PoseSkeleton: COCO-18 mapping + drawing offline; MediaPipe on a real photo
when POSE_TEST_IMAGE and the model ($POSE_LANDMARKER_MODEL) are available."""
import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pose_skeleton import pose  # noqa: E402


def _standing(w=400, h=800, vis=1.0):
    """33 MediaPipe points of a simple standing figure (px)."""
    p = [(w / 2, h * 0.12, vis)] * 33
    pts = dict(enumerate(p))
    pts.update({
        11: (w * 0.62, h * 0.25, vis), 12: (w * 0.38, h * 0.25, vis),
        13: (w * 0.68, h * 0.40, vis), 14: (w * 0.32, h * 0.40, vis),
        15: (w * 0.70, h * 0.55, vis), 16: (w * 0.30, h * 0.55, vis),
        23: (w * 0.57, h * 0.55, vis), 24: (w * 0.43, h * 0.55, vis),
        25: (w * 0.58, h * 0.75, vis), 26: (w * 0.42, h * 0.75, vis),
        27: (w * 0.58, h * 0.95, vis), 28: (w * 0.42, h * 0.95, vis),
    })
    return [pts[i] for i in range(33)]


def test_neck_is_between_the_shoulders():
    k = pose.to_coco18(_standing())
    assert len(k) == 18
    assert k[1][:2] == pytest.approx((200, 200))
    assert k[2][0] < k[5][0]  # right shoulder on the image left


def test_stick_width_grows_with_the_image():
    assert pose.stick_width(400, 800) == 4
    assert pose.stick_width(832, 1248) == 8
    assert pose.stick_width(1664, 2496) == 16


def test_draws_limbs_on_black_with_openpose_colors():
    img = pose.draw_skeleton(_standing(), 400, 800)
    assert img.shape == (800, 400, 3) and img.dtype == np.uint8
    assert img[5, 5].tolist() == [0, 0, 0]  # background stays black
    # middle of the right upper arm (shoulder 2 -> elbow 3), colour 2 * 0.6
    mid = (int((0.38 + 0.32) / 2 * 400), int((0.25 + 0.40) / 2 * 800))
    assert img[mid[1], mid[0]].tolist() == [int(255 * .6), int(170 * .6), 0]


def test_hidden_limbs_are_skipped_below_min_visibility():
    pts = _standing()
    pts[27] = (pts[27][0], pts[27][1], 0.02)  # left ankle hidden
    img = pose.draw_skeleton(pts, 400, 800, min_visibility=0.05)
    x, y = int(0.58 * 400), int(0.85 * 800)  # left shin (12 -> 13)
    assert img[y, x].tolist() == [0, 0, 0]
    kept = pose.draw_skeleton(pts, 400, 800, min_visibility=0.0)
    assert kept[y, x].tolist() != [0, 0, 0]


def test_missing_model_file_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("POSE_LANDMARKER_MODEL", raising=False)
    monkeypatch.setattr(pose.os.path, "dirname", lambda p: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="pose_landmarker_heavy"):
        pose.find_model()


@pytest.mark.skipif(not (os.environ.get("POSE_TEST_IMAGE")
                         and os.environ.get("POSE_LANDMARKER_MODEL")),
                    reason="needs POSE_TEST_IMAGE + POSE_LANDMARKER_MODEL")
def test_real_photo_gives_a_full_skeleton():
    pytest.importorskip("mediapipe")
    from PIL import Image
    rgb = np.asarray(Image.open(os.environ["POSE_TEST_IMAGE"]).convert("RGB"))
    sk = pose.pose_skeleton(rgb)
    assert sk.shape == rgb.shape
    assert (sk.sum(axis=2) > 0).mean() > 0.01  # something was drawn
    with pytest.raises(ValueError, match="no person"):
        pose.pose_skeleton(np.zeros((256, 256, 3), np.uint8))
