# PoseSkeleton (folder of ComfyUI-DesignLayout)

One ComfyUI node, **Pose Skeleton (MediaPipe)**: a pose photo becomes an
OpenPose-style skeleton (COCO-18 sticks + wrist→thumb/index/pinky lines) on
black, the same size as the input.

Built for the "AI model" try-on on Qwen-Image-Edit 2511: the skeleton is
"Picture 3", so the model takes the pose without copying the face or clothes of
the person in the pose photo.

```
LoadImage (pose photo) → PoseSkeleton → TextEncodeQwenImageEditPlus.image3
```

| Input | Default | |
|---|---|---|
| `image` | | pose photo (one skeleton per frame) |
| `min_visibility` | 0.05 | MediaPipe gives ~0.1 to limbs hidden by wide sleeves while their position is still right |

- Stick width is ~1% of the short side (never under 4px). Thin sticks (DWPose
  draws 4px at any size) are read by 2511 as a dark figure in the picture.
- No person found → the node raises, the job fails cleanly.

## Install

1. `pip install -r requirements.txt`, then keep a single OpenCV build:
   `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless && pip install opencv-contrib-python-headless`
2. Linux: `apt-get install -y libportaudio2`
3. Model `pose_landmarker_heavy.task` (~30 MB) in `ComfyUI/models/mediapipe/`
   (or `$POSE_LANDMARKER_MODEL`, or `pose_skeleton/models/`):
   https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

## Tests

```
python -m pytest -q
# with a real photo:
POSE_TEST_IMAGE=pose.png POSE_LANDMARKER_MODEL=pose_landmarker_heavy.task python -m pytest -q tests/test_pose_skeleton.py
```
