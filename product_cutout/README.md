# ProductCutout (folder of ComfyUI-DesignLayout)

One ComfyUI node, **Product Cutout (BiRefNet)**: a product photo becomes a
clean reference for Qwen-Image-Edit 2511 (AI Background) — the product on
white, cropped tight with a margin.

```
LoadImage (product photo) → ProductCutout → TextEncodeQwenImageEditPlus.image1
```

| Input | Default | |
|---|---|---|
| `image` | | product photo (frame 0 is used) |
| `padding` | 0.08 | white margin, fraction of the product's long side |
| `min_part` | 0.15 | a second separate object at least this big (vs the largest) means "several objects": the original photo is returned unchanged |

Outputs: `image`, `mask` (product alpha on the crop), `cropped` (false when the
original photo was returned).

- Why: the product fills the reference, so 2511 reads labels and logos even
  when the photo was taken from afar or on a cluttered table. The AI still
  repaints the whole picture, so light and shadows stay natural.
- Specks and small bits of neighbours are dropped (largest object only).

# TransparentCutout (same folder, same model)

**Transparent Cutout (BiRefNet)**: a picture the AI drew on a plain background
becomes a transparent RGBA image (logo / sticker); `SaveImage` writes it as a
transparent PNG.

```
VAEDecode → TransparentCutout → SaveImage
```

| Input | Default | |
|---|---|---|
| `image` | | generated picture (frame 0 is used) |
| `outline` | 0 | white sticker outline, fraction of the long side (0 = logo; the server sends 0.02) |

Outputs: `image` (RGBA, 4 channels), `mask`, `plain_background`.

- BiRefNet decides what the subject is by meaning, not by colour, so white
  parts inside a logo on a white page are kept.
- Edge clean-up: when the picture's border is one plain colour, that colour is
  taken out of the soft edge pixels, `F = (C - (1 - a) * bg) / a` (no halo on
  a dark page). A busy border skips this step (`plain_background` = false).
- Sticker outline drawn by code after the cut (like Canva), so it is always
  even; holes narrower than the outline fill white, wide holes stay open. The
  canvas grows only when the outline would hit an edge.
- Not named `RemoveBackground`: ComfyUI core (v0.38+) has a node with that
  name and a custom node can never replace a core one (it is silently skipped).
- Weak spot: a white detail on the subject's outer edge (white on white) can
  fade.

## GPU memory

Same pattern as ComfyUI's upscale models: before each run the node asks
ComfyUI to free ~3 GB (`free_memory`, ComfyUI unloads its own models if
needed), moves BiRefNet to the GPU, runs, then moves it back to the CPU and
empties the CUDA cache. Between runs it holds **no VRAM**, so ComfyUI keeps full
control of loading / unloading 2511, the text encoder and the VAE. The weights
stay in RAM (fp16, ~0.5 GB) so the next run does not reload them from disk.

## Install

1. `pip install scipy "transformers>=4.50,<5" timm kornia einops`
2. Model (MIT, ~0.9 GB) in `ComfyUI/models/birefnet/BiRefNet/` (or
   `$BIREFNET_MODEL_DIR`):
   `python -c "from huggingface_hub import snapshot_download as d; d('ZhengPeng7/BiRefNet', local_dir='models/birefnet/BiRefNet')"`

## Tests

```
python -m pytest -q tests/test_product_cutout.py tests/test_transparent.py
```
