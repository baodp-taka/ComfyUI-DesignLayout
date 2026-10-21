# -*- coding: utf-8 -*-
"""TransparentCutout logic (numpy only, no model)."""
import numpy as np

from product_cutout.transparent import transparent


def _logo(h=200, w=200, bg=1.0):
    rgb = np.full((h, w, 3), bg, np.float32)
    alpha = np.zeros((h, w), np.float32)
    rgb[60:140, 60:140] = (0.1, 0.2, 0.8)          # blue square logo
    alpha[60:140, 60:140] = 1.0
    return rgb, alpha


def test_logo_keeps_size_and_is_transparent_outside():
    rgb, alpha = _logo()
    out, a, plain = transparent(rgb, alpha)
    assert plain and out.shape == (200, 200, 3) and a.shape == (200, 200)
    assert a[0, 0] == 0.0 and a[100, 100] == 1.0
    assert np.allclose(out[100, 100], (0.1, 0.2, 0.8))


def test_soft_edge_loses_the_background_colour():
    rgb, alpha = _logo()
    # half-covered edge pixel: 50% blue + 50% white background
    rgb[59, 100] = 0.5 * np.array((0.1, 0.2, 0.8)) + 0.5
    alpha[59, 100] = 0.5
    out, a, _ = transparent(rgb, alpha)
    assert a[59, 100] == 0.5
    assert np.allclose(out[59, 100], (0.1, 0.2, 0.8), atol=1e-5)


def test_busy_border_skips_edge_cleaning():
    rgb, alpha = _logo()
    rgb[0, :] = np.linspace(0, 1, 200)[:, None]      # gradient on every edge
    rgb[-1, :] = rgb[0, :]
    rgb[:, 0] = np.linspace(0, 1, 200)[:, None]
    rgb[:, -1] = rgb[:, 0]
    mixed = rgb[59, 100].copy()
    alpha[59, 100] = 0.5
    out, _, plain = transparent(rgb, alpha)
    assert not plain and np.allclose(out[59, 100], mixed)


def test_alpha_noise_is_cleared():
    rgb, alpha = _logo()
    alpha[5, 5], alpha[100, 100] = 0.01, 0.99
    _, a, _ = transparent(rgb, alpha)
    assert a[5, 5] == 0.0 and a[100, 100] == 1.0


def test_sticker_outline_is_white_and_even():
    rgb, alpha = _logo()
    out, a, _ = transparent(rgb, alpha, outline=0.05)   # 10 px on 200 px
    assert out.shape == (200, 200, 3)                   # room: no padding
    for y, x in ((55, 100), (100, 55), (145, 100), (100, 145)):
        assert a[y, x] == 1.0 and np.allclose(out[y, x], 1.0)
    assert a[100, 100] == 1.0 and np.allclose(out[100, 100], (0.1, 0.2, 0.8))
    assert a[100, 40] == 0.0 and a[0, 0] == 0.0         # beyond the outline


def test_outline_grows_the_canvas_only_when_it_hits_an_edge():
    rgb = np.ones((100, 100, 3), np.float32)
    alpha = np.zeros((100, 100), np.float32)
    alpha[0:50, 20:80] = 1.0                            # touches the top
    out, a, _ = transparent(rgb, alpha, outline=0.05)   # 5 px
    assert out.shape[0] > 100 and out.shape[0] == out.shape[1]
    assert a[0].max() == 0.0                            # outline not clipped


def test_narrow_holes_fill_wide_holes_stay_open():
    rgb = np.ones((300, 300, 3), np.float32)
    alpha = np.zeros((300, 300), np.float32)
    alpha[50:250, 50:250] = 1.0
    alpha[60:240, 140:146] = 0.0                       # 6 px slit
    alpha[80:220, 160:230] = 0.0                       # 70 px window
    _, a, _ = transparent(rgb, alpha, outline=0.02)     # 6 px outline
    assert a[150, 143] == 1.0                           # slit filled white
    assert a[150, 195] == 0.0                           # window stays open
