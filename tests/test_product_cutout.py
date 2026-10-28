# -*- coding: utf-8 -*-
"""ProductCutout crop logic (numpy only, no model)."""
import numpy as np

from product_cutout.cutout import cutout


def _photo(h=400, w=300):
    rgb = np.full((h, w, 3), 0.3, np.float32)  # grey cluttered "table"
    return rgb, np.zeros((h, w), np.float32)


def test_single_product_is_cropped_on_white_with_margin():
    rgb, alpha = _photo()
    rgb[100:300, 120:180] = (0.8, 0.1, 0.1)
    alpha[100:300, 120:180] = 1.0
    out, mask, cropped = cutout(rgb, alpha, padding=0.1)
    assert cropped
    # 200 px tall product + 2 * 20 px margin (soft edge adds a few px)
    assert 236 <= out.shape[0] <= 250 and out.shape[2] == 3
    assert np.allclose(out[0, 0], 1.0)            # white margin
    cy, cx = out.shape[0] // 2, out.shape[1] // 2
    assert np.allclose(out[cy, cx], (0.8, 0.1, 0.1))  # product pixels kept
    assert mask[cy, cx] == 1.0 and mask[0, 0] == 0.0


def test_margin_may_go_past_the_photo_edge():
    rgb, alpha = _photo()
    alpha[0:200, 0:100] = 1.0                     # product touches the corner
    out, _, cropped = cutout(rgb, alpha, padding=0.1)
    assert cropped and np.allclose(out[0, 0], 1.0)


def test_small_specks_are_dropped():
    rgb, alpha = _photo()
    alpha[100:300, 120:180] = 1.0                 # product: 12000 px
    alpha[10:20, 10:20] = 1.0                     # speck: 100 px
    out, _, cropped = cutout(rgb, alpha, padding=0.0)
    assert cropped and out.shape[0] < 220         # crop ignores the speck


def test_several_objects_are_all_kept():
    # bottle next to a bigger stack of towels: both stay, old background goes
    rgb, alpha = _photo(h=400, w=400)
    rgb[120:320, 170:230] = (0.6, 0.6, 0.6)
    alpha[120:320, 170:230] = 1.0                 # bottle
    alpha[150:350, 290:380] = 1.0                 # towels
    out, mask, cropped = cutout(rgb, alpha, padding=0.0)
    assert cropped
    assert 210 <= out.shape[1] <= 220             # bottle .. towels
    assert np.allclose(out[100, 30], (0.6, 0.6, 0.6))   # bottle pixels
    assert np.allclose(out[100, 80], 1.0)         # gap is white now


def test_object_cut_by_the_frame_is_dropped():
    rgb, alpha = _photo(h=300, w=300)
    alpha[120:180, 120:180] = 1.0                 # the product
    alpha[0:120, 0:120] = 1.0                     # a big thing cut by the corner
    out, _, cropped = cutout(rgb, alpha, padding=0.0)
    assert cropped and out.shape[0] < 80


def test_objects_all_cut_by_the_frame_are_kept():
    rgb, alpha = _photo(h=300, w=300)
    alpha[0:100, 0:100] = 1.0
    alpha[200:300, 200:300] = 1.0
    out, _, cropped = cutout(rgb, alpha, padding=0.0)
    assert cropped and out.shape[0] == 300


def test_tiny_speck_is_not_kept_next_to_the_product():
    rgb, alpha = _photo(h=300, w=300)
    alpha[145:155, 245:255] = 1.0                 # speck (1%)
    alpha[20:200, 20:120] = 1.0                   # the product
    out, _, cropped = cutout(rgb, alpha, padding=0.0)
    assert cropped and out.shape[1] < 120


def test_nothing_found_keeps_the_original_photo():
    rgb, alpha = _photo()
    out, _, cropped = cutout(rgb, alpha)
    assert not cropped and out is rgb
