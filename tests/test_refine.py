"""Crop geometry of the instance refiner: rotate to horizontal, never up-sample, invert exactly."""
import cv2
import numpy as np

from filament_seg.refine import CROP_H, CROP_W, crop_geometry, paste_back, warp_from_crop, warp_to_crop


def _filament(angle_deg=35, length=300, width=12, centre=(900, 1100)):
    m = np.zeros((2048, 2048), np.uint8)
    box = cv2.boxPoints((centre, (length, width), angle_deg)).astype(np.int32)
    cv2.fillPoly(m, [box], 1)
    return m.astype(bool)


def test_crop_is_horizontal_centred_and_invertible():
    seed = _filament(length=200)  # 200 px x 1.5 padding + 16 < 384: fits at native scale
    g = crop_geometry(seed)
    crop = warp_to_crop(seed.astype(np.float32), g)
    ys, xs = np.nonzero(crop > 0.5)
    assert np.ptp(xs) > 4 * np.ptp(ys)  # long axis now horizontal
    assert abs(xs.mean() - CROP_W / 2) < 8 and abs(ys.mean() - CROP_H / 2) < 8
    assert np.isclose(np.hypot(*g.matrix[0, :2]), 1.0)  # native scale, never up-sampled
    patch, (x0, y0, x1, y1) = warp_from_crop(crop, g)
    back = np.zeros(seed.shape, bool)
    back[y0:y1, x0:x1] = patch > 0.5
    iou = (back & seed).sum() / (back | seed).sum()
    assert iou > 0.95


def test_long_filament_is_downscaled_not_cropped():
    seed = _filament(angle_deg=-20, length=900, width=14, centre=(1024, 1024))
    g = crop_geometry(seed)
    assert np.hypot(*g.matrix[0, :2]) < 1.0
    crop = warp_to_crop(seed.astype(np.float32), g)
    assert crop[:, :2].max() == 0 and crop[:, -2:].max() == 0  # filament ends visible inside the crop


def test_paste_back_assigns_pixels_to_most_confident_instance():
    a, b = _filament(centre=(800, 800)), _filament(centre=(808, 806))  # overlapping instances
    seeds = np.zeros(a.shape, np.int32)
    seeds[a] = 1
    seeds[b & ~a] = 2
    ga, gb = crop_geometry(a), crop_geometry(b)
    ca = warp_to_crop(a.astype(np.float32), ga) * 0.9
    cb = warp_to_crop(b.astype(np.float32), gb) * 0.6
    out = paste_back(a.shape, [ga, gb], [ca, cb], 0.5, seeds)
    assert set(np.unique(out)) <= {0, 1, 2}
    assert (out[a] == 1).mean() > 0.9  # shared pixels go to the more confident instance
