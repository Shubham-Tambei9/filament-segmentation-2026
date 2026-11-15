"""Stage 2: re-segment every detected filament in its own rotated, native-resolution crop.

A level set of one probability map cannot always represent the extent annotators draw: a filament
may be cut short, merged with a neighbour or drawn thicker than the threshold allows.  For each
instance produced by the rules we therefore

1. rotate a crop so the filament's long axis is horizontal (padding 50% along the axis so the crop
   shows where the filament ends), without ever up-sampling: thin filaments keep native detail;
2. feed four channels to a small U-Net -- image, ensemble probability, the instance's own mask
   (the "seed") and the other instances' masks (where a neighbour begins);
3. predict P(an annotator marks the pixel | the annotator draws this filament): the training target
   is the consensus of the annotator records that drew the matching filament;
4. warp the refined probability back and give each pixel to the instance with the highest refined
   probability above the threshold.

Existence is still decided by stage 1; the refiner only changes extent.  The idea follows the public
Kaggle notebook "Deciding under disagreement" (user tushhaaarrrr); this is an independent
implementation.  This module is torch-free (crop geometry and data generation); the network lives
in :mod:`refine_train`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CROP_W, CROP_H = 384, 128  # long axis horizontal
AXIS_PAD = 1.5  # crop covers 150% of the filament length
MIN_SIDE = 24.0  # minimum source extent (px) across the axis


@dataclass
class CropGeometry:
    matrix: np.ndarray  # 2x3 affine: source (2048 image) -> crop
    box: tuple[int, int, int, int]  # source bounding box (x0, y0, x1, y1) of the crop footprint


def crop_geometry(mask: np.ndarray) -> CropGeometry:
    """Rotated crop around one instance (boolean 2048 x 2048 mask)."""
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.stack([xs - cx, ys - cy])) if len(xs) > 1 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    ux, uy = evecs[:, 1]  # principal (long) axis
    along = (xs - cx) * ux + (ys - cy) * uy
    across = -(xs - cx) * uy + (ys - cy) * ux
    length = (along.max() - along.min() + 1) * AXIS_PAD + 16
    width = max((across.max() - across.min() + 1) * 3, MIN_SIDE)
    scale = min(1.0, CROP_W / length, CROP_H / width)  # never up-sample
    # re-centre on the extent midpoints, not the centroid, so asymmetric filaments stay centred
    mid_a, mid_c = (along.max() + along.min()) / 2, (across.max() + across.min()) / 2
    cx, cy = cx + mid_a * ux - mid_c * uy, cy + mid_a * uy + mid_c * ux
    m = np.array([
        [scale * ux, scale * uy, CROP_W / 2 - scale * (cx * ux + cy * uy)],
        [-scale * uy, scale * ux, CROP_H / 2 - scale * (-cx * uy + cy * ux)],
    ])
    inv = cv2.invertAffineTransform(m)
    corners = np.array([[0, 0, 1], [CROP_W, 0, 1], [0, CROP_H, 1], [CROP_W, CROP_H, 1]], float) @ inv.T
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 1
    h, w = mask.shape
    return CropGeometry(m, (max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)))


def warp_to_crop(arr: np.ndarray, g: CropGeometry, nearest: bool = False) -> np.ndarray:
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(arr, g.matrix, (CROP_W, CROP_H), flags=flags, borderValue=0)


def warp_from_crop(crop: np.ndarray, g: CropGeometry) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop-space map -> the source bounding box ``g.box`` (returns the patch and the box)."""
    x0, y0, x1, y1 = g.box
    m = g.matrix.copy()
    m[:, 2] += m[:, :2] @ np.array([x0, y0], float)  # source coords relative to the box origin
    inv = cv2.invertAffineTransform(m)
    patch = cv2.warpAffine(crop.astype(np.float32), inv, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=0)
    return patch, g.box


def crop_inputs(image: np.ndarray, prob: np.ndarray, labels: np.ndarray, k: int, g: CropGeometry) -> np.ndarray:
    """(4, CROP_H, CROP_W) float32: image, probability, own seed, other instances."""
    img = warp_to_crop(image.astype(np.float32) / 255.0, g)
    p = warp_to_crop(prob.astype(np.float32), g)
    lab = warp_to_crop(labels.astype(np.float32), g, nearest=True)
    return np.stack([img, p, (lab == k).astype(np.float32), ((lab > 0) & (lab != k)).astype(np.float32)])


def target_for(seed: np.ndarray, records: list[np.ndarray], min_iou: float = 0.2):
    """Consensus mask of the reference filaments matching ``seed`` across annotator records.

    For each record take the filament with the highest IoU with the seed (if >= ``min_iou``: that
    annotator "drew" this filament).  Returns (soft target, number of records that drew it).
    """
    acc = np.zeros(seed.shape, np.float32)
    n = 0
    s_idx = np.flatnonzero(seed)
    for gt in records:
        at = gt.ravel()[s_idx]
        cand = np.unique(at[at > 0])
        if len(cand) == 0:
            continue
        best, best_iou = None, 0.0
        for c in cand:
            g = gt == c
            inter = np.count_nonzero(at == c)
            iou = inter / (np.count_nonzero(g) + len(s_idx) - inter)
            if iou > best_iou:
                best, best_iou = g, iou
        if best_iou >= min_iou:
            acc += best
            n += 1
    return (acc / n if n else acc), n


def paste_back(shape, geoms: list[CropGeometry], crops: list[np.ndarray], threshold: float, seeds: np.ndarray) -> np.ndarray:
    """Combine refined crop probabilities into a pixel-disjoint label map (1..K, same ids as seeds).

    Each pixel goes to the instance with the highest refined probability, if that probability
    exceeds ``threshold``.  An instance whose refined mask vanishes keeps its original seed.
    """
    best_p = np.zeros(shape, np.float32)
    owner = np.zeros(shape, np.int32)
    for k, (g, c) in enumerate(zip(geoms, crops), start=1):
        patch, (x0, y0, x1, y1) = warp_from_crop(c, g)
        view_p, view_o = best_p[y0:y1, x0:x1], owner[y0:y1, x0:x1]
        better = patch > view_p
        view_p[better] = patch[better]
        view_o[better] = k
    out = np.where(best_p > threshold, owner, 0).astype(np.int32)
    present = np.bincount(out.ravel(), minlength=len(geoms) + 1)
    for k in np.flatnonzero(present[1:] == 0) + 1:  # refined mask vanished: fall back to the seed
        out[(seeds == k) & (out == 0)] = k
    return out
