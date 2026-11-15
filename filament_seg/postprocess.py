"""Turn a per-pixel filament probability map into separate filament instances.

Steps (all parameters are tuned on the validation folds for PQ):
  1. threshold the probability map;
  2. optionally treat blobs separated by at most ``merge_gap`` pixels as one instance (a
     filament drawn as one object often breaks into beads when thresholded).  Grouping uses a
     distance map, so its cost does not depend on the gap, and each instance keeps only its
     original pixels — merging does not fatten masks (which would hurt SQ);
  3. drop instances that are too small or not confident enough (mostly false positives),
     but keep at least ``min_instances`` per image: every one of the 707 training images
     contains a filament, so an empty prediction is almost certainly wrong.

Steps 1-2 (:func:`components`) and step 3 (:func:`keep_remap`) are separate so the
parameter search can reuse the expensive connected-component pass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class PostprocessParams:
    threshold: float = 0.5
    merge_gap: int = 0  # pixels at 2048 resolution; blobs closer than this form one instance
    min_area: int = 100  # pixels at 2048 resolution
    min_mean_prob: float = 0.0  # drop instances whose mean probability is below this
    min_instances: int = 1  # if filtering leaves fewer, keep the best-scoring (area x mean prob) ones

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Components:
    labels: np.ndarray  # int32 label map, 0 = background
    fg_lab: np.ndarray  # label of every foreground pixel, in raster order
    area: np.ndarray  # (K+1,) pixel count per label
    mean_prob: np.ndarray  # (K+1,) mean probability per label


def components(prob: np.ndarray, threshold: float, merge_gap: float = 0) -> Components:
    binary = (prob > threshold).astype(np.uint8)
    if merge_gap > 0:
        # distance of every background pixel to the nearest blob; pixels within merge_gap / 2
        # bridge two blobs exactly when the gap between them is at most merge_gap
        dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        n, labels = cv2.connectedComponents((dist <= merge_gap / 2).astype(np.uint8), connectivity=8)
        labels[binary == 0] = 0
    else:
        n, labels = cv2.connectedComponents(binary, connectivity=8)
    flat = labels.ravel()
    fg_idx = np.flatnonzero(flat)
    fg_lab = flat[fg_idx]
    area = np.bincount(fg_lab, minlength=n)
    prob_sum = np.bincount(fg_lab, weights=prob.ravel()[fg_idx], minlength=n)
    return Components(labels, fg_lab, area, prob_sum / np.maximum(area, 1))


def keep_remap(c: Components, min_area: int, min_mean_prob: float, min_instances: int = 0) -> np.ndarray:
    """Lookup table old label -> new label (0 = dropped, kept instances renumbered 1..K)."""
    keep = (c.area >= max(min_area, 1)) & (c.mean_prob >= min_mean_prob)
    keep[0] = False
    shortfall = min(min_instances, len(keep) - 1) - int(keep.sum())
    if shortfall > 0:  # rescue the strongest rejected candidates (large and confident first)
        score = np.where(keep, -np.inf, c.area * c.mean_prob)
        score[0] = -np.inf
        keep[np.argsort(score)[::-1][:shortfall]] = True
    remap = np.zeros(len(keep), np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap


def filter_components(c: Components, min_area: int, min_mean_prob: float, min_instances: int = 0) -> np.ndarray:
    """Drop small / low-confidence instances; returns the relabelled int32 label map."""
    return keep_remap(c, min_area, min_mean_prob, min_instances)[c.labels]


def instances(prob: np.ndarray, p: PostprocessParams) -> tuple[Components, np.ndarray]:
    """Components and keep/relabel table for one probability map (see module docstring).

    If no pixel exceeds the threshold at all, ``min_instances`` cannot rescue anything, so the
    most confident region (pixels above half the maximum probability) is used instead.  The
    grid search in :mod:`tune` skips this rare fallback.
    """
    comp = components(prob, p.threshold, p.merge_gap)
    remap = keep_remap(comp, p.min_area, p.min_mean_prob, p.min_instances)
    if p.min_instances > 0 and not remap.any() and prob.max() > 0:
        comp = components(prob, 0.5 * float(prob.max()), p.merge_gap)
        # an impossible confidence filter keeps nothing, so only the rescue step selects
        remap = keep_remap(comp, min_area=0, min_mean_prob=np.inf, min_instances=p.min_instances)
    return comp, remap


def probs_to_labels(prob: np.ndarray, p: PostprocessParams) -> np.ndarray:
    """Return an int32 label map (0 = background, 1..K = filament instances)."""
    comp, remap = instances(prob, p)
    return remap[comp.labels]


def labels_to_masks(labels: np.ndarray) -> list[np.ndarray]:
    return [labels == k for k in range(1, int(labels.max()) + 1)]
