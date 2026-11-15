"""Panoptic Quality (PQ) for single-class instance segmentation.

PQ = SQ x RQ, where a predicted and a ground-truth instance are matched when IoU > 0.5
(this threshold makes the matching unique, so no Hungarian assignment is needed):

    SQ = mean IoU over matched pairs (TP)
    RQ = TP / (TP + FP/2 + FN/2)

We also count the two error modes the organisers report:
  * one-to-many  (fragmentation): a GT filament overlapped by >= 2 predicted instances
  * many-to-one  (over-merging):  a predicted instance overlapping >= 2 GT filaments
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def match_sparse(gt_lab: np.ndarray, pred_at_gt: np.ndarray, pred_lab: np.ndarray):
    """Instance matching from foreground pixels only.

    Args:
        gt_lab: label (1..n_gt) of every ground-truth foreground pixel.
        pred_at_gt: predicted label at those same pixels (0 = background).
        pred_lab: label (1..n_pred) of every predicted foreground pixel.

    Returns (IoUs of matched pairs, n_gt, n_pred, intersection matrix [n_gt, n_pred]).
    Filaments cover ~0.5% of the image, so this is ~100x cheaper than working on full maps.
    """
    gt_lab = gt_lab.astype(np.int64, copy=False)
    n_gt = int(gt_lab.max()) if gt_lab.size else 0
    n_pred = int(pred_lab.max()) if pred_lab.size else 0
    area_gt = np.bincount(gt_lab, minlength=n_gt + 1)[1:, None].astype(np.float64)
    area_pred = np.bincount(pred_lab, minlength=n_pred + 1)[None, 1:].astype(np.float64)
    joint = np.bincount(gt_lab * (n_pred + 1) + pred_at_gt, minlength=(n_gt + 1) * (n_pred + 1))
    inter = joint.reshape(n_gt + 1, n_pred + 1)[1:, 1:].astype(np.float64)
    union = area_gt + area_pred - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return iou[iou > 0.5], n_gt, n_pred, inter


def match_instances(gt_labels: np.ndarray, pred_labels: np.ndarray):
    """:func:`match_sparse` for two dense label maps (0 = background)."""
    gt_flat, pred_flat = gt_labels.ravel(), pred_labels.ravel()
    gt_idx = np.flatnonzero(gt_flat)
    pred_lab = pred_flat[np.flatnonzero(pred_flat)].astype(np.int64)
    return match_sparse(gt_flat[gt_idx], pred_flat[gt_idx].astype(np.int64), pred_lab)


@dataclass
class PQAccumulator:
    """Accumulates PQ statistics over many (image, annotator) evaluations."""

    sum_iou: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    one_to_many: int = 0
    many_to_one: int = 0
    n_gt: int = 0
    n_pred: int = 0
    per_image_pq: list[float] = field(default_factory=list)
    per_image_dice: list[float] = field(default_factory=list)  # foreground (semantic) Dice
    matched_ious: list[float] = field(default_factory=list)

    def update(self, gt_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        gt_flat, pred_flat = gt_labels.ravel(), pred_labels.ravel()
        gt_idx = np.flatnonzero(gt_flat)
        pred_lab = pred_flat[np.flatnonzero(pred_flat)].astype(np.int64)
        return self.update_sparse(gt_flat[gt_idx], pred_flat[gt_idx].astype(np.int64), pred_lab)

    def update_sparse(self, gt_lab: np.ndarray, pred_at_gt: np.ndarray, pred_lab: np.ndarray) -> float:
        n_both = int(np.count_nonzero(pred_at_gt))
        denom = gt_lab.size + pred_lab.size
        self.per_image_dice.append(2 * n_both / denom if denom else 1.0)
        return self.add(*match_sparse(gt_lab, pred_at_gt, pred_lab))

    def add(self, matched: np.ndarray, n_gt: int, n_pred: int, inter: np.ndarray) -> float:
        """Add one (image, annotator) comparison; returns that image's PQ."""
        tp = len(matched)
        fp, fn = n_pred - tp, n_gt - tp
        self.sum_iou += float(matched.sum())
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.n_gt += n_gt
        self.n_pred += n_pred
        self.matched_ious.extend(matched.tolist())
        if inter.size:
            touching = inter > 0
            self.one_to_many += int((touching.sum(1) >= 2).sum())
            self.many_to_one += int((touching.sum(0) >= 2).sum())
        denom = tp + 0.5 * fp + 0.5 * fn
        pq = float(matched.sum()) / denom if denom else 1.0
        self.per_image_pq.append(pq)
        return pq

    def merge(self, other: "PQAccumulator") -> "PQAccumulator":
        for name in ("sum_iou", "tp", "fp", "fn", "one_to_many", "many_to_one", "n_gt", "n_pred"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.per_image_pq.extend(other.per_image_pq)
        self.per_image_dice.extend(other.per_image_dice)
        self.matched_ious.extend(other.matched_ious)
        return self

    def summary(self) -> dict[str, float]:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return {
            "PQ": self.sum_iou / denom if denom else 1.0,
            "SQ": self.sum_iou / self.tp if self.tp else 0.0,
            "RQ": self.tp / denom if denom else 1.0,
            "PQ_image_mean": float(np.mean(self.per_image_pq)) if self.per_image_pq else 0.0,
            "Dice_image_mean": float(np.mean(self.per_image_dice)) if self.per_image_dice else 0.0,
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "one_to_many": self.one_to_many,
            "many_to_one": self.many_to_one,
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
        }
