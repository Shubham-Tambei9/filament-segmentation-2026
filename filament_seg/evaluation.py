"""Validation: PQ of predicted instances against every annotator of each image.

The hidden test labels are (most likely) a single annotator per image, so a prediction is scored
against each available annotator separately and the statistics are pooled.  This rewards
predicting what a *typical* annotator would draw.

This module is deliberately free of PyTorch so evaluation can run in many worker processes.
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .annotations import AnnotatedImage, label_map
from .decision import decide_image
from .imaging import read_gray, upsample_prob
from .metrics import PQAccumulator, match_instances
from .postprocess import PostprocessParams, instances, probs_to_labels

GT_COLUMNS = ["image", "annotator", "area", "category", "best_iou", "matched", "n_pred_touching"]
PRED_COLUMNS = ["image", "annotator", "area", "mean_prob", "best_iou", "matched", "n_gt_touching"]


class ProbLoader:
    """``get_prob(stem)`` reading a cache of network-resolution maps (picklable, unlike a lambda)."""

    def __init__(self, prob_dir: str | Path):
        self.prob_dir = Path(prob_dir)

    def __call__(self, stem: str) -> np.ndarray:
        return upsample_prob(np.load(self.prob_dir / f"{stem}.npy"))


def prob_loader(prob_dir: str | Path) -> ProbLoader:
    return ProbLoader(prob_dir)


class DecisionLabeler:
    """``labeler(stem, prob)`` applying a trained decision layer.

    With ``fold_of`` (stem -> fold) every image is scored by the scorer that never saw its fold
    (``decision["fold_scorers"]``), which reproduces the cross-fitted out-of-fold evaluation.
    """

    def __init__(self, decision: dict, image_dir: str | Path, fold_of: dict[str, int] | None = None):
        self.decision, self.image_dir, self.fold_of = decision, Path(image_dir), fold_of

    def __call__(self, stem: str, prob: np.ndarray) -> np.ndarray:
        scorer = self.decision["fold_scorers"][self.fold_of[stem]] if self.fold_of else None
        return decide_image(prob, read_gray(self.image_dir / f"{stem}.jpeg"), self.decision, scorer)


def evaluate_probs(
    get_prob: Callable[[str], np.ndarray], items: list[AnnotatedImage], params: PostprocessParams
) -> dict:
    """``get_prob(stem)`` must return the 2048 x 2048 probability map for that image."""
    acc = PQAccumulator()
    for it in items:
        pred = probs_to_labels(get_prob(it.stem), params)
        for filaments in it.annotations.values():
            acc.update(label_map(filaments), pred)
    return acc.summary()


def _detailed(args) -> tuple[PQAccumulator, list, list]:
    get_prob, items, params, labeler, progress = args
    acc = PQAccumulator()
    gt_rows, pred_rows = [], []
    for it in tqdm(items, desc="evaluate", disable=True if not progress else None):
        prob = get_prob(it.stem)
        if labeler is None:
            comp, remap = instances(prob, params)
            pred = remap[comp.labels]
        else:
            pred = labeler(it.stem, prob)
        flat = pred.ravel()
        n_inst = int(flat.max())
        inst_area = np.bincount(flat, minlength=n_inst + 1)[1:].astype(float)
        inst_prob = np.bincount(flat, weights=prob.ravel(), minlength=n_inst + 1)[1:] / np.maximum(inst_area, 1)
        for annotator, filaments in it.annotations.items():
            gt = label_map(filaments)
            acc.update(gt, pred)
            _, n_gt, n_pred, inter = match_instances(gt, pred)
            area_gt = np.bincount(gt.ravel(), minlength=n_gt + 1)[1:].astype(float)
            union = area_gt[:, None] + inst_area[None, :n_pred] - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            for k, f in enumerate(filaments):
                best = float(iou[k].max()) if n_pred else 0.0
                gt_rows.append((it.stem, annotator, area_gt[k], f.category_id, best, best > 0.5,
                                int((inter[k] > 0).sum()) if n_pred else 0))
            for j in range(n_pred):
                best = float(iou[:, j].max()) if n_gt else 0.0
                pred_rows.append((it.stem, annotator, inst_area[j], float(inst_prob[j]), best,
                                  best > 0.5, int((inter[:, j] > 0).sum()) if n_gt else 0))
    return acc, gt_rows, pred_rows


def evaluate_detailed(
    get_prob: Callable[[str], np.ndarray],
    items: list[AnnotatedImage],
    params: PostprocessParams | None = None,
    labeler: Callable[[str, np.ndarray], np.ndarray] | None = None,
    workers: int = 1,
) -> tuple[PQAccumulator, pd.DataFrame, pd.DataFrame]:
    """Full report: pooled accumulator plus one row per GT filament and per predicted filament.

    Instances come from the fixed rules (``params``) or from any ``labeler(stem, prob)`` returning
    a label map (e.g. :class:`DecisionLabeler`).  GT rows: image, annotator, area, category, best
    IoU with any prediction, matched (IoU > 0.5), number of predictions touching it.  Prediction
    rows: image, annotator, area, mean probability, best IoU, matched, GT filaments touched.
    With ``workers > 1`` images are split over processes (``get_prob`` / ``labeler`` must be
    picklable, e.g. :class:`ProbLoader` / :class:`DecisionLabeler`).
    """
    if workers <= 1:
        parts = [_detailed((get_prob, items, params, labeler, True))]
    else:
        chunks = [items[i::workers] for i in range(workers)]
        with Pool(workers) as pool:
            parts = pool.map(_detailed, [(get_prob, c, params, labeler, False) for c in chunks])
    acc = PQAccumulator()
    gt_rows, pred_rows = [], []
    for a, g, p in parts:
        acc.merge(a)
        gt_rows += g
        pred_rows += p
    return acc, pd.DataFrame(gt_rows, columns=GT_COLUMNS), pd.DataFrame(pred_rows, columns=PRED_COLUMNS)
