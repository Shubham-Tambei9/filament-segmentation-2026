"""PQ-optimal selection of filament candidates from a multi-threshold component tree.

Why a decision rule can be derived.  For one annotator record, TP + FP = |P| and TP + FN = |G|, so
the PQ denominator TP + FP/2 + FN/2 equals (|P| + |G|) / 2: *every* predicted instance adds exactly
1/2 to it, whether it is matched or not.  Pooled over all (image, record) pairs,

    PQ(P) = sum_i R_i sum_{c in P_i} u_i(c)  /  ( 1/2 sum_i R_i |P_i|  +  1/2 N_G ),

where R_i is the number of annotator records of image i, N_G the total number of reference
filaments, and u_i(c) the *annotator-expected matched IoU* of candidate c: the mean over records of
IoU(c, g) if c matches a filament g of that record (IoU > 0.5) and 0 otherwise.  Matches are
unique for pixel-disjoint predictions, so contributions add up.  For a ratio of additive terms the
optimum P* maximises  sum_c (u(c) - tau)  with tau = PQ* / 2  (Dinkelbach, 1967): keep a candidate
exactly when its expected matched IoU exceeds half the attainable PQ.

Candidates.  Thresholding the probability map at increasing levels gives nested masks, so the
connected components form a forest (each component lies inside exactly one component of the level
below).  Any antichain of that forest is a valid pixel-disjoint submission, and it contains every
single-threshold segmentation as a special case — a faint filament can be taken at a low level while
a neighbour is split off at a higher one.  The best antichain for given scores is found exactly by
dynamic programming:  best(v) = max(0, s(v) - tau, sum_{children w} best(w)).

The idea of this decision layer follows the public Kaggle notebook "Deciding under disagreement"
(user tushhaaarrrr); this module is an independent implementation with its own features and scorer.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

LEVELS = tuple(np.round(np.arange(0.10, 0.901, 0.05), 2))
DISK_CENTER = (1023.1, 1024.1)  # (x, y): the GONG disk geometry is fixed in this dataset
DISK_RADIUS = 899.7

RING_PX = 6  # width of the surrounding ring used for contrast features

FEATURES = [
    "threshold", "log_area", "mean_p", "max_p", "std_p", "margin", "log_length", "width",
    "elongation", "fill", "radius", "darkness", "log_parent_ratio", "n_children", "n_siblings",
    "depth_below", "log_n_level", "level_area_frac", "log_image_activity",
    # how distinct the candidate is from its surroundings and how stable it is across levels
    "ring_p", "ring_contrast", "child_area_frac", "parent_mean_p", "stability",
]


@dataclass
class Tree:
    """Candidate forest of one image.  Node arrays are ordered by level, then component id."""

    levels: tuple[float, ...]
    level: np.ndarray  # (N,) level index
    local: np.ndarray  # (N,) component id inside that level's label map (1-based)
    parent: np.ndarray  # (N,) parent node index, -1 for roots (level 0)
    features: np.ndarray  # (N, len(FEATURES)) float32
    merge_gap: float = 0.0
    sum_iou: np.ndarray | None = None  # (N,) training target: matched IoU summed over records
    n_match: np.ndarray | None = None  # (N,) training target: number of records matched


def _disk_mask(shape: tuple[int, int], frac: float = 0.95) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx - DISK_CENTER[0]) ** 2 + (yy - DISK_CENTER[1]) ** 2 <= (frac * DISK_RADIUS) ** 2


_DISK = None


def level_labels(prob: np.ndarray, threshold: float, merge_gap: float = 0.0) -> tuple[int, np.ndarray]:
    """Connected components of ``prob > threshold`` (optionally grouping blobs within ``merge_gap``)."""
    binary = (prob > threshold).astype(np.uint8)
    if merge_gap > 0:
        dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        n, labels = cv2.connectedComponents((dist <= merge_gap / 2).astype(np.uint8), connectivity=8)
        labels[binary == 0] = 0
        # grouping can leave label ids without pixels? no: every group contains a seed pixel
    else:
        n, labels = cv2.connectedComponents(binary, connectivity=8)
    return n, labels


def build_tree(
    prob: np.ndarray,
    image: np.ndarray,
    levels: tuple[float, ...] = LEVELS,
    merge_gap: float = 0.0,
    keep_maps: bool = False,
    records: list[tuple[np.ndarray, np.ndarray]] | None = None,
):
    """Build the candidate forest and its node features.

    Returns ``(tree, maps)``; ``maps`` is the list of per-level label maps if ``keep_maps`` else None.
    Only two label maps are held at a time unless ``keep_maps``.  If ``records`` (per annotator:
    flat indices and labels of its reference pixels) is given, the training targets are computed in
    the same pass and stored on the tree.
    """
    global _DISK
    if _DISK is None or _DISK.shape != prob.shape:
        _DISK = _disk_mask(prob.shape)
    h, w = prob.shape
    on_disk = image[_DISK]
    disk_median = float(np.median(on_disk if on_disk.size else image)) or 1.0
    p_flat, img_flat = prob.ravel(), image.ravel().astype(np.float64)

    lvl_l, loc_l, par_l, feat_l, maps, si_l, nm_l = [], [], [], [], [], [], []
    prev_labels, prev_offset, prev_area, offset = None, 0, None, 0
    level_area_05 = None
    for k, t in enumerate(levels):
        n, labels = level_labels(prob, t, merge_gap)
        m = n - 1  # number of components at this level
        flat = labels.ravel()
        idx = np.flatnonzero(flat)
        lab = flat[idx]
        if keep_maps:
            maps.append(labels)
        if abs(t - 0.5) < 1e-6:
            level_area_05 = len(idx)
        if m == 0:
            prev_labels, prev_offset, prev_area = labels, offset, np.zeros(0)
            continue
        p = p_flat[idx]
        area = np.bincount(lab, minlength=n)[1:].astype(np.float64)
        psum = np.bincount(lab, weights=p, minlength=n)[1:]
        p2sum = np.bincount(lab, weights=p * p, minlength=n)[1:]
        mean_p = psum / area
        std_p = np.sqrt(np.maximum(p2sum / area - mean_p**2, 0))
        ys, xs = np.divmod(idx, w)
        # per-component max / bounding box from the foreground pixels only (sorting the whole
        # 2048 x 2048 map, as scipy.ndimage.maximum does, is ~20x slower)
        order = np.argsort(lab, kind="stable")
        starts = np.flatnonzero(np.r_[True, lab[order][1:] != lab[order][:-1]])
        max_p = np.maximum.reduceat(p[order], starts).astype(np.float64)
        bbox_w = np.maximum.reduceat(xs[order], starts) - np.minimum.reduceat(xs[order], starts) + 1
        bbox_h = np.maximum.reduceat(ys[order], starts) - np.minimum.reduceat(ys[order], starts) + 1
        bbox_area = (bbox_w * bbox_h).astype(np.float64)
        xs, ys = xs.astype(np.float64), ys.astype(np.float64)
        mx = np.bincount(lab, weights=xs, minlength=n)[1:] / area
        my = np.bincount(lab, weights=ys, minlength=n)[1:] / area
        cxx = np.bincount(lab, weights=xs * xs, minlength=n)[1:] / area - mx**2
        cyy = np.bincount(lab, weights=ys * ys, minlength=n)[1:] / area - my**2
        cxy = np.bincount(lab, weights=xs * ys, minlength=n)[1:] / area - mx * my
        tr, det = cxx + cyy, cxx * cyy - cxy**2
        disc = np.sqrt(np.maximum(tr**2 / 4 - det, 0))
        lam1, lam2 = tr / 2 + disc + 1 / 12, np.maximum(tr / 2 - disc, 0) + 1 / 12  # +1/12: pixel extent
        length = 4 * np.sqrt(lam1)
        intensity = np.bincount(lab, weights=img_flat[idx], minlength=n)[1:] / area
        radius = np.hypot(mx - DISK_CENTER[0], my - DISK_CENTER[1]) / DISK_RADIUS
        # ring of background pixels within RING_PX of a component, assigned to the nearest
        # component: nearest[] numbers the foreground ("zero") pixels in raster order, i.e. the
        # order of ``idx``, so nearest - 1 indexes ``lab`` directly
        dist, nearest = cv2.distanceTransformWithLabels(
            (labels == 0).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        dflat = dist.ravel()
        ring = np.flatnonzero((dflat > 0) & (dflat <= RING_PX))
        ring_lab = lab[nearest.ravel()[ring] - 1]
        ring_n = np.maximum(np.bincount(ring_lab, minlength=n)[1:], 1)
        ring_p = np.bincount(ring_lab, weights=p_flat[ring], minlength=n)[1:] / ring_n
        ring_int = np.bincount(ring_lab, weights=img_flat[ring], minlength=n)[1:] / ring_n
        ring_contrast = intensity / np.maximum(ring_int, 1.0)

        if k == 0 or prev_labels is None or prev_area is None or len(prev_area) == 0:
            parent = np.full(m, -1, np.int64)
            parent_ratio = np.zeros(m)
        else:
            _, first = np.unique(lab, return_index=True)  # one pixel of every component
            parent_local = prev_labels.ravel()[idx[first]]
            parent = prev_offset + parent_local.astype(np.int64) - 1
            parent_ratio = np.log(area / prev_area[parent_local - 1])

        if records is not None:
            si, nm = _level_targets(flat, n, area, records)
            si_l.append(si)
            nm_l.append(nm)
        lvl_l.append(np.full(m, k, np.int16))
        loc_l.append(np.arange(1, n, dtype=np.int32))
        par_l.append(parent)
        feat_l.append(np.column_stack([
            np.full(m, t), np.log(area), mean_p, max_p, std_p, mean_p - t, np.log(length),
            area / length, np.sqrt(lam1 / lam2), area / bbox_area, radius, intensity / disk_median,
            parent_ratio, np.zeros(m), np.zeros(m), np.zeros(m),
            np.full(m, np.log1p(m)), area / len(idx), np.zeros(m),
            ring_p, ring_contrast, np.zeros(m), np.zeros(m), np.zeros(m),
        ]))
        prev_labels, prev_offset, prev_area = labels, offset, area
        offset += m

    if not lvl_l:
        empty = np.zeros(0, np.int64)
        return Tree(levels, empty.astype(np.int16), empty.astype(np.int32), empty,
                    np.zeros((0, len(FEATURES)), np.float32), merge_gap), (maps if keep_maps else None)

    level = np.concatenate(lvl_l)
    parent = np.concatenate(par_l)
    feats = np.concatenate(feat_l)
    n_nodes = len(level)
    has_parent = parent >= 0
    n_children = np.bincount(parent[has_parent], minlength=n_nodes)
    feats[:, FEATURES.index("n_children")] = n_children
    feats[has_parent, FEATURES.index("n_siblings")] = n_children[parent[has_parent]]
    depth = np.zeros(n_nodes)
    for v in np.argsort(-level, kind="stable"):  # deepest first
        if parent[v] >= 0:
            depth[parent[v]] = max(depth[parent[v]], depth[v] + 1)
    feats[:, FEATURES.index("depth_below")] = depth
    feats[:, FEATURES.index("log_image_activity")] = np.log1p(level_area_05 or 0)
    area = np.exp(feats[:, FEATURES.index("log_area")].astype(np.float64))
    child_area = np.bincount(parent[has_parent], weights=area[has_parent], minlength=n_nodes)
    feats[:, FEATURES.index("child_area_frac")] = child_area / area  # boundary sharpness
    mean_p_all = feats[:, FEATURES.index("mean_p")]
    feats[has_parent, FEATURES.index("parent_mean_p")] = mean_p_all[parent[has_parent]]
    stability = np.zeros(n_nodes)  # levels above this one over which it stays a single piece
    for k in range(int(level.max()), 0, -1):
        v = np.flatnonzero((level == k) & has_parent)
        single = n_children[parent[v]] == 1
        stability[parent[v][single]] = stability[v[single]] + 1
    feats[:, FEATURES.index("stability")] = stability
    tree = Tree(levels, level, np.concatenate(loc_l), parent, feats.astype(np.float32), merge_gap)
    if records is not None:
        tree.sum_iou, tree.n_match = np.concatenate(si_l), np.concatenate(nm_l)
    return tree, (maps if keep_maps else None)


def _level_targets(flat: np.ndarray, n: int, area_c: np.ndarray, records) -> tuple[np.ndarray, np.ndarray]:
    """Matched IoU summed over records, and number of records matched, for each component 1..n-1."""
    sum_iou, n_match = np.zeros(n - 1), np.zeros(n - 1, np.int32)
    for gt_idx, gt_lab in records:
        n_gt = int(gt_lab.max()) if gt_lab.size else 0
        if n_gt == 0:
            continue
        at = flat[gt_idx].astype(np.int64)
        joint = np.bincount(gt_lab * n + at, minlength=(n_gt + 1) * n).reshape(n_gt + 1, n)
        inter = joint[1:, 1:].astype(np.float64)
        area_g = np.bincount(gt_lab, minlength=n_gt + 1)[1:, None].astype(np.float64)
        union = area_g + area_c[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0).max(axis=0)
        hit = iou > 0.5
        sum_iou += np.where(hit, iou, 0.0)
        n_match += hit
    return sum_iou, n_match


def node_targets(
    prob: np.ndarray, tree: Tree, records: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """For every node: summed matched IoU over records and number of records it matches.

    ``records``: per annotator record, (flat indices, labels) of its reference foreground pixels.
    The level label maps are recomputed one at a time to keep memory low.
    """
    n_nodes = len(tree.level)
    sum_iou, n_match = np.zeros(n_nodes), np.zeros(n_nodes, np.int32)
    for k, t in enumerate(tree.levels):
        nodes = np.flatnonzero(tree.level == k)
        if len(nodes) == 0:
            continue
        n, labels = level_labels(prob, t, tree.merge_gap)
        flat = labels.ravel()
        area_c = np.bincount(flat[flat > 0], minlength=n)[1:].astype(np.float64)
        si, nm = _level_targets(flat, n, area_c, records)
        sum_iou[nodes] = si[tree.local[nodes] - 1]
        n_match[nodes] = nm[tree.local[nodes] - 1]
    return sum_iou, n_match


def select(level: np.ndarray, parent: np.ndarray, score: np.ndarray, tau: float, min_instances: int = 1) -> np.ndarray:
    """Exact best antichain: boolean mask of selected nodes maximising sum(score - tau).

    Works on any concatenation of independent forests (e.g. a whole bank of images) as long as
    parents always sit exactly one level below their children.  ``min_instances`` is applied by
    the caller per image (see :func:`ensure_min`).
    """
    n = len(level)
    gain = score - tau
    child_sum = np.zeros(n)
    best = np.zeros(n)
    take = np.zeros(n, bool)
    max_level = int(level.max()) if n else -1
    by_level = [np.flatnonzero(level == k) for k in range(max_level + 1)]
    for k in range(max_level, -1, -1):  # bottom-up
        v = by_level[k]
        take[v] = (gain[v] > child_sum[v]) & (gain[v] > 0)
        best[v] = np.maximum(np.maximum(gain[v], child_sum[v]), 0)
        has_p = parent[v] >= 0
        np.add.at(child_sum, parent[v][has_p], best[v][has_p])
    selected = np.zeros(n, bool)
    open_ = np.zeros(n, bool)
    for k in range(max_level + 1):  # top-down
        v = by_level[k]
        roots = parent[v] < 0
        open_[v[roots]] = True
        child = v[~roots]
        open_[child] = open_[parent[child]] & ~take[parent[child]]
        selected[v] = open_[v] & take[v]
    return selected


def ensure_min(selected: np.ndarray, image_id: np.ndarray, score: np.ndarray, min_instances: int = 1) -> np.ndarray:
    """Per image, if fewer than ``min_instances`` nodes are selected, add the best-scoring one(s)."""
    if min_instances <= 0 or len(score) == 0:
        return selected
    selected = selected.copy()
    counts = np.bincount(image_id, weights=selected, minlength=int(image_id.max()) + 1)
    for i in np.flatnonzero(counts < min_instances):
        nodes = np.flatnonzero(image_id == i)
        if len(nodes):
            selected[nodes[np.argmax(score[nodes])]] = True  # one node; disjointness is preserved
    return selected


def decide_image(prob: np.ndarray, image: np.ndarray, decision: dict, scorer=None) -> np.ndarray:
    """Apply a trained decision layer to one image; returns the int32 instance label map.

    ``decision`` is the payload saved by :mod:`decision_train` (scorer, tau, levels, merge_gap).
    ``scorer`` overrides the payload's scorer (used for honest out-of-fold evaluation).
    """
    tree, maps = build_tree(prob, image, decision["levels"], decision["merge_gap"], keep_maps=True)
    score = (scorer or decision["scorer"]).predict(tree.features)
    sel = select(tree.level, tree.parent, score, decision["tau"])
    sel = ensure_min(sel, np.zeros(len(sel), np.int64), score, 1)
    labels = labels_from_selection(tree, maps, sel)
    if labels.max() == 0 and prob.max() > 0:
        # no candidate above the lowest level: take the most confident region instead (every
        # training image contains a filament, so an empty prediction is almost surely wrong)
        from .postprocess import PostprocessParams, probs_to_labels

        labels = probs_to_labels(prob, PostprocessParams(threshold=1.0, min_area=0, min_instances=1))
    return labels


def labels_from_selection(tree: Tree, maps: list[np.ndarray], selected: np.ndarray) -> np.ndarray:
    """Pixel-disjoint int32 label map of the selected nodes (1..K)."""
    out = np.zeros(maps[0].shape if maps else (2048, 2048), np.int32)
    next_id = 1
    for k in np.unique(tree.level[selected]):
        nodes = np.flatnonzero(selected & (tree.level == k))
        lut = np.zeros(int(maps[k].max()) + 1, np.int32)
        lut[tree.local[nodes]] = np.arange(next_id, next_id + len(nodes), dtype=np.int32)
        part = lut[maps[k]]
        out[part > 0] = part[part > 0]
        next_id += len(nodes)
    return out


class HurdleScorer:
    """u_hat = P(match) x E[IoU | match], both gradient-boosted trees on the node features."""

    # Strong regularisation: every filament appears as a near-identical candidate at many levels,
    # so the effective number of independent training examples is far smaller than the row count.
    PARAMS = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=300, l2_regularization=1.0)

    def __init__(self, seed: int = 0, **overrides):
        common = dict(self.PARAMS, early_stopping=False, random_state=seed)
        common.update(overrides)
        self.h = HistGradientBoostingClassifier(**common)
        self.m = HistGradientBoostingRegressor(**common)

    def fit(self, X, sum_iou, n_match, R):
        miss = R - n_match
        Xh = np.concatenate([X, X])
        yh = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
        wh = np.concatenate([n_match, miss]).astype(np.float64)
        keep = wh > 0
        self.h.fit(Xh[keep], yh[keep], sample_weight=wh[keep])
        hit = n_match > 0
        self.m.fit(X[hit], sum_iou[hit] / n_match[hit], sample_weight=n_match[hit].astype(np.float64))
        return self

    def predict(self, X):
        if len(X) == 0:
            return np.zeros(0)
        return self.h.predict_proba(X)[:, 1] * np.clip(self.m.predict(X), 0.5, 1.0)
