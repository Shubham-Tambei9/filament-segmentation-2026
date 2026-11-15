"""Sanity tests for the metric, post-processing and RLE round trip.  Run: python -m pytest tests"""
import numpy as np

from filament_seg.metrics import PQAccumulator, match_instances
from filament_seg.postprocess import PostprocessParams, probs_to_labels


def _blob(shape, y0, y1, x0, x1, k, labels=None):
    labels = np.zeros(shape, np.int32) if labels is None else labels
    labels[y0:y1, x0:x1] = k
    return labels


def test_perfect_prediction_has_pq_one():
    gt = _blob((64, 64), 5, 15, 5, 30, 1)
    gt = _blob((64, 64), 40, 50, 10, 20, 2, gt)
    acc = PQAccumulator()
    acc.update(gt, gt)
    s = acc.summary()
    assert s["PQ"] == 1.0 and s["TP"] == 2 and s["FP"] == 0 and s["FN"] == 0


def test_iou_threshold_and_counts():
    gt = _blob((64, 64), 0, 10, 0, 10, 1)  # area 100
    pred = _blob((64, 64), 0, 10, 0, 6, 1)  # IoU 0.6 -> match
    pred = _blob((64, 64), 30, 40, 30, 40, 2, pred)  # extra -> FP
    matched, n_gt, n_pred, _ = match_instances(gt, pred)
    assert n_gt == 1 and n_pred == 2 and np.allclose(matched, [0.6])
    acc = PQAccumulator()
    acc.update(gt, pred)
    s = acc.summary()
    assert s["TP"] == 1 and s["FP"] == 1 and s["FN"] == 0
    assert np.isclose(s["PQ"], 0.6 / (1 + 0.5))


def test_fragmentation_is_counted():
    gt = _blob((64, 64), 0, 10, 0, 40, 1)
    pred = _blob((64, 64), 0, 10, 0, 18, 1)
    pred = _blob((64, 64), 0, 10, 22, 40, 2, pred)
    acc = PQAccumulator()
    acc.update(gt, pred)
    s = acc.summary()
    assert s["one_to_many"] == 1 and s["TP"] == 0  # neither half reaches IoU 0.5


def test_merge_gap_groups_nearby_blobs_without_fattening():
    prob = np.zeros((64, 64), np.float32)
    prob[10:14, 5:20] = 0.9
    prob[10:14, 24:40] = 0.9  # columns 20-23 empty: a 4 px gap
    n_instances = {
        gap: int(probs_to_labels(prob, PostprocessParams(threshold=0.5, merge_gap=gap, min_area=0)).max())
        for gap in (0, 2, 4)
    }
    assert n_instances == {0: 2, 2: 2, 4: 1}
    merged = probs_to_labels(prob, PostprocessParams(threshold=0.5, merge_gap=4, min_area=0))
    assert (merged > 0).sum() == (prob > 0.5).sum()  # grouping keeps the original pixels


def test_min_instances_rescues_best_candidate():
    prob = np.zeros((64, 64), np.float32)
    prob[0:4, 0:4] = 0.6  # small, weak
    prob[20:30, 20:30] = 0.7  # larger, stronger -> should be rescued
    strict = PostprocessParams(threshold=0.5, min_area=500, min_mean_prob=0.0, min_instances=0)
    assert probs_to_labels(prob, strict).max() == 0
    rescued = probs_to_labels(prob, PostprocessParams(threshold=0.5, min_area=500, min_instances=1))
    assert rescued.max() == 1 and (rescued > 0).sum() == 100
    # nothing above the threshold at all: fall back to the most confident region
    faint = prob * 0.5  # max 0.35 < threshold 0.5
    fallback = probs_to_labels(faint, PostprocessParams(threshold=0.5, min_area=500, min_instances=1))
    assert fallback.max() == 1 and (fallback > 0).sum() == 100


def test_sparse_and_dense_matching_agree():
    from filament_seg.metrics import match_instances, match_sparse

    rng = np.random.default_rng(0)
    gt = rng.integers(0, 4, (32, 32)) * (rng.random((32, 32)) > 0.5)
    pred = rng.integers(0, 5, (32, 32)) * (rng.random((32, 32)) > 0.5)
    dense = match_instances(gt, pred)
    idx = np.flatnonzero(gt)
    sparse = match_sparse(gt.ravel()[idx], pred.ravel()[idx], pred.ravel()[np.flatnonzero(pred)])
    assert np.allclose(np.sort(dense[0]), np.sort(sparse[0])) and dense[1:3] == sparse[1:3]


def test_min_area_filter():
    prob = np.zeros((64, 64), np.float32)
    prob[0:3, 0:3] = 0.9  # 9 px
    prob[20:30, 20:30] = 0.9  # 100 px
    labels = probs_to_labels(prob, PostprocessParams(threshold=0.5, min_area=50))
    assert labels.max() == 1 and (labels > 0).sum() == 100


def test_rasterisation_matches_reference_areas():
    """Our polygon rasterisation must reproduce the organisers' (pycocotools) ``area`` field."""
    import json

    import pytest

    from filament_seg.annotations import load_annotations
    from filament_seg.paths import TRAIN_JSON

    if not TRAIN_JSON.exists():
        pytest.skip("competition data not available")
    coco = json.loads(TRAIN_JSON.read_text())
    ref = {}
    for a in coco["annotations"]:
        ref.setdefault(a["image_id"], []).append(a["area"])
    n = exact = 0
    for item in load_annotations(TRAIN_JSON)[::10]:
        for ann_id, filaments in item.annotations.items():
            for f, area in zip(filaments, ref[ann_id]):
                n += 1
                exact += int(f.mask_crop()[2].sum()) == int(area)
    assert exact / n > 0.98  # remaining cases differ by ~1 px (floating-point ties)


def test_rle_round_trip():
    from filament_seg import rle

    m = np.zeros((2048, 2048), bool)
    m[100:200, 300:350] = True
    m[1500, 7] = True
    assert np.array_equal(rle.decode(rle.encode(m)), m)


def test_blend_dirs_weights_and_sizes(tmp_path):
    from filament_seg.blend import blend_dirs

    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    a.mkdir(), b.mkdir()
    np.save(a / "x.npy", np.full((4, 4), 1.0, np.float16))
    np.save(b / "x.npy", np.zeros((8, 8), np.float16))  # different size -> resized to the larger map
    np.save(a / "y.npy", np.full((4, 4), 0.5, np.float16))  # only in group a -> weights renormalised
    assert blend_dirs(out, [(3.0, [a]), (1.0, [b])]) == 2
    x, y = np.load(out / "x.npy"), np.load(out / "y.npy")
    assert x.shape == (8, 8) and np.allclose(x, 0.75)
    assert np.allclose(y, 0.5)
