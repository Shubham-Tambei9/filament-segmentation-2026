"""Build the out-of-fold candidate bank, fit the candidate scorer, tune tau, save the decision layer.

Example:
    python -m filament_seg.decision_train --prob-dirs work/runs/resnet34_s1024_f0/val_probs \
        work/runs/resnet34_s1024_f1/val_probs --out weights/decision.pkl

The bank stores, for every candidate node of every out-of-fold image, its features, its summed
matched IoU over the image's annotator records and the number of records it matches.  The exact
pooled PQ of *any* selection then follows from sums over the bank (see :mod:`decision`), so the
scorer can be cross-fitted and tau tuned without re-rasterising a single mask.

Scorer ("hurdle" model): u_hat = P(an annotator's filament is matched) x E[IoU | matched], with a
gradient-boosting classifier for the first factor (fractional labels via sample weights) and a
gradient-boosting regressor for the second.  Cross-fitting: the scores used to evaluate a fold come
from models that never saw that fold's images.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .annotations import label_map, load_annotations, make_folds
from .imaging import read_gray, upsample_prob
from .decision import FEATURES, LEVELS, HurdleScorer, build_tree, ensure_min, select
from .paths import TRAIN_IMAGES, TRAIN_JSON

_ITEMS = None


def _init_worker():
    global _ITEMS
    _ITEMS = {it.stem: it for it in load_annotations(TRAIN_JSON)}


def _bank_one(job):
    stem, prob_file, merge_gap, cache_dir = job
    cache = Path(cache_dir) / f"{stem}.pkl" if cache_dir else None
    if cache is not None and cache.exists():
        return pickle.loads(cache.read_bytes())
    result = _bank_compute(stem, prob_file, merge_gap)
    if cache is not None:
        cache.write_bytes(pickle.dumps(result))
    return result


def _bank_compute(stem, prob_file, merge_gap):
    item = _ITEMS[stem]
    prob = upsample_prob(np.load(prob_file))
    records = []
    for filaments in item.annotations.values():
        flat = label_map(filaments).ravel()
        idx = np.flatnonzero(flat)
        records.append((idx, flat[idx].astype(np.int64)))
    # features and targets in one pass over the threshold levels
    tree, _ = build_tree(prob, read_gray(TRAIN_IMAGES / item.file_name), LEVELS, merge_gap, records=records)
    n_nodes = len(tree.level)
    sum_iou = tree.sum_iou if tree.sum_iou is not None else np.zeros(n_nodes)
    n_match = tree.n_match if tree.n_match is not None else np.zeros(n_nodes, np.int32)
    n_gt = sum(len(f) for f in item.annotations.values())
    return stem, tree.level, tree.parent, tree.features, sum_iou, n_match, len(records), n_gt


def build_bank(prob_files: dict[str, str], merge_gap: float, workers: int, cache_dir: str | None = None) -> dict:
    """Candidate bank of all images.  With ``cache_dir`` each image's result is cached on disk, so
    adding folds later only processes the new images (the cache must belong to one set of maps
    and one ``merge_gap``)."""
    stems = sorted(prob_files)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with Pool(workers, initializer=_init_worker) as pool:
        parts = pool.map(_bank_one, [(s, prob_files[s], merge_gap, cache_dir) for s in stems], chunksize=2)
    image_id, level, parent, feats, sum_iou, n_match, n_rec, n_gt, offset = [], [], [], [], [], [], [], [], 0
    for i, (stem, lv, par, f, si, nm, r, g) in enumerate(parts):
        image_id.append(np.full(len(lv), i, np.int32))
        level.append(lv)
        parent.append(np.where(par >= 0, par + offset, -1))
        feats.append(f)
        sum_iou.append(si)
        n_match.append(nm)
        n_rec.append(r)
        n_gt.append(g)
        offset += len(lv)
    return {
        "stems": [p[0] for p in parts], "image_id": np.concatenate(image_id), "level": np.concatenate(level),
        "parent": np.concatenate(parent), "X": np.concatenate(feats), "sum_iou": np.concatenate(sum_iou),
        "n_match": np.concatenate(n_match), "R": np.array(n_rec), "n_gt": np.array(n_gt), "merge_gap": merge_gap,
    }


def bank_pq(bank: dict, selected: np.ndarray) -> dict:
    """Exact pooled PQ statistics of a node selection (see :mod:`decision`)."""
    R_node = bank["R"][bank["image_id"]]
    tp = float(bank["n_match"][selected].sum())
    num = float(bank["sum_iou"][selected].sum())
    n_pred = float(R_node[selected].sum())  # predictions counted once per record
    fp, fn = n_pred - tp, float(bank["n_gt"].sum()) - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    return {"PQ": num / denom if denom else 0.0, "SQ": num / tp if tp else 0.0, "RQ": tp / denom if denom else 0.0,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "instances": int(selected.sum())}


def decide(bank: dict, score: np.ndarray, tau: float, min_instances: int = 1) -> np.ndarray:
    sel = select(bank["level"], bank["parent"], score, tau)
    return ensure_min(sel, bank["image_id"], score, min_instances)


def cross_fit(bank: dict, folds: np.ndarray, seed: int = 0, return_models: bool = False, **params):
    """Out-of-fold scores: each image is scored by a model trained on the other folds' images.

    With ``return_models`` also returns ``{fold: scorer}`` so out-of-fold predictions can be
    reproduced image by image (e.g. for the validation report).
    """
    node_fold = folds[bank["image_id"]]
    R_node = bank["R"][bank["image_id"]]
    score, models = np.zeros(len(node_fold)), {}
    for f in np.unique(node_fold):
        tr, te = node_fold != f, node_fold == f
        model = HurdleScorer(seed, **params).fit(bank["X"][tr], bank["sum_iou"][tr], bank["n_match"][tr], R_node[tr])
        score[te] = model.predict(bank["X"][te])
        models[int(f)] = model
    return (score, models) if return_models else score


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prob-dirs", nargs="+", required=True, help="directories of out-of-fold <stem>.npy maps")
    p.add_argument("--merge-gap", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--bank", default=None, help="cache file for the bank (.pkl); reused if it exists")
    p.add_argument("--cache-dir", default=None, help="per-image candidate cache (incremental bank building)")
    p.add_argument("--out", default=None, help="save fitted scorer + tau here (.pkl)")
    args = p.parse_args(argv)

    t0 = time.time()
    if args.bank and Path(args.bank).exists():
        bank = pickle.loads(Path(args.bank).read_bytes())
    else:
        prob_files = {f.stem: str(f) for d in args.prob_dirs for f in Path(d).glob("*.npy")}
        bank = build_bank(prob_files, args.merge_gap, args.workers, args.cache_dir)
        if args.bank:
            Path(args.bank).parent.mkdir(parents=True, exist_ok=True)
            Path(args.bank).write_bytes(pickle.dumps(bank))
    print(f"bank: {len(bank['stems'])} images, {len(bank['level'])} candidate nodes "
          f"({time.time() - t0:.0f}s)", flush=True)

    items = load_annotations(TRAIN_JSON)
    fold_of = dict(zip((it.stem for it in items), make_folds(items, 5, 42)))
    folds = np.array([fold_of[s] for s in bank["stems"]])

    # sanity reference: oracle selection with the true expected IoU (an upper bound)
    R_node = bank["R"][bank["image_id"]]
    oracle = bank["sum_iou"] / R_node
    oracle_pq = max(bank_pq(bank, decide(bank, oracle, t))["PQ"] for t in np.arange(0.1, 0.4, 0.02))
    print(f"oracle selection (true expected IoU, upper bound): PQ {oracle_pq:.4f}")

    score, fold_models = cross_fit(bank, folds, return_models=True)
    results = []
    for tau in np.round(np.arange(0.12, 0.40, 0.01), 2):
        s = bank_pq(bank, decide(bank, score, tau))
        results.append((s["PQ"], tau, s))
    results.sort(key=lambda r: -r[0])
    for pq, tau, s in results[:6]:
        print(f"tau {tau:.2f}: PQ {pq:.4f} SQ {s['SQ']:.3f} RQ {s['RQ']:.3f} TP {s['TP']} FP {s['FP']} FN {s['FN']}")
    best_pq, best_tau, _ = results[0]
    print(f"cross-fitted decision layer: PQ {best_pq:.4f} at tau {best_tau} ({time.time() - t0:.0f}s total)")

    if args.out:
        final = HurdleScorer().fit(bank["X"], bank["sum_iou"], bank["n_match"], R_node)
        payload = {"scorer": final, "fold_scorers": fold_models, "tau": float(best_tau), "levels": LEVELS,
                   "merge_gap": bank["merge_gap"], "features": FEATURES, "cv_pq": best_pq,
                   "n_images": len(bank["stems"])}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes(pickle.dumps(payload))
        Path(args.out).with_suffix(".json").write_text(json.dumps(
            {k: v for k, v in payload.items() if k not in ("scorer", "fold_scorers")}, indent=2, default=float))
        print("saved", args.out)


if __name__ == "__main__":
    main()
