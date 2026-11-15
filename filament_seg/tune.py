"""Grid-search post-processing parameters on saved validation probabilities.

Example (pool the out-of-fold predictions of several runs):
    python -m filament_seg.tune --runs resnet34_s1024_f0 resnet34_s1024_f1 --out weights/postprocess.json

Tuning on held-out folds only (never on the images a model was trained on) keeps the chosen
threshold / size filter honest.  For speed, each candidate is scored from foreground pixels only
(:func:`metrics.match_sparse`) and images are split across worker processes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .annotations import label_map, load_annotations
from .imaging import upsample_prob
from .metrics import PQAccumulator
from .paths import TRAIN_JSON, WORK_DIR
from .postprocess import PostprocessParams, components, keep_remap

GRID = {
    "threshold": [0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    "merge_gap": [0, 8, 16, 24, 32, 48, 64],
    "min_area": [0, 100, 200, 300, 500, 800],
    "min_mean_prob": [0.0, 0.55, 0.6, 0.65, 0.7],
    "min_instances": [0, 1],
}


def _evaluate_chunk(job: tuple[list[str], dict[str, str], dict]) -> dict[tuple, PQAccumulator]:
    """Score every grid point on a subset of images (runs in a worker process)."""
    stems, prob_files, grid = job
    items = {it.stem: it for it in load_annotations(TRAIN_JSON)}
    accs: dict[tuple, PQAccumulator] = defaultdict(PQAccumulator)
    for stem in stems:
        gts = []  # per annotator: (flat indices, labels) of GT foreground pixels
        for filaments in items[stem].annotations.values():
            flat = label_map(filaments).ravel()
            idx = np.flatnonzero(flat)
            gts.append((idx, flat[idx].astype(np.int64)))
        prob = upsample_prob(np.load(prob_files[stem]))
        for thr in grid["threshold"]:
            for gap in grid["merge_gap"]:
                comp = components(prob, thr, gap)
                comp_flat = comp.labels.ravel()
                comp_at_gt = [comp_flat[idx] for idx, _ in gts]
                for min_area in grid["min_area"]:
                    for mmp in grid["min_mean_prob"]:
                        if mmp and mmp <= thr:
                            continue  # filter would be a no-op
                        for min_inst in grid["min_instances"]:
                            remap = keep_remap(comp, min_area, mmp, min_inst)
                            pred_lab = remap[comp.fg_lab]
                            pred_lab = pred_lab[pred_lab > 0]
                            acc = accs[(thr, gap, min_area, mmp, min_inst)]  # PostprocessParams field order
                            for (_, gt_lab), at_gt in zip(gts, comp_at_gt):
                                acc.update_sparse(gt_lab, remap[at_gt], pred_lab)
    return dict(accs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=[], help="run names under work/runs/ (uses their val_probs/)")
    p.add_argument("--prob-dirs", nargs="*", default=[], help="extra directories of <stem>.npy probability maps")
    p.add_argument("--out", default=None, help="where to write the best parameters (JSON)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--grid", default=None, help='JSON overriding GRID entries, e.g. \'{"threshold": [0.3, 0.35]}\'')
    args = p.parse_args(argv)
    grid = {**GRID, **(json.loads(args.grid) if args.grid else {})}

    dirs = [WORK_DIR / "runs" / r / "val_probs" for r in args.runs] + [Path(d) for d in args.prob_dirs]
    if not dirs:
        p.error("give --runs and/or --prob-dirs")
    prob_files = {f.stem: str(f) for d in dirs for f in d.glob("*.npy")}
    stems = sorted(prob_files)
    print(f"tuning on {len(stems)} validation images from {len(dirs)} director(ies), {args.workers} workers")

    chunks = [stems[i :: args.workers] for i in range(args.workers)]
    with Pool(args.workers) as pool:
        partials = pool.map(_evaluate_chunk, [(c, prob_files, grid) for c in chunks])
    totals: dict[tuple, PQAccumulator] = defaultdict(PQAccumulator)
    for part in partials:
        for key, acc in part.items():
            totals[key].merge(acc)

    results = sorted(
        ((acc.summary(), PostprocessParams(*key)) for key, acc in totals.items()), key=lambda r: -r[0]["PQ"]
    )
    for s, params in results[:15]:
        print(f"PQ {s['PQ']:.4f} SQ {s['SQ']:.3f} RQ {s['RQ']:.3f} TP {s['TP']} FP {s['FP']} FN {s['FN']} "
              f"1:n {s['one_to_many']} n:1 {s['many_to_one']} | {params.as_dict()}")
    best_summary, best = results[0]
    print("\nbest:", best.as_dict(), json.dumps(best_summary))

    results_path = Path(args.out).with_suffix(".results.jsonl") if args.out else dirs[0].parent / "tune_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        for s, params in results:
            f.write(json.dumps({**params.as_dict(), **s}) + "\n")
    print("all grid results:", results_path)
    if args.out:
        Path(args.out).write_text(json.dumps(best.as_dict(), indent=2))


if __name__ == "__main__":
    main()
