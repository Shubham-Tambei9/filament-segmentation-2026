"""Build refiner training crops from out-of-fold probability maps (CPU, torch-free, parallel).

Example:
    python -m filament_seg.refine_data --prob-dir work/oof/resnet34_s1024_d4 --out work/refine/crops_r34_1024

For every training image, the instances produced by the tuned rules (plus a looser variant, for more
varied seeds) are cropped with :func:`refine.crop_geometry`; the target is the consensus of the
annotator records that drew the matching filament.  Instances no annotator drew are skipped: the
refiner learns extent, not existence.  One ``<stem>.npz`` per image, uint8 (x255) to keep it small.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .annotations import label_map, load_annotations
from .imaging import read_gray, upsample_prob
from .paths import TRAIN_IMAGES, TRAIN_JSON, WEIGHTS_DIR
from .postprocess import PostprocessParams, probs_to_labels
from .refine import crop_geometry, crop_inputs, target_for, warp_to_crop

_ITEMS = None


def seed_variants() -> list[PostprocessParams]:
    rules = PostprocessParams(**json.loads((WEIGHTS_DIR / "postprocess.json").read_text()))
    loose = PostprocessParams(threshold=0.35, merge_gap=rules.merge_gap, min_area=100, min_mean_prob=0.7)
    return [rules, loose]


def _init():
    global _ITEMS
    _ITEMS = {it.stem: it for it in load_annotations(TRAIN_JSON)}


def _one(job):
    stem, prob_file, out_dir = job
    out = Path(out_dir) / f"{stem}.npz"
    if out.exists():
        return stem, -1
    item = _ITEMS[stem]
    prob = upsample_prob(np.load(prob_file))
    image = read_gray(TRAIN_IMAGES / item.file_name)
    records = [label_map(f) for f in item.annotations.values()]
    xs, ys, drawn, variant = [], [], [], []
    for v, params in enumerate(seed_variants()):
        labels = probs_to_labels(prob, params)
        for k in range(1, int(labels.max()) + 1):
            seed = labels == k
            target, n = target_for(seed, records)
            if n == 0:
                continue
            g = crop_geometry(seed)
            xs.append(np.round(crop_inputs(image, prob, labels, k, g) * 255).astype(np.uint8))
            ys.append(np.round(warp_to_crop(target, g) * 255).astype(np.uint8))
            drawn.append(n)
            variant.append(v)
    if xs:
        np.savez_compressed(out, x=np.stack(xs), y=np.stack(ys), drawn=np.array(drawn), variant=np.array(variant),
                            records=len(records))
    else:
        np.savez_compressed(out, x=np.zeros((0, 4, 1, 1), np.uint8), y=np.zeros((0, 1, 1), np.uint8),
                            drawn=np.zeros(0), variant=np.zeros(0), records=len(records))
    return stem, len(xs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prob-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(f.stem, str(f), args.out) for f in sorted(Path(args.prob_dir).glob("*.npy"))]
    with Pool(args.workers, initializer=_init) as pool:
        counts = [n for _, n in pool.imap_unordered(_one, jobs, chunksize=2) if n >= 0]
    print(f"{len(jobs)} images, {sum(counts)} new crops written to {args.out}")


if __name__ == "__main__":
    main()
