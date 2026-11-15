"""Predict test images and write the Kaggle submission CSV.

Example:
    python -m filament_seg.predict --weights weights/model_f0.pt --params weights/postprocess.json \
        --out submission.csv

Several ``--weights`` files are averaged (probability ensemble).  One CSV row per predicted
filament: ``filament_id`` = ``<image stem>_<k>`` (k = 1..K), ``segmentation_rle`` = COCO RLE counts.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from . import rle
from .dataset import read_gray
from .decision import decide_image
from .inference import predict_prob, upsample_prob
from .model import load_checkpoint
from .paths import TEST_IMAGES
from .postprocess import PostprocessParams, labels_to_masks, probs_to_labels


def resize_prob(prob: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(prob.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--params", default=None, help="JSON with PostprocessParams fields (fixed-rule instances)")
    p.add_argument("--decision", default=None, help="decision layer .pkl from decision_train (overrides --params)")
    p.add_argument("--refiner", default=None, help="instance refiner .pt from refine_train, applied after the rules")
    p.add_argument("--images", default=str(TEST_IMAGES))
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--tta", default="d4", choices=["none", "flip4", "d4"], help="test-time augmentation views")
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N images")
    p.add_argument("--prob-cache", default=None,
                   help="directory caching the ensemble probability maps (float16); reused when present, so "
                        "post-processing variants can be written without re-running the networks")
    p.add_argument("--maps-only", action="store_true", help="only fill --prob-cache, write no CSV")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_checkpoint(w, device) for w in args.weights]
    params = PostprocessParams(**json.loads(Path(args.params).read_text())) if args.params else PostprocessParams()
    decision = pickle.loads(Path(args.decision).read_bytes()) if args.decision else None
    if decision:
        print(f"decision layer: tau {decision['tau']}, {len(decision['levels'])} levels, cv PQ {decision['cv_pq']:.4f}")
    else:
        print("post-processing:", params.as_dict())

    refiner = None
    if args.refiner:
        from .refine_train import build_refiner, refine_labels

        ckpt = torch.load(args.refiner, map_location="cpu", weights_only=False)
        refiner = build_refiner(pretrained=False)
        refiner.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
        refiner = refiner.to(device).eval()
        print(f"refiner: threshold {ckpt['threshold']}, cv PQ {ckpt['cv_pq']:.4f} (rules {ckpt['rules_pq']:.4f})")

    def to_labels(prob: np.ndarray, img: np.ndarray) -> np.ndarray:
        labels = probs_to_labels(prob, params) if decision is None else decide_image(prob, img, decision)
        if refiner is not None:
            labels = refine_labels(refiner, img, prob, labels, ckpt["threshold"], device)
        return labels

    files = sorted(Path(args.images).glob("*.jp*g"))
    if args.limit:
        files = files[: args.limit]
    cache = Path(args.prob_cache) if args.prob_cache else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)

    def ensemble_prob(path: Path, img: np.ndarray) -> np.ndarray:
        cached = cache / f"{path.stem}.npy" if cache is not None else None
        if cached is not None and cached.exists():
            return upsample_prob(np.load(cached))
        size = max(meta["size"] for _, meta in models)
        maps = [predict_prob(m, img, meta["size"], args.tta, device) for m, meta in models]
        maps = [p if p.shape[0] == size else resize_prob(p, size) for p in maps]
        prob = np.mean(maps, axis=0)
        if cached is not None:
            np.save(cached, prob.astype(np.float16))
        return upsample_prob(prob)

    if args.maps_only:
        for path in tqdm(files, desc="maps"):
            ensemble_prob(path, read_gray(path))
        print(f"probability maps for {len(files)} images in {cache}")
        return

    n_rows, t0 = 0, time.time()
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        for path in tqdm(files, desc="predict"):
            img = read_gray(path)
            prob = ensemble_prob(path, img)
            for k, mask in enumerate(labels_to_masks(to_labels(prob, img)), start=1):
                writer.writerow([f"{path.stem}_{k}", rle.encode(mask)])
                n_rows += 1
    dt = time.time() - t0
    print(f"wrote {n_rows} filaments for {len(files)} images to {args.out} ({dt:.0f}s, {dt / max(len(files), 1):.2f}s/image)")


if __name__ == "__main__":
    main()
