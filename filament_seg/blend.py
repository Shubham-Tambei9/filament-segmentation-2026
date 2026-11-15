"""Weighted blending of probability-map directories and the final pipeline configuration (torch-free).

A pipeline configuration (``weights/final_config.json``) lists model *groups*; each group is one
architecture whose fold models (``weights/<member>_f<k>.pt``) are averaged, and groups are blended with
the given weights.  The same weights are used for out-of-fold maps (tuning, validation report) and for
the test set, so what is validated is exactly what is submitted.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def blend_dirs(out_dir, specs: list[tuple[float, list]]) -> int:
    """Blend ``<stem>.npy`` maps: ``specs`` = [(weight, [dir, ...]), ...] (the dirs of one group are pooled).

    For every stem the groups that have it are averaged with their weights (renormalised over the groups
    present), at the size of the largest map.  Returns the number of maps written.
    """
    groups = []
    for weight, dirs in specs:
        files = {}
        for d in dirs:
            for f in Path(d).glob("*.npy"):
                files[f.stem] = f
        groups.append((float(weight), files))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(set().union(*[files.keys() for _, files in groups]))
    for stem in stems:
        maps = [(w, np.load(files[stem]).astype(np.float32)) for w, files in groups if stem in files and w > 0]
        size = max(m.shape[0] for _, m in maps)
        acc = sum(w * (m if m.shape[0] == size else cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR))
                  for w, m in maps)
        np.save(out_dir / f"{stem}.npy", (acc / sum(w for w, _ in maps)).astype(np.float16))
    return len(stems)


def load_config(path) -> dict:
    """Read a pipeline configuration; file names are resolved relative to the config's directory."""
    path = Path(path)
    cfg = json.loads(path.read_text())
    for key in ("params", "refiner"):
        if cfg.get(key):
            cfg[key] = str(path.parent / cfg[key])
    return cfg


def member_weights(cfg: dict, member: str, folds) -> list[Path]:
    """Checkpoint paths of one group member for the given folds (only those that exist)."""
    base = Path(cfg.get("weights_dir", "weights"))
    return [p for p in (base / f"{member}_f{f}.pt" for f in folds) if p.exists()]
