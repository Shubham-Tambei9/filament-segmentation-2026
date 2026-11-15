"""Loading the MAGFiLO COCO-style annotation file.

The same physical image (``file_name``) can be annotated independently by up to
three annotators; each annotation set is a separate COCO ``image`` entry whose id
is ``<annotator>-<file stem>``.  Everything downstream is therefore keyed by
``file_name`` and keeps the per-annotator instance lists separate.

Polygons are rasterised with pycocotools, the same code the organisers used to derive the
reference masks (pycocotools areas match the JSON ``area`` field exactly).  OpenCV's
``fillPoly`` is *not* equivalent: it includes boundary pixels and produces masks ~11% larger
(median; IoU 0.88 with the pycocotools mask), which would bias training and validation towards
over-thick filaments.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

IMAGE_SIZE = 2048  # all GONG H-alpha images in this competition are 2048 x 2048


@dataclass
class Filament:
    """One annotated filament: a single polygon (x0, y0, x1, y1, ...) in pixel coords."""

    polygon: np.ndarray  # (N, 2) float
    bbox: tuple[float, float, float, float]  # x, y, w, h
    category_id: int

    def mask(self, size: int = IMAGE_SIZE) -> np.ndarray:
        """Rasterise the polygon into a boolean ``size`` x ``size`` mask."""
        out = np.zeros((size, size), bool)
        y0, x0, crop = self.mask_crop(size)
        out[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
        return out

    def mask_crop(self, size: int = IMAGE_SIZE) -> tuple[int, int, np.ndarray]:
        """Rasterise only the polygon's bounding box: returns (y0, x0, boolean crop).

        pycocotools rasterises on a 5x up-sampled grid with ``floor(5 * x + 0.5)``, so shifting
        the polygon by whole pixels shifts the mask.  Drawing on a small canvas (1-pixel margin,
        clipped to the image) is ~8x faster than decoding a 2048 x 2048 mask per filament and
        reproduces the full-image rasterisation up to floating-point ties at half-pixel vertices:
        74 of the 8199 training filaments differ from the JSON ``area`` by 117 px in total.
        """
        pts = self.polygon * (size / IMAGE_SIZE)
        x0 = max(int(np.floor(pts[:, 0].min())) - 1, 0)
        y0 = max(int(np.floor(pts[:, 1].min())) - 1, 0)
        x1 = min(int(np.ceil(pts[:, 0].max())) + 2, size)
        y1 = min(int(np.ceil(pts[:, 1].max())) + 2, size)
        rle = mask_utils.frPyObjects([(pts - (x0, y0)).ravel().tolist()], y1 - y0, x1 - x0)
        return y0, x0, mask_utils.decode(rle)[..., 0].astype(bool)


@dataclass
class AnnotatedImage:
    """One physical image together with every annotator's instance list."""

    file_name: str
    annotations: dict[str, list[Filament]] = field(default_factory=dict)  # annotator id -> filaments

    @property
    def stem(self) -> str:
        return Path(self.file_name).stem

    @property
    def n_annotators(self) -> int:
        return len(self.annotations)


def load_annotations(json_path: str | Path) -> list[AnnotatedImage]:
    """Parse the train JSON into a list of :class:`AnnotatedImage`, sorted by file name."""
    with open(json_path) as f:
        coco = json.load(f)

    image_by_id = {im["id"]: im for im in coco["images"]}
    by_file: dict[str, AnnotatedImage] = {}
    for im in coco["images"]:
        item = by_file.setdefault(im["file_name"], AnnotatedImage(im["file_name"]))
        item.annotations.setdefault(im["id"], [])

    for ann in coco["annotations"]:
        im = image_by_id[ann["image_id"]]
        for poly in ann["segmentation"]:  # always exactly one polygon in this dataset
            by_file[im["file_name"]].annotations[ann["image_id"]].append(
                Filament(
                    polygon=np.asarray(poly, np.float64).reshape(-1, 2),
                    bbox=tuple(ann["bbox"]),
                    category_id=ann["category_id"],
                )
            )
    return [by_file[k] for k in sorted(by_file)]


def label_map(filaments: list[Filament], size: int = IMAGE_SIZE) -> np.ndarray:
    """Rasterise one annotator's filaments into a label map (0 = background, k = k-th filament).

    Filaments from one annotator essentially never overlap (a handful of pixels in the whole
    set), so writing them in order loses nothing meaningful.  uint8 keeps validation label maps
    small (at most 26 filaments per annotation set in this dataset).
    """
    labels = np.zeros((size, size), np.uint8 if len(filaments) < 256 else np.int32)
    for k, f in enumerate(filaments, start=1):
        y0, x0, crop = f.mask_crop(size)
        labels[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]][crop] = k
    return labels


def union_mask(filaments: list[Filament], size: int = IMAGE_SIZE) -> np.ndarray:
    """Boolean mask of all filaments of one annotator."""
    out = np.zeros((size, size), bool)
    for f in filaments:
        y0, x0, crop = f.mask_crop(size)
        out[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] |= crop
    return out


def consensus_mask(item: AnnotatedImage, size: int = IMAGE_SIZE) -> np.ndarray:
    """Soft target: fraction of annotators that marked each pixel as filament (float32 in [0, 1]).

    Annotators disagree strongly (inter-annotator PQ ~0.36), so instead of picking one
    annotation set we train on the per-pixel agreement level.  For single-annotator images
    this reduces to the ordinary binary mask.
    """
    acc = np.zeros((size, size), np.float32)
    for filaments in item.annotations.values():
        acc += union_mask(filaments, size)
    return acc / max(item.n_annotators, 1)


def make_folds(items: list[AnnotatedImage], n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Assign each physical image to a fold.

    Splitting is done per ``file_name`` (never per annotation set), so the same image never
    appears in both train and validation.  Folds are stratified by GONG site code (last two
    characters of the stem, e.g. ``Bh``) so every fold sees all six observatories.
    """
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(items), np.int64)
    by_site: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_site[it.stem[-2:]].append(i)
    offset = 0
    for site in sorted(by_site):
        idx = np.array(by_site[site])
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[i] = (j + offset) % n_folds
        offset += len(idx)
    return folds
