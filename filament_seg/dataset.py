"""Training data: cached resized images + soft consensus targets, and augmentations.

Rasterising polygons and decoding 2048 x 2048 JPEGs every step is slow, so images and soft
targets are resized once to the training resolution and cached as ``.npy`` files under
``work/cache/size<N>/``.  Targets are rasterised at full resolution and then area-resized,
which gives anti-aliased soft edges for the thin filaments.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .annotations import AnnotatedImage, consensus_mask
from .imaging import read_gray, resize  # noqa: F401  (re-exported for existing callers)

# Global intensity normalisation (H-alpha JPEGs: mean ~84/255, std ~57/255 incl. off-disk background)
PIXEL_MEAN = 0.33
PIXEL_STD = 0.22


def normalize(img_u8: np.ndarray) -> np.ndarray:
    return ((img_u8.astype(np.float32) / 255.0) - PIXEL_MEAN) / PIXEL_STD


def build_cache(items: list[AnnotatedImage], image_dir: Path, cache_dir: Path, size: int) -> None:
    """Write ``<stem>_img.npy`` (uint8) and ``<stem>_tgt.npy`` (uint8, agreement x 255)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for it in tqdm(items, desc=f"caching {size}px"):
        img_p, tgt_p = cache_dir / f"{it.stem}_img.npy", cache_dir / f"{it.stem}_tgt.npy"
        if img_p.exists() and tgt_p.exists():
            continue
        np.save(img_p, resize(read_gray(image_dir / it.file_name), size))
        tgt = resize(consensus_mask(it), size)  # float32 in [0, 1]
        np.save(tgt_p, np.round(tgt * 255).astype(np.uint8))


def augment(img: np.ndarray, tgt: np.ndarray, rng: np.random.Generator):
    """Geometric + photometric augmentation (numpy / OpenCV only).

    The Sun has no preferred orientation in the image plane for this task, so arbitrary
    rotations about the disk centre and flips are label-preserving; small scale jitter
    simulates the ~10% variation in apparent disk radius between sites/seasons.
    """
    h, w = img.shape
    if rng.random() < 0.5:
        img, tgt = img[:, ::-1], tgt[:, ::-1]
    if rng.random() < 0.5:
        img, tgt = img[::-1], tgt[::-1]
    angle = rng.uniform(0, 360)
    scale = rng.uniform(0.9, 1.1)
    shift = rng.uniform(-0.02, 0.02, size=2) * w
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += shift
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    tgt = cv2.warpAffine(np.ascontiguousarray(tgt), m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    x = img.astype(np.float32) / 255.0
    if rng.random() < 0.8:  # contrast / brightness / gamma
        x = np.clip(x * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08), 0, 1)
        x = x ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.2:  # seeing / focus variation between sites
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        x = x + rng.normal(0, rng.uniform(0.005, 0.03), x.shape).astype(np.float32)
    img = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return img, tgt


class FilamentDataset(Dataset):
    def __init__(self, stems: list[str], cache_dir: Path, train: bool):
        self.stems = stems
        self.cache_dir = Path(cache_dir)
        self.train = train
        self._rng: np.random.Generator | None = None
        self._rng_pid = -1

    def __len__(self) -> int:
        return len(self.stems)

    def rng(self) -> np.random.Generator:
        """One augmentation RNG per process (main or DataLoader worker).

        Seeded from ``torch.initial_seed()``, which PyTorch makes distinct per worker and
        derives from ``torch.manual_seed``, so runs are reproducible.  The generator lives as
        long as the (persistent) worker, so every epoch draws fresh augmentations.
        """
        if self._rng is None or self._rng_pid != os.getpid():
            self._rng = np.random.default_rng(torch.initial_seed() % 2**63)
            self._rng_pid = os.getpid()
        return self._rng

    def __getitem__(self, i: int):
        stem = self.stems[i]
        img = np.load(self.cache_dir / f"{stem}_img.npy")
        tgt = np.load(self.cache_dir / f"{stem}_tgt.npy")
        if self.train:
            img, tgt = augment(img, tgt, self.rng())
        x = torch.from_numpy(normalize(img))[None]
        y = torch.from_numpy(tgt.astype(np.float32) / 255.0)[None]
        return x, y
