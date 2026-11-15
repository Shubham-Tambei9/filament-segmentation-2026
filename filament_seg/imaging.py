"""Image I/O and resizing helpers with no PyTorch dependency.

CPU-only stages (post-processing search, decision-layer bank) run in many worker processes;
importing torch there would commit ~2 GB of memory per worker on Windows for nothing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_SIZE = 2048  # all GONG H-alpha images in this competition are 2048 x 2048


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size:
        return img
    interp = cv2.INTER_AREA if size < img.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def upsample_prob(prob: np.ndarray) -> np.ndarray:
    """Network-resolution probability map -> float32 at the native 2048 x 2048 resolution."""
    if prob.shape[0] == IMAGE_SIZE:
        return prob.astype(np.float32)
    return cv2.resize(prob.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
