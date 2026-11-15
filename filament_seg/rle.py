"""COCO run-length encoding used by the submission file.

The submission stores only the ``counts`` string of a pycocotools RLE; the size is implicitly
2048 x 2048.  pycocotools expects column-major (Fortran-order) uint8 masks.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from .annotations import IMAGE_SIZE


def encode(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def decode(counts: str, size: int = IMAGE_SIZE) -> np.ndarray:
    return mask_utils.decode({"size": [size, size], "counts": counts.encode("ascii")}).astype(bool)
