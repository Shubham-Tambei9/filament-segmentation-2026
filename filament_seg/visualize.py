"""Overlays of filament instances on H-alpha images (used by the notebook and report figures)."""
from __future__ import annotations

import cv2
import numpy as np

# colour-blind-safe outline colours (RGB): blue, orange, green, magenta
PALETTE = [(0, 114, 178), (230, 159, 0), (0, 158, 115), (204, 121, 167)]


def outlines(
    img_u8: np.ndarray,
    label_maps: list[np.ndarray],
    colors: list[tuple[int, int, int]] | None = None,
    scale: float = 0.5,
    thickness: int = 2,
    box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """RGB image with the instance outlines of each label map drawn in its own colour.

    Args:
        img_u8: 2048 x 2048 grayscale image.
        label_maps: label maps at the image resolution (0 = background), e.g. one per annotator
            plus the prediction.
        scale: output scale factor (applied after cropping).
        box: optional crop (x0, y0, x1, y1) in full-resolution pixels.
    """
    colors = colors or PALETTE
    x0, y0, x1, y1 = box or (0, 0, img_u8.shape[1], img_u8.shape[0])
    size = (int((x1 - x0) * scale), int((y1 - y0) * scale))
    rgb = cv2.cvtColor(cv2.resize(img_u8[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)
    for labels, color in zip(label_maps, colors):
        crop = labels[y0:y1, x0:x1]
        small = cv2.resize(crop.astype(np.int32).astype(np.float32), size, interpolation=cv2.INTER_NEAREST).astype(np.int32)
        for k in np.unique(small):
            if k == 0:
                continue
            contours, _ = cv2.findContours((small == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, contours, -1, color, thickness)
    return rgb


def bounding_box(label_maps: list[np.ndarray], margin: int = 64, shape: tuple[int, int] = (2048, 2048)):
    """Smallest box (x0, y0, x1, y1) containing all labelled pixels, padded by ``margin``."""
    ys, xs = np.nonzero(np.any([lm > 0 for lm in label_maps], axis=0))
    if len(xs) == 0:
        return 0, 0, shape[1], shape[0]
    return (max(xs.min() - margin, 0), max(ys.min() - margin, 0),
            min(xs.max() + margin, shape[1]), min(ys.max() + margin, shape[0]))
