"""Full-disk inference: probability map at the original 2048 x 2048 resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .dataset import normalize
from .imaging import read_gray, resize, upsample_prob  # noqa: F401  (upsample_prob re-exported)

# test-time augmentation views: (transpose, flip H, flip V), each undone on the output.
# "flip4" = the four flips; "d4" = all eight symmetries of the square (flips x transpose).
TTA_VIEWS = {
    "none": [(False, False, False)],
    "flip4": [(False, fh, fv) for fh in (False, True) for fv in (False, True)],
    "d4": [(tr, fh, fv) for tr in (False, True) for fh in (False, True) for fv in (False, True)],
}


def _apply(x: torch.Tensor, transpose: bool, fh: bool, fv: bool) -> torch.Tensor:
    if transpose:
        x = x.transpose(2, 3)
    dims = [d for d, f in ((3, fh), (2, fv)) if f]
    return torch.flip(x, dims) if dims else x


def _undo(y: torch.Tensor, transpose: bool, fh: bool, fv: bool) -> torch.Tensor:
    dims = [d for d, f in ((3, fh), (2, fv)) if f]
    y = torch.flip(y, dims) if dims else y
    return y.transpose(2, 3) if transpose else y


def amp_dtype(device) -> torch.dtype:
    """Autocast dtype: bfloat16 only on GPUs with native support (compute capability >= 8.0), else float16.

    ``torch.cuda.is_bf16_supported()`` is also true for *emulated* bfloat16 on older GPUs (e.g. Kaggle's T4 and
    P100), where cuDNN has no bfloat16 kernel for some convolutions (ConvNeXt's 7x7 depthwise convolution fails
    with "unable to find an engine").
    """
    device = torch.device(device)
    if device.type != "cuda":
        return torch.float16
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


@torch.no_grad()
def predict_prob(model: torch.nn.Module, img_u8: np.ndarray, size: int, tta, device) -> np.ndarray:
    """Return the averaged sigmoid probability at the network resolution ``size`` (float32).

    ``tta`` is one of ``TTA_VIEWS`` ("none", "flip4", "d4"); ``True`` / ``False`` mean "flip4" / "none".
    """
    mode = {True: "flip4", False: "none"}.get(tta, tta)
    views = TTA_VIEWS[mode]
    x = torch.from_numpy(normalize(resize(img_u8, size)))[None, None].to(device)
    acc = torch.zeros((size, size), device=device)
    dev_type = torch.device(device).type
    with torch.autocast(device_type=dev_type, dtype=amp_dtype(device), enabled=dev_type == "cuda"):
        for view in views:
            out = model(_apply(x, *view)).float()
            acc += torch.sigmoid(_undo(out, *view))[0, 0]
    return (acc / len(views)).cpu().numpy()


def predict_to_dir(
    checkpoints: list[tuple[torch.nn.Module, dict]],
    stems: Iterable[str],
    image_dir: Path,
    out_dir: Path,
    tta,
    device,
) -> None:
    """Save the (ensemble-averaged) probability map of each image as float16 ``<stem>.npy``.

    Maps are stored at the largest network resolution among the models (not 2048) to keep the
    cache small; :func:`upsample_prob` restores full resolution when needed.  Existing files are
    kept, so an interrupted run resumes where it stopped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = max(meta["size"] for _, meta in checkpoints)
    for stem in tqdm(list(stems), desc=f"predict -> {out_dir.name}"):
        path = out_dir / f"{stem}.npy"
        if path.exists():
            continue
        img = read_gray(Path(image_dir) / f"{stem}.jpeg")
        probs = []
        for model, meta in checkpoints:
            p = predict_prob(model, img, meta["size"], tta, device)
            probs.append(p if p.shape[0] == size else cv2.resize(p, (size, size), interpolation=cv2.INTER_LINEAR))
        np.save(path, np.mean(probs, axis=0).astype(np.float16))
