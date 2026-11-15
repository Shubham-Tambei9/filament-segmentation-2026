"""Make sure a weight file is present before it is loaded (torch-free, standard library only).

The trained weights are stored in the repository under ``weights/``.  GitHub refuses single files larger
than 100 MB, so the few larger checkpoints are stored as numbered parts (``<name>.pt.part0``,
``<name>.pt.part1``, ...).  ``ensure_weight`` rebuilds the ``.pt`` file from its parts the first time
it is needed.  As a last resort a missing file is downloaded from the repository's v1.0 GitHub release,
which holds the same files.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/Shubham-Tambei9/filament-segmentation-2026/releases/download/v1.0/"
PART_SIZE = 95 * 1024 * 1024  # below GitHub's 100 MB per-file limit


def _parts(path: Path) -> list[Path]:
    parts, k = [], 0
    while (p := path.with_name(f"{path.name}.part{k}")).exists():
        parts.append(p)
        k += 1
    return parts


def ensure_weight(path, download: bool = True) -> Path:
    """Return ``path`` after making sure it exists: as is, joined from its parts, or downloaded."""
    path = Path(path)
    if path.exists():
        return path
    tmp = path.with_name(path.name + ".tmp")
    parts = _parts(path)
    if parts:
        with open(tmp, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out)
        os.replace(tmp, path)
        print(f"joined {path.name} from {len(parts)} parts")
    elif download and path.suffix == ".pt":
        print(f"downloading {path.name} from the v1.0 release ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RELEASE_URL + path.name, tmp)
        os.replace(tmp, path)
    return path


def split_weight(path, part_size: int = PART_SIZE) -> list[Path]:
    """Write ``<name>.part0``, ``<name>.part1``, ... for a file larger than ``part_size`` (used once, to store it)."""
    path = Path(path)
    out = []
    with open(path, "rb") as f:
        while chunk := f.read(part_size):
            p = path.with_name(f"{path.name}.part{len(out)}")
            p.write_bytes(chunk)
            out.append(p)
    return out
