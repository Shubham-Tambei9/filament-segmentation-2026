"""Project paths and cache locations.

All caches (pretrained encoder weights, temp files) are redirected into ``work/.cache`` inside
the project so nothing large is written to the system drive.  ``FILAMENT_DATA_ROOT`` can
override where the competition data lives.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "FILAMENT_DATA_ROOT",
        PROJECT_ROOT / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026",
    )
)
TRAIN_JSON = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMAGES = DATA_ROOT / "train" / "train_images"
TEST_IMAGES = DATA_ROOT / "test" / "test_images"

WORK_DIR = PROJECT_ROOT / "work"
CACHE_DIR = WORK_DIR / ".cache"
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def configure_caches() -> None:
    """Point torch / HuggingFace download caches at the project folder (call before model creation)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
