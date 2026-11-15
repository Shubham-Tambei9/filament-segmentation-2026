"""Solar filament instance segmentation (Kaggle filament-segmentation-2026)."""
from .paths import configure_caches

# Point torch / Hugging Face download caches into the project before any submodule imports
# segmentation_models_pytorch (huggingface_hub reads HF_HOME once, at import time).
configure_caches()
