# Solar Filament Instance Segmentation (Kaggle filament-segmentation-2026)

Solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(Kaggle, organised with the U.S. NSF National Solar Observatory; Kaggle: shubhami9).
Instance segmentation of solar filaments in full-disk GONG H-alpha images (2048 x 2048), scored by
Panoptic Quality (PQ). Ground truth comes from the MAGFiLO dataset (several annotators per image).

**Result:** out-of-fold PQ **0.443** on the 707 training images, scored against every annotator (95% bootstrap
interval 0.434-0.452; per fold 0.443 ± 0.011; two human annotators agree with each other at 0.340); public
leaderboard **0.39**. A plain-language explanation of every
component is in [docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md); the reasoning and evidence behind
each choice is in [docs/design_decisions.md](docs/design_decisions.md); the 4-page report is
[report/Report_Shubham_Tambe.pdf](report/Report_Shubham_Tambe.pdf), written in the organisers' Overleaf template
(LaTeX source in [report/](report/); `python report/build_overleaf_zip.py` packs it for Overleaf).

https://github.com/user-attachments/assets/a2a12683-7489-4111-8845-55f9acb9af9c

*72-second video overview (full-quality MP4 on the [v1.0 release](https://github.com/Shubham-Tambei9/filament-segmentation-2026/releases/download/v1.0/filament-segmentation-product.mp4)). Music: Sascha Ende, ende.app (CC BY 4.0).*

## Approach

```
H-alpha JPEG (2048²) ─resize─▶ 1024² / 1536² ─▶ 20 U-Nets (4 encoder groups x 5 folds, 8-view TTA) ─▶ probability map
   ─▶ tuned rules: threshold, group fragments, drop small / low-confidence instances
   ─▶ instance refiner: each filament redrawn in its own rotated native-resolution crop ─▶ RLE instances
```

1. **Agreement-aware targets.** Annotators disagree a lot (annotator vs. annotator: PQ 0.340). The networks
   are trained on the per-pixel *fraction of annotators* (0, 1/3, 1/2, 2/3, 1) with BCE + soft Dice
   ([`annotations.py`](filament_seg/annotations.py)). Polygons are rasterised with pycocotools, like the
   organisers' reference masks (OpenCV's `fillPoly` is ~11% too thick).
2. **Ensemble of modern encoders.** U-Nets with ImageNet-pretrained timm encoders: EfficientNetV2-S and
   ConvNeXt-T at 1024 px, ConvNeXt-T at 1536 px and ConvNeXt-S at 1024 px, 5 folds each, 8-view test-time
   augmentation, group weights chosen out of fold ([`blend.py`](filament_seg/blend.py), configuration in
   `weights/final_config.json`). Only weights under permissive licences (Apache-2.0 / BSD) are used.
3. **Rules tuned for PQ** ([`postprocess.py`](filament_seg/postprocess.py), [`tune.py`](filament_seg/tune.py)):
   threshold, fragments grouped without thickening masks, small and low-confidence instances dropped;
   values in `weights/postprocess_blendH_FR.json`. PQ 0.397 (plain threshold) → 0.439.
4. **Instance refiner** ([`refine.py`](filament_seg/refine.py), [`refine_train.py`](filament_seg/refine_train.py)):
   a small U-Net redraws each detected filament's extent from a rotated 384 x 128 crop (image, probability,
   own mask, neighbours' masks). Cross-fitted gain 0.439 → **0.443**.
5. **Validation that matches the task.** Folds split by physical image and stratified by GONG site; each
   prediction is scored against *every* annotator and the statistics pooled
   ([`evaluation.py`](filament_seg/evaluation.py)). [`report/official_eval.py`](report/official_eval.py)
   reproduces the organisers' self-evaluation analysis and gives the same PQ.
6. **Analysis kept out of the final pipeline:** a PQ-optimal decision layer (component tree + learned scorer +
   exact selection, [`decision.py`](filament_seg/decision.py)) ties the tuned rules, so the simpler rules are
   used. Ideas for the decision layer and the refiner follow the public notebook *Deciding under disagreement*
   (Kaggle user tushhaaarrrr); the implementations are our own.

## Results (out-of-fold, 707 images, scored against every annotator)

| Pipeline | Models | s / image* | PQ | Public LB |
|---|---|---|---|---|
| Annotator vs. annotator (human reference) | – | – | 0.340 | – |
| ResNet-34 U-Net, 5 folds x 1024 px, rules | 5 | 1.1 | 0.418 | 0.36 |
| ResNet-34 + UNet++ (2 sizes), refiner | 20 | 10.5 | 0.435 | 0.38 |
| + EfficientNetV2-S + ConvNeXt-T | 30 | 13.6 | 0.438 | 0.38 |
| EfficientNetV2-S + ConvNeXt-T, refiner (`MODE = "fast"`) | 10 | 3.7 | 0.442 | 0.38 |
| **+ ConvNeXt-T 1536 + ConvNeXt-S, refiner (final, `MODE = "full"`)** | **20** | **8.6** | **0.443** | **0.39** |

\* end-to-end GPU time per 2048 x 2048 image on an RTX 3070 Ti Laptop GPU (8-view TTA, rules and refiner
included). Organisers' analysis of the final out-of-fold predictions: mean IoU 0.579 (annotators: 0.517),
mean Dice 0.711 (0.660); 3.9% of reference filaments split (annotators: 2.9%), 3.4% of predictions merged (3.4%).

## Setup

Python 3.12, NVIDIA GPU (developed on an RTX 3070 Ti Laptop, 8 GB). The pinned versions are in
[`requirements.txt`](requirements.txt).

Training was shared between the laptop GPU and free Kaggle notebook GPUs (Tesla T4): EfficientNetV2-S
folds 0-4 and ConvNeXt-T 1536 / ConvNeXt-S folds 0-3 on Kaggle; ConvNeXt-T 1024 folds 0-4, ConvNeXt-T 1536 /
ConvNeXt-S fold 4 and the refiners on the laptop (one early UNet++ fold, not in the final pipeline, on Google
Colab). T4 / P100 GPUs only emulate bf16, so the Kaggle runs trained with `--amp fp16` (the default, bf16, is for
GPUs of compute capability 8.0 or newer); inference and refiner training choose the precision automatically
(`filament_seg.inference.amp_dtype`). All inference timings are measured on the laptop GPU.

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Place the competition data at `filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/` (or set
`FILAMENT_DATA_ROOT`):

```
MAGFiLO_1.0_Kaggle_2026/
├── train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
├── train/train_images/*.jpeg
└── test/test_images/*.jpeg
```

The trained weights of the submitted pipelines (20 U-Nets and 2 refiners, 1.4 GB) are stored in `weights/`, so
nothing has to be downloaded or retrained. The five ConvNeXt-S checkpoints are larger than GitHub's 100 MB file
limit and are stored as two parts each (`*.pt.part0`, `*.pt.part1`); they are joined automatically the first
time they are loaded ([`weights_store.py`](filament_seg/weights_store.py)). The same 22 files are also on the
[v1.0 release](https://github.com/Shubham-Tambei9/filament-segmentation-2026/releases/tag/v1.0).

## Usage

The notebook [`notebooks/filament_pipeline.ipynb`](notebooks/filament_pipeline.ipynb) runs everything end to
end and is saved with its outputs: `MODE = "full"` is the submitted pipeline, `MODE = "fast"` the 10-model
version; `TRAIN = True` retrains every model, `RUN_TUNING = True` re-fits rules, decision layer and refiner.
It also runs unchanged as a Kaggle notebook (its first cell sets up Kaggle's input folders).
The individual steps:

```bash
# train one cross-validation fold (outputs in work/runs/<name>/)
python -m filament_seg.train --fold 0 --size 1024 --encoder tu-convnext_tiny --batch-size 2 --lr 2.5e-4 --epochs 40

# tune the rules on out-of-fold probability maps; train the refiner (cross-fitted)
python -m filament_seg.tune --prob-dirs work/oof/<blend> --out weights/postprocess.json
python -m filament_seg.refine_data --prob-dir work/oof/<blend> --out work/refine/crops
python -m filament_seg.refine_train --crops work/refine/crops --prob-dir work/oof/<blend> --out weights/refiner.pt

# write a submission (fold ensemble, 8-view TTA, rules, refiner)
python -m filament_seg.predict --weights weights/tu-convnext_tiny_s1024_f*.pt --params weights/postprocess.json \
    --refiner weights/refiner.pt --out submission.csv

# tests
python -m pytest
```

## Repository layout

| Path | Purpose |
|---|---|
| `filament_seg/annotations.py` | COCO JSON loading, pycocotools rasterisation, consensus targets, folds |
| `filament_seg/dataset.py` | resized-image cache, augmentations, PyTorch dataset |
| `filament_seg/model.py` | U-Net / UNet++ builder, fp16 checkpoint save/load |
| `filament_seg/inference.py` | full-disk inference with 4- or 8-view TTA, out-of-fold prediction cache |
| `filament_seg/blend.py` | weighted blending of probability maps, pipeline configuration |
| `filament_seg/postprocess.py` | rules: probability map → filament instances |
| `filament_seg/refine.py`, `refine_data.py`, `refine_train.py` | instance refiner (crops, data, training) |
| `filament_seg/decision.py`, `decision_train.py` | PQ-optimal decision layer (analysis) |
| `filament_seg/metrics.py` | PQ / SQ / RQ, Dice, fragmentation and over-merging counts |
| `filament_seg/evaluation.py` | multi-annotator validation (parallel, torch-free) |
| `filament_seg/train.py`, `tune.py`, `predict.py` | command-line entry points |
| `notebooks/filament_pipeline.ipynb` | full pipeline with explanations, validation report and timings |
| `weights/*.json`, `decision.pkl` | pipeline configurations, tuned rules, decision layer |
| `report/` | 4-page report (LaTeX), figure / number generators, organisers' analysis |
| `docs/` | pipeline walkthrough and design decisions |
| `tests/` | unit tests (metric, post-processing, RLE, rasterisation, decision layer, refiner, blending) |

## License

MIT (code). Model weights derive from ImageNet-pretrained encoders under Apache-2.0 / BSD licences and were
trained on the competition data, which may be used for non-commercial purposes only (competition, academic
research and education); the released weights are therefore provided for non-commercial, academic use.
The competition data itself is not included.
