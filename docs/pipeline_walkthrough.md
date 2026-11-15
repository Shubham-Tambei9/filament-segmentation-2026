# Pipeline walkthrough

A plain-language guide to every component of the solution: what it does, why it is there, where it
lives in the code and how much it is worth. Numbers are out-of-fold (OOF) Panoptic Quality on the 707
training images unless stated otherwise. The submitted pipeline is called **H** in the logs and configs
(`weights/final_config.json`).

## 1. The task and the metric

- **Input:** a 2048 x 2048 grayscale H-alpha image of the full solar disk (GONG network, 6 sites).
- **Output:** one binary mask per filament (a dark, thread-like structure on the disk).
- **Metric, Panoptic Quality (PQ):** a predicted filament matches a reference filament when their
  IoU is above 0.5. Then PQ = (sum of IoUs of matches) / (matches + 0.5 false positives + 0.5 false
  negatives). It equals SQ x RQ: SQ = average IoU of matches (how well drawn), RQ = an F1 score
  (how many found, how few false alarms).
- **Several annotators per image:** each annotator's set is scored separately and everything is pooled.
  So the best prediction is what a *typical* annotator draws.
- **Human level:** one annotator scored against another gets PQ 0.340. Our final pipeline gets **0.443**
  (public leaderboard 0.39).

Code: `filament_seg/metrics.py` (PQ), `report/official_eval.py` (organisers' definitions, gives the same PQ).

## 2. Turning annotations into a training target

- Each filament polygon is rasterised with pycocotools (the same library the organisers use), because
  OpenCV draws polygons about 11% thicker.
- Instead of picking one annotator, the target pixel value is **the fraction of annotators that marked
  it** (0, 1/3, 1/2, 2/3, 1). The network learns "how likely is an annotator to mark this pixel".

Code: `filament_seg/annotations.py` (`Filament.mask_crop`, `consensus_mask`).

## 3. The segmentation networks

- A **U-Net**: an encoder pretrained on ImageNet compresses the image, a decoder rebuilds a
  full-resolution probability map; skip connections keep the thin details.
- Inputs are the whole disk resized to **1024** or **1536** pixels (filaments are only ~6 px wide at 1024).
- **Loss:** binary cross-entropy (works with soft targets) + soft Dice (filaments are only 0.3% of pixels).
- **Augmentation:** any rotation around the disk centre, flips, +-10% scale, brightness/contrast/gamma,
  blur and noise.
- **5-fold cross-validation**, split by image and stratified by site, so every training image gets an
  honest prediction from a model that never saw it (that is what "out of fold" means).
- **Test-time augmentation (TTA):** each model predicts the 8 flipped/rotated versions of the image and
  averages them (+0.002 PQ over 4 flips).
- **Final ensemble (20 models):** four encoder groups, 5 folds each: EfficientNetV2-S (1024 px),
  ConvNeXt-T (1024 px), ConvNeXt-T (1536 px), ConvNeXt-S (1024 px). Group weights 1 : 1 : 4 : 0.5, chosen
  by a grid search on OOF PQ. All encoder weights are Apache-2.0 / BSD (no non-commercial models).

How we got there (OOF PQ, rules + refiner where noted):

| Models | OOF PQ |
|---|---|
| ResNet-34, 5 folds x 1024 px, rules | 0.418 |
| ResNet-34 + UNet++ at 1024/1536 (20 models), refiner | 0.435 |
| + EfficientNetV2-S + ConvNeXt-T (30 models), refiner | 0.438 |
| EfficientNetV2-S + ConvNeXt-T only (10 models), refiner (fast mode) | 0.442 |
| + ConvNeXt-T 1536 + ConvNeXt-S (20 models), refiner (**final**) | **0.443** |

Lesson: modern encoders beat the older ResNet-34/UNet++ ensemble, and more models are not automatically
better: the 10-model modern ensemble beats the 30-model mixed one.

Code: `filament_seg/model.py`, `dataset.py`, `train.py`, `inference.py`, `blend.py`.

## 4. From probabilities to filament instances (the "rules")

1. Threshold the averaged probability map (0.5).
2. Pieces closer than 16 px are treated as one filament (they are grouped, but each keeps only its own
   pixels, so masks do not get thicker). This fixes filaments split into fragments.
3. Remove instances smaller than 300 px or with mean probability below 0.85 (removes weak false alarms);
   keep the best one if an image would otherwise be empty.

The four numbers are chosen by grid search on the OOF predictions only (`weights/postprocess_blendH_FR.json`).
Effect: PQ 0.397 (plain threshold 0.5) -> 0.439; fragmented filaments 754 -> 319.

Code: `filament_seg/postprocess.py`, `tune.py`.

## 5. Instance refiner (second stage)

- For each detected filament, cut a crop rotated so the filament is horizontal (384 x 128 native
  pixels, no up-sampling).
- A small U-Net (ResNet-18) sees 4 channels: image, ensemble probability, this filament's mask, the other
  filaments' masks, and redraws the filament's extent as annotators would.
- It only changes the *shape*; whether a filament exists is still decided by the rules.
- Trained cross-fitted (the refiner used on fold k never saw fold k). Effect: 0.439 -> **0.443**
  (SQ 0.679, RQ 0.653).
- Tried and rejected: looser rules feeding the refiner, and training it longer (20 instead of 12 epochs);
  neither helped out of fold.

Code: `filament_seg/refine.py`, `refine_data.py`, `refine_train.py`.

## 6. Analysis that did not make the final pipeline

- **PQ-optimal decision layer:** because every prediction adds exactly 1/2 to the PQ denominator, the
  best rule is "keep a candidate if its expected matched IoU exceeds PQ/2". We built candidates at 17
  thresholds, scored them with gradient-boosted trees and picked the best non-overlapping set by dynamic
  programming. It tied the simple rules, so we kept the rules. Code: `decision.py`.

## 7. Validation and honesty

- Model selection uses OOF PQ only; the public leaderboard (about 90 test images) is a sanity check.
- Our instances never overlap and are never duplicated, so the score does not depend on the known
  scorer weakness that rewards duplicate masks.
- Only the official training data is used; ideas from public notebooks are cited in the report.
- The executed notebook `notebooks/filament_pipeline.ipynb` recomputes every number from the weights and
  reproduces the submitted file (same 1,131 filaments; differing masks agree at IoU 0.999).

## 8. Likely judge questions

- *Why soft targets?* Annotators disagree; the fraction of annotators is the expected label, which is
  what pooled PQ rewards.
- *Why not one-to-one assignment in the scorer?* PQ's IoU > 0.5 rule guarantees unique matches when
  predictions do not overlap (Kirillov et al. 2019); ours do not overlap.
- *Why is the public score (0.39) below OOF (0.443)?* Only ~90 public images (noise +-0.02) and possible
  differences between the test and training distributions; we never tuned on the public score.
- *Why these four encoder groups and weights?* Each group was added only if it raised OOF PQ in a joint
  grid search of the group weights; the 1536-px ConvNeXt-T gives the largest gain (finer detail).
- *What limits the score?* Label ambiguity: the remaining errors are where annotators disagree most
  (small filaments: recall 0.39 below 600 px; about a third of misses are near misses with IoU 0.25-0.5).
- *Speed?* 8.6 s per 2048 x 2048 image on an RTX 3070 Ti laptop GPU with 8-view TTA (20 models); the
  10-model fast mode takes 3.7 s for PQ 0.442. Without TTA every row is about five times faster.
