# Design decisions log

Running record of what was decided, why, and what evidence supported it. Feeds the 4-page report.

## D1. Data facts that drive the design (EDA, 2026-11-09)

- 707 physical training images, 1154 annotation sets, 8199 filaments; 180 test images, no overlap.
  Same six GONG sites and 2011–2022 year mix in train and test; global intensity statistics match
  (mean 83.7 vs 82.6 / 255), so no domain-adaptation step is needed.
- All images 2048 x 2048, 8-bit grayscale, disk centred near (1024, 1024) with radius 915–1035 px.
- Filaments are small and thin: median area 1228 px, median bbox long side 96 px (p95 315, max 1067).
- Each annotation is a single polygon; filaments of one annotator do not overlap (2 px in 150 sampled
  images). Category "Ambiguous" has no instances; categories are not part of the submission.
- **Inter-annotator PQ = 0.340 (SQ 0.634, RQ 0.536)** over all annotator pairs on the 296 images with
  multiple annotators (pycocotools masks; a first estimate of 0.361 used OpenCV masks, which are too
  thick and inflate IoU — see D8). Disagreements: (a) boundary thickness, (b) which faint filaments are
  included, (c) whether a long filament is one instance or several pieces.

## D2. Train on the per-pixel annotator agreement (soft consensus)

Options: pick one annotator per image / treat each annotation set as a separate sample / train on the
agreement fraction. Separate samples give the network contradictory hard targets for the same input;
picking one throws away information. The agreement fraction is the expected label under a random
annotator, which is exactly what a per-annotator PQ evaluation rewards. Loss: BCE (works with soft
targets) + batch soft-Dice (counters the ~99.7% background imbalance).

## D3. Semantic U-Net + connected components instead of an instance network (Mask R-CNN etc.)

Instances of one annotator never overlap and are spatially separated, so instance identity is mostly
a matter of connectivity. A semantic network is simpler, faster and lets the instance-forming rules
(threshold, fragment grouping, size filter) be tuned directly against PQ. Fragmentation vs.
over-merging is handled by `merge_gap`: blobs separated by at most this many pixels are one instance
(computed with a Euclidean distance transform, so the cost is independent of the gap), but masks keep
their original pixels (grouping does not fatten masks, which would hurt SQ).

## D4. Resolution: full disk at 1024 x 1024 (baseline)

Median filament width ~13 px at native resolution → ~6 px at 1024, still resolvable. 1024 keeps the
whole disk in one forward pass (global context, no tiling seams) and trains at ~14 img/s on an 8 GB
laptop GPU (ResNet-34 U-Net, bf16, batch 4, peak 3.9 GB). Full-resolution crops are a planned
experiment for boundary accuracy (SQ).

## D5. Validation protocol

5 folds split by physical image (`file_name`), never by annotation set, stratified by GONG site.
Each prediction is scored against every annotator of the image and statistics are pooled (the hidden
test labels are assumed to come from one annotator per image). Post-processing parameters are tuned
only on held-out folds.

## D6. Augmentations

Rotation by any angle about the disk centre + flips (no preferred orientation for this task), ±10%
scale (disk radius varies ~915–1035 px between sites/seasons), small shifts, brightness / contrast /
gamma, occasional blur (seeing) and noise. Implemented with NumPy/OpenCV only (no extra dependency).

## D8. Rasterise polygons with pycocotools, not OpenCV

The JSON `area` field is computed by pycocotools (polygon → RLE). Measured on 400 random filaments:
pycocotools masks reproduce `area` exactly (ratio 1.0000), while OpenCV `fillPoly` masks are 10.6%
larger (median; p90 18.8%) because they include boundary pixels — IoU with the reference mask only
0.88 (p10 0.80). Training on OpenCV masks would teach the network systematically over-thick filaments
and cap SQ near 0.88 even for perfect shapes; validating on them would mis-measure PQ. A first training
run was stopped after one epoch and restarted when this was found.
For speed each filament is rasterised on its own bounding-box canvas (8x faster than full 2048²
decodes); this matches the JSON area for 99.1% of filaments (117 px total difference over 8199).
The data also contains two within-annotator overlaps (a near-duplicate filament drawn twice, a 60 px
filament inside a larger one); in label maps the later polygon wins, as in any per-pixel GT.

## D9. PQ-optimal decision layer (2026-11-10)

Pooled PQ = sum of matched IoUs / (TP + FP/2 + FN/2), and per annotator record TP + FP/2 + FN/2 =
(|P| + |G|)/2, so each predicted instance adds exactly 1/2 to the denominator whether or not it matches.
Keeping a candidate therefore raises PQ iff its annotator-expected matched IoU u exceeds PQ*/2
(Dinkelbach). Implementation (`filament_seg/decision.py`, `decision_train.py`):
candidates = connected components at 17 thresholds 0.10-0.90 (fragments within 16 px grouped), which nest
into a forest; a cross-fitted gradient-boosting "hurdle" scorer (P(match) x E[IoU | match], scikit-learn)
predicts u from 24 features; the exact best antichain (tree cut) for sum(u - tau) comes from dynamic
programming. The bank stores per-candidate summed matched IoU / match counts, so the exact pooled PQ of any
selection is a sum (verified identical to the pixel-level evaluator).
Evidence (folds 0+1, 284 images, 2-fold cross-fit): first version 0.406 (overfit: rank correlation 0.85
in-sample vs 0.63 out-of-sample); strong regularisation (150 iterations, 7 leaves, min leaf 300) 0.413;
+ ring-contrast / stability features 0.417; + 16-px fragment grouping 0.424 vs tuned fixed rules 0.422.
Oracle (true u) 0.544 shows the candidate tree contains much better selections than any scorer finds.
Scorer variants (direct regression, monotone constraints, prefilters) were all within +-0.002.
Idea credit: public notebook "Deciding under disagreement" (Kaggle user tushhaaarrrr); independent code.

## D10. 8-view test-time augmentation

All eight symmetries of the square (flips x transpose) instead of four flips: fold-0 PQ 0.4190 -> 0.4212
(fewer false positives) for 2x the network evaluations (~0.25 s per image on the laptop GPU).

## D11. Engineering: memory on Windows

Running CPU-only workers (bank building, tuning) next to GPU training exhausted the commit limit (the page
file lives on the nearly full C: drive) and crashed one training run with MemoryError: every worker imported
PyTorch (~2.4 GB committed each). CPU-only modules now import a torch-free `imaging.py`; worker processes
commit a few hundred MB.

## D7. Dependencies and licences

PyTorch (BSD), segmentation_models_pytorch (MIT), timm (Apache-2.0), pycocotools (BSD), OpenCV
(Apache-2.0), NumPy/SciPy/scikit-image/scikit-learn (BSD). Avoided AGPL tooling (e.g. Ultralytics) because the
rules require an OSI licence that does not restrict commercial use.
