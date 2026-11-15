"""Pipeline figure for the report (report/figures/pipeline.pdf).

Top row: inference on one training image, shown with its out-of-fold probability map and the final
out-of-fold instances (rules + the cross-fitted refiner of its fold), so every picture is real output.
Bottom row: how the parts are trained and fitted.

    python report/make_pipeline_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from filament_seg.annotations import load_annotations, make_folds  # noqa: E402
from filament_seg.imaging import read_gray, upsample_prob  # noqa: E402
from filament_seg.paths import TRAIN_IMAGES, TRAIN_JSON  # noqa: E402
from filament_seg.postprocess import PostprocessParams, probs_to_labels  # noqa: E402
from filament_seg.refine import paste_back  # noqa: E402
from filament_seg.refine_train import build_refiner, refine_crops  # noqa: E402
from filament_seg.visualize import outlines  # noqa: E402

PROB_DIR = ROOT / "work" / "oof" / "blendH_FR"
FOLD_DIR = ROOT / "work" / "refine" / "folds_blendH_FR"
PARAMS = ROOT / "weights" / "postprocess_blendH_FR.json"
MAG, INK, MUTE, LINE = "#B8006F", "#1A1A1A", "#555555", "#9A9A9A"
FILL_INF, FILL_TRAIN = "#F3E6EE", "#EAF1F7"


def example():
    """A three-annotator image with a typical number of filaments (median of those with 8-16)."""
    items = load_annotations(TRAIN_JSON)
    folds = make_folds(items, 5, 42)
    cand = [(it, f) for it, f in zip(items, folds) if it.n_annotators == 3
            and 8 <= np.median([len(a) if hasattr(a, "__len__") else 0 for a in it.annotations.values()]) <= 16]
    if not cand:
        cand = [(it, f) for it, f in zip(items, folds) if it.n_annotators == 3]
    cand.sort(key=lambda c: c[0].stem)
    return cand[len(cand) // 2]


def thumbs(item, fold):
    img = read_gray(TRAIN_IMAGES / item.file_name)
    prob = upsample_prob(np.load(PROB_DIR / f"{item.stem}.npy"))
    params = PostprocessParams(**json.loads(PARAMS.read_text()))
    labels = probs_to_labels(prob, params)
    ck = torch.load(FOLD_DIR / f"fold{fold}.pt", map_location="cpu", weights_only=False)
    model = build_refiner(pretrained=False)
    model.load_state_dict({n: w.float() for n, w in ck["state_dict"].items()})
    model.eval()
    geoms, crops = refine_crops(model, img, prob, labels, torch.device("cpu"))
    final = paste_back(labels.shape, geoms, crops, 0.5, labels) if geoms else labels
    size = 300
    t_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    # thin filaments vanish when averaged down to a thumbnail: dilate first (display only)
    t_prob = cv2.resize(cv2.dilate(prob.astype(np.float32), np.ones((9, 9), np.uint8)), (size, size),
                        interpolation=cv2.INTER_AREA)
    t_out = outlines(img, [final], colors=[(214, 0, 140)], scale=size / img.shape[0], thickness=2)
    return t_img, t_prob, t_out, int(final.max())


def box(ax, x, y, w, h, title, lines, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fill, ec=LINE, lw=0.6))
    ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=7, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.25, "\n".join(lines), ha="center", va="top", fontsize=6, color=MUTE,
            linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, label=None, dashed=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                 color=MAG if not dashed else MUTE, linestyle="--" if dashed else "-"))
    if label:
        ax.text((x0 + x1) / 2 + 0.04, (y0 + y1) / 2, label, fontsize=5.5, color=MUTE, ha="left", va="center",
                style="italic")


def thumb(fig, x, y, s, im, caption, cmap=None):
    a = fig.add_axes([x / W, y / H, s / W, s / H])
    a.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap == "magma" else None)
    a.set_xticks([]), a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(LINE), sp.set_linewidth(0.6)
    fig.text((x + s / 2) / W, (y - 0.04) / H, caption, ha="center", va="top", fontsize=6, color=INK)


W, H = 7.0, 2.35


def main():
    item, fold = example()
    t_img, t_prob, t_out, n_inst = thumbs(item, fold)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")

    s, yt = 0.86, 1.38  # thumbnail size, top-row bottom
    yb, hb = 1.30, 1.00  # inference boxes
    thumb(fig, 0.02, yt, s, t_img, "H$\\alpha$ image, 2048$^2$", cmap="gray")
    box(ax, 1.03, yb, 1.33, hb, "U-Net ensemble", ["resize to 1024$^2$ / 1536$^2$",
        "EffNetV2-S, ConvNeXt-T,", "ConvNeXt-T 1536, ConvNeXt-S", "5 folds each, 8-view TTA,",
        "group weights 1 : 1 : 4 : 0.5"], FILL_INF)
    thumb(fig, 2.53, yt, s, t_prob, "probability map", cmap="magma")
    box(ax, 3.55, yb, 1.13, hb, "Instance rules", ["threshold 0.5", "join pieces $\\leq$16 px",
        "drop < 300 px or", "mean probability < 0.85"], FILL_INF)
    box(ax, 4.85, yb, 1.13, hb, "Instance refiner", ["per filament: rotated", "384$\\times$128 native crop,",
        "ResNet-18 U-Net", "redraws its extent"], FILL_INF)
    thumb(fig, 6.12, yt, s, t_out, f"{n_inst} filaments $\\rightarrow$\none RLE row each")
    ym = yb + hb / 2
    for x0, x1 in [(0.02 + s, 1.03), (2.36, 2.53), (2.53 + s, 3.55), (4.68, 4.85), (5.98, 6.12)]:
        arrow(ax, x0 + 0.02, ym, x1 - 0.02, ym)

    yb2, hb2 = 0.05, 0.78  # training boxes
    box(ax, 0.02, yb2, 1.62, hb2, "Annotations", ["MAGFiLO polygons, 1-3", "annotators per image,",
        "rasterised with pycocotools"], FILL_TRAIN)
    box(ax, 1.86, yb2, 1.62, hb2, "Soft training target", ["fraction of annotators",
        "marking each pixel;", "BCE + soft Dice, AdamW"], FILL_TRAIN)
    box(ax, 3.70, yb2, 3.28, hb2, "5-fold cross-validation",
        ["folds split by physical image, stratified by GONG site;",
         "out-of-fold predictions scored against every annotator (pooled PQ);",
         "group weights and rules by grid search on out-of-fold PQ; refiner cross-fitted"], FILL_TRAIN)
    arrow(ax, 1.64, yb2 + hb2 / 2, 1.86, yb2 + hb2 / 2, dashed=True)
    arrow(ax, 3.48, yb2 + hb2 / 2, 3.70, yb2 + hb2 / 2, dashed=True)
    arrow(ax, 2.67, yb2 + hb2, 1.70, yb - 0.01, "train", dashed=True)
    arrow(ax, 4.40, yb2 + hb2, 4.12, yb - 0.01, "fit", dashed=True)
    arrow(ax, 5.45, yb2 + hb2, 5.42, yb - 0.01, "fit", dashed=True)
    out = ROOT / "report" / "figures"
    fig.savefig(out / "pipeline.pdf")
    fig.savefig(ROOT / "work" / "tmp" / "pipeline_preview.png", dpi=220)
    print("example", item.stem, "fold", fold, "instances", n_inst)


if __name__ == "__main__":
    main()
