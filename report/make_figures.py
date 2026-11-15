"""Generate the report's figures (report/figures/*.pdf) and numbers (report/numbers.tex).

Run after the notebook has produced out-of-fold predictions:
    python report/make_figures.py --oof work/oof/resnet34_s1024 --params weights/postprocess.json

Every number quoted in main.tex comes from report/numbers.tex, so text and results cannot drift apart.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from filament_seg import paths  # noqa: E402
from filament_seg.annotations import label_map, load_annotations, make_folds  # noqa: E402
from filament_seg.evaluation import DecisionLabeler, evaluate_detailed, prob_loader  # noqa: E402
from filament_seg.imaging import read_gray  # noqa: E402
from filament_seg.metrics import PQAccumulator  # noqa: E402
from filament_seg.postprocess import PostprocessParams, probs_to_labels  # noqa: E402
from filament_seg.visualize import PALETTE, bounding_box, outlines  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oof", required=True, help="directory of out-of-fold probability maps")
    p.add_argument("--params", default=str(paths.WEIGHTS_DIR / "postprocess.json"))
    p.add_argument("--decision", default=str(paths.WEIGHTS_DIR / "decision.pkl"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--extra-json", default=None, help="optional JSON of additional named numbers")
    args = p.parse_args()
    out_fig = ROOT / "report" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    items = load_annotations(paths.TRAIN_JSON)
    fold_of = {it.stem: int(k) for it, k in zip(items, make_folds(items, 5, 42))}
    oof = Path(args.oof)
    oof_items = [it for it in items if (oof / f"{it.stem}.npy").exists()]
    rules = PostprocessParams(**json.loads(Path(args.params).read_text()))
    default = PostprocessParams(threshold=0.5, merge_gap=0, min_area=100, min_mean_prob=0.0, min_instances=0)
    decision = pickle.loads(Path(args.decision).read_bytes())
    labeler = DecisionLabeler(decision, paths.TRAIN_IMAGES, fold_of)  # cross-fitted fold scorers

    human = PQAccumulator()
    for it in items:
        if it.n_annotators > 1:
            for a, b in combinations([label_map(f) for f in it.annotations.values()], 2):
                human.update(a, b)
    h = human.summary()
    loader = prob_loader(oof)
    acc_default, _, _ = evaluate_detailed(loader, oof_items, default, workers=args.workers)
    # final method = tuned rules (figures, cv* numbers); decision layer reported for comparison (dec* numbers)
    acc_dec, _, _ = evaluate_detailed(loader, oof_items, labeler=labeler, workers=args.workers)
    acc, gt_df, pred_df = evaluate_detailed(loader, oof_items, rules, workers=args.workers)
    d, r, t = acc_default.summary(), acc_dec.summary(), acc.summary()

    # ---- figure 1: IoU / Dice / PQ distributions ----
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.9))
    ax[0].hist(acc.matched_ious, bins=np.linspace(0.5, 1, 21), color=BLUE)
    ax[0].set(xlabel="IoU of matched filaments", ylabel="count")
    ax[1].hist(acc.per_image_dice, bins=np.linspace(0, 1, 21), color=ORANGE)
    ax[1].set(xlabel="foreground Dice per image")
    ax[2].hist(acc.per_image_pq, bins=np.linspace(0, 1, 21), color=GREEN)
    ax[2].set(xlabel="PQ per image")
    fig.tight_layout()
    fig.savefig(out_fig / "distributions.pdf")

    # ---- figure 2: recall by filament size, precision by confidence ----
    bins = [0, 300, 600, 1000, 2000, 4000, 8000, np.inf]
    labels = ["<300", "300-600", "600-1k", "1k-2k", "2k-4k", "4k-8k", ">8k"]
    rec = gt_df.groupby(pd.cut(gt_df.area, bins, labels=labels), observed=False).matched.mean()
    conf_bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
    prec = pred_df.groupby(pd.cut(pred_df.mean_prob, conf_bins), observed=True).matched.mean()
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.8))
    ax[0].bar(range(len(rec)), rec.values, color=BLUE)
    ax[0].set_xticks(range(len(rec)), labels, rotation=60)
    ax[0].set(ylabel="recall", xlabel="GT area (px)", ylim=(0, 1))
    ax[1].bar(range(len(prec)), prec.values, color=ORANGE)
    ax[1].set_xticks(range(len(prec)), [f"{iv.left:.2f}" for iv in prec.index], rotation=60)
    ax[1].set(ylabel="precision", xlabel="mean probability", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(out_fig / "size_confidence.pdf")

    # ---- figure 3: qualitative example (3 annotators vs. model, median-quality image) ----
    img_recall = gt_df.groupby("image").matched.mean()
    three = [it for it in oof_items if it.n_annotators == 3 and it.stem in img_recall.index]
    three.sort(key=lambda it: img_recall[it.stem])
    ex = three[len(three) // 2]
    img = read_gray(paths.TRAIN_IMAGES / ex.file_name)
    ann_maps = [label_map(f) for f in ex.annotations.values()]
    pred = probs_to_labels(loader(ex.stem), rules)
    box = bounding_box(ann_maps + [pred], margin=48)
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.9))
    ax[0].imshow(outlines(img, ann_maps, colors=PALETTE[:3], scale=0.5, thickness=2, box=box))
    ax[0].set_title("three annotators", fontsize=7)
    ax[1].imshow(outlines(img, [pred], colors=[(214, 0, 140)], scale=0.5, thickness=2, box=box))
    ax[1].set_title("model (out-of-fold)", fontsize=7)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_fig / "qualitative.pdf")

    # ---- numbers ----
    missed = gt_df[~gt_df.matched]
    near = ((missed.best_iou >= 0.25) & (missed.best_iou <= 0.5)).mean()
    numbers = {
        "nImages": len(items), "nSets": sum(it.n_annotators for it in items),
        "nFilaments": sum(len(f) for it in items for f in it.annotations.values()),
        "nMulti": sum(it.n_annotators > 1 for it in items), "nOOF": len(oof_items),
        "humanPQ": h["PQ"], "humanSQ": h["SQ"], "humanRQ": h["RQ"], "humanDice": h["Dice_image_mean"],
        "defPQ": d["PQ"], "defSQ": d["SQ"], "defRQ": d["RQ"], "defFP": d["FP"], "defFN": d["FN"],
        "defOneToMany": d["one_to_many"], "defManyToOne": d["many_to_one"],
        "cvPQ": t["PQ"], "cvSQ": t["SQ"], "cvRQ": t["RQ"], "cvDice": t["Dice_image_mean"],
        "cvTP": t["TP"], "cvFP": t["FP"], "cvFN": t["FN"], "cvOneToMany": t["one_to_many"],
        "cvManyToOne": t["many_to_one"], "cvMeanIoU": float(np.mean(acc.matched_ious)),
        "nearMissShare": near, "ppThreshold": rules.threshold, "ppMergeGap": rules.merge_gap,
        "ppMinArea": rules.min_area, "ppMinMeanProb": rules.min_mean_prob,
        "decPQ": r["PQ"], "decSQ": r["SQ"], "decRQ": r["RQ"], "decDice": r["Dice_image_mean"],
        "decFP": r["FP"], "decOneToMany": r["one_to_many"], "decisionTau": decision["tau"],
        "recallSmall": float(gt_df[gt_df.area < 600].matched.mean()),
        "recallLarge": float(gt_df[gt_df.area >= 600].matched.mean()),
    }
    if args.extra_json:
        numbers.update(json.loads(Path(args.extra_json).read_text()))

    def fmt(v):
        if isinstance(v, str):
            return v
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}".replace(",", "{,}")
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"

    lines = ["% generated by report/make_figures.py -- do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{fmt(v)}}}" for k, v in numbers.items()]
    (ROOT / "report" / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: (float(v) if not isinstance(v, int) else v) for k, v in numbers.items()}, indent=1))


if __name__ == "__main__":  # guard required: evaluation uses worker processes
    main()
