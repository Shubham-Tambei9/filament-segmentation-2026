"""Reproduce the organisers' self-evaluation analysis on our out-of-fold predictions.

Definitions follow the organisers' public self-evaluation notebook: for every (image, annotator record)
pair, IoU and Dice are computed between every reference filament and every predicted filament;
the distributions use all pairs with overlap > 0; the matching-case chart counts, for each reference
filament, how many predictions overlap it (1:n) and, for each prediction, how many reference
filaments it overlaps (n:1).  The same analysis between two annotators gives the human baseline.

    python report/official_eval.py --oof work/oof/blendF_211 --params weights/postprocess_blendF.json
Writes report/figures/official_eval.pdf and report/official_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from filament_seg import paths  # noqa: E402
from filament_seg.annotations import label_map, load_annotations  # noqa: E402
from filament_seg.evaluation import prob_loader  # noqa: E402
from filament_seg.postprocess import PostprocessParams, probs_to_labels  # noqa: E402

plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})


def overlaps(gt: np.ndarray, pred: np.ndarray):
    """IoU and Dice matrices (n_gt x n_pred) of two label maps (0 = background)."""
    ng, npd = int(gt.max()), int(pred.max())
    m = (gt > 0) | (pred > 0)
    c = np.bincount(gt[m].astype(np.int64) * (npd + 1) + pred[m], minlength=(ng + 1) * (npd + 1))
    c = c.reshape(ng + 1, npd + 1)
    inter = c[1:, 1:].astype(np.float64)
    ga, pa = c[1:, :].sum(1)[:, None], c[:, 1:].sum(0)[None, :]
    union = ga + pa - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    dice = np.divide(2 * inter, ga + pa, out=np.zeros_like(inter), where=(ga + pa) > 0)
    return iou, dice


class Acc:
    def __init__(self):
        self.iou, self.dice, self.gt_deg, self.pred_deg = [], [], Counter(), Counter()
        self.tp_iou, self.fp, self.fn = [], 0, 0

    def update(self, gt, pred):
        iou, dice = overlaps(gt, pred)
        hit = iou > 0
        self.iou += iou[hit].tolist()
        self.dice += dice[hit].tolist()
        self.gt_deg.update(hit.sum(1).tolist())
        self.pred_deg.update(hit.sum(0).tolist())
        m = iou > 0.5
        self.tp_iou += iou[m].tolist()
        self.fp += int((m.sum(0) == 0).sum())
        self.fn += int((m.sum(1) == 0).sum())

    def merge(self, o):
        self.iou += o.iou; self.dice += o.dice; self.tp_iou += o.tp_iou  # noqa: E702
        self.gt_deg += o.gt_deg; self.pred_deg += o.pred_deg; self.fp += o.fp; self.fn += o.fn  # noqa: E702

    def summary(self):
        tp = len(self.tp_iou)
        return {
            "PQ": sum(self.tp_iou) / (tp + 0.5 * self.fp + 0.5 * self.fn),
            "mean_IoU_overlapping": float(np.mean(self.iou)), "mean_Dice_overlapping": float(np.mean(self.dice)),
            "gt_hits": sum(v for k, v in self.gt_deg.items() if k > 0), "gt_misses": self.gt_deg.get(0, 0),
            "pred_hits": sum(v for k, v in self.pred_deg.items() if k > 0), "pred_misses": self.pred_deg.get(0, 0),
            "gt_1to1": self.gt_deg.get(1, 0), "gt_1to2plus": sum(v for k, v in self.gt_deg.items() if k >= 2),
            "pred_1to1": self.pred_deg.get(1, 0), "pred_2plus_to1": sum(v for k, v in self.pred_deg.items() if k >= 2),
        }


def job(args):
    stems, oof, params = args
    items = {it.stem: it for it in load_annotations(paths.TRAIN_JSON)}
    loader = prob_loader(Path(oof))
    model, human = Acc(), Acc()
    for s in stems:
        it = items[s]
        records = [label_map(f) for f in it.annotations.values()]
        pred = probs_to_labels(loader(s), params)
        for gt in records:
            model.update(gt, pred)
        for a, b in combinations(records, 2):
            human.update(a, b)
    return model, human


def plot_m2n(ax, acc, title):
    off = 0.5
    for deg, n in acc.gt_deg.items():
        ax.bar(deg + off, n, width=0.8, color="crimson" if deg == 0 else "forestgreen", alpha=0.8)
    for deg, n in acc.pred_deg.items():
        ax.bar(-(deg + off), n, width=0.8, color="darkorange" if deg == 0 else "steelblue", alpha=0.8)
    ks = sorted({d for d in acc.gt_deg if d <= 4} | {1})
    pos = [k + off for k in ks] + [-(k + off) for k in sorted({d for d in acc.pred_deg if d <= 4} | {1})]
    ax.set_xticks(pos, [f"1:{int(p - off)}" if p > 0 else f"{int(-p - off)}:1" for p in pos], rotation=60)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlim(-5, 5)
    ax.set(title=title, ylabel="count")


def write_macros(out):
    """report/official_numbers.tex: \\offModelIoU, \\offHumanIoU, ... (percentages for split/merge rates)."""
    lines = ["% generated by report/official_eval.py -- do not edit by hand"]
    for tag, s in (("Model", out["model"]), ("Human", out["annotator_vs_annotator"])):
        n_gt, n_pred = s["gt_hits"] + s["gt_misses"], s["pred_hits"] + s["pred_misses"]
        vals = {"PQ": f"{s['PQ']:.3f}", "IoU": f"{s['mean_IoU_overlapping']:.3f}",
                "Dice": f"{s['mean_Dice_overlapping']:.3f}",
                "Split": f"{100 * s['gt_1to2plus'] / n_gt:.1f}", "Merge": f"{100 * s['pred_2plus_to1'] / n_pred:.1f}",
                "Miss": f"{100 * s['gt_misses'] / n_gt:.1f}", "Spurious": f"{100 * s['pred_misses'] / n_pred:.1f}"}
        lines += [f"\\newcommand{{\\off{tag}{k}}}{{{v}}}" for k, v in vals.items()]
    (ROOT / "report" / "official_numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oof", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    params = PostprocessParams(**json.loads(Path(a.params).read_text()))
    stems = sorted(f.stem for f in Path(a.oof).glob("*.npy"))
    with Pool(a.workers) as pool:
        res = pool.map(job, [(stems[i::a.workers], a.oof, params) for i in range(a.workers)])
    model, human = res[0]
    for m, h in res[1:]:
        model.merge(m)
        human.merge(h)
    out = {"n_images": len(stems), "model": model.summary(), "annotator_vs_annotator": human.summary()}
    (ROOT / "report" / "official_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    write_macros(out)

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
    bins = np.linspace(0, 1, 26)
    for i, (key, lab) in enumerate([("iou", "IoU"), ("dice", "Dice")]):
        ax[i].hist(getattr(human, key), bins=bins, density=True, alpha=0.5, color="gray", label="annotator vs annotator")
        ax[i].hist(getattr(model, key), bins=bins, density=True, alpha=0.6, color="#0072B2", label="model vs annotators")
        ax[i].axvline(np.mean(getattr(model, key)), color="crimson", ls="--", lw=1)
        ax[i].set(xlabel=f"{lab} of overlapping pairs", ylabel="density" if i == 0 else None)
    ax[0].legend(fontsize=6, frameon=False)
    plot_m2n(ax[2], model, "matching cases (IoU > 0)")
    fig.tight_layout()
    (ROOT / "report" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "report" / "figures" / "official_eval.pdf")


if __name__ == "__main__":
    main()
