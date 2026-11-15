"""Uncertainty of the final out-of-fold PQ: per-fold spread and bootstrap confidence intervals.

Repeats the cross-fitted evaluation of the final pipeline exactly as ``refine_train`` scores it (rules from
``weights/postprocess_blendH_FR.json`` on the out-of-fold blend, each fold refined by the refiner that never
saw that fold), but keeps the PQ statistics of every image, so the result can be resampled:

* per-fold PQ (mean and standard deviation over the five folds),
* a 95% bootstrap interval of the pooled PQ (images resampled with replacement),
* a paired bootstrap interval of the refiner's gain over the rules.

Writes ``report/uncertainty.json`` and ``report/uncertainty_numbers.tex``.

    python report/uncertainty.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from filament_seg.annotations import label_map, load_annotations, make_folds  # noqa: E402
from filament_seg.imaging import read_gray, upsample_prob  # noqa: E402
from filament_seg.metrics import PQAccumulator  # noqa: E402
from filament_seg.paths import TRAIN_IMAGES, TRAIN_JSON  # noqa: E402
from filament_seg.postprocess import PostprocessParams, probs_to_labels  # noqa: E402
from filament_seg.refine import paste_back  # noqa: E402
from filament_seg.refine_train import build_refiner, refine_crops  # noqa: E402

PROB_DIR = ROOT / "work" / "oof" / "blendH_FR"
FOLD_DIR = ROOT / "work" / "refine" / "folds_blendH_FR"
PARAMS = ROOT / "weights" / "postprocess_blendH_FR.json"
THRESHOLD = 0.5
N_BOOT, SEED = 2000, 0


def stats(acc: PQAccumulator) -> list[float]:
    """Additive statistics of one image: summed matched IoU, TP, FP, FN (pooled over its annotators)."""
    return [acc.sum_iou, acc.tp, acc.fp, acc.fn]


def pq(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    return float(s[..., 0] / (s[..., 1] + 0.5 * s[..., 2] + 0.5 * s[..., 3]))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = PostprocessParams(**json.loads(PARAMS.read_text()))
    items = load_annotations(TRAIN_JSON)
    folds = make_folds(items, 5, 42)
    rows = []  # (fold, rules stats, refined stats) per image
    for k in range(5):
        ck = torch.load(FOLD_DIR / f"fold{k}.pt", map_location="cpu", weights_only=False)
        model = build_refiner(pretrained=False)
        model.load_state_dict({n: w.float() for n, w in ck["state_dict"].items()})
        model.to(device).eval()
        for it in (it for it, f in zip(items, folds) if f == k):
            prob = upsample_prob(np.load(PROB_DIR / f"{it.stem}.npy"))
            image = read_gray(TRAIN_IMAGES / it.file_name)
            labels = probs_to_labels(prob, params)
            geoms, crops = refine_crops(model, image, prob, labels, device)
            out = paste_back(labels.shape, geoms, crops, THRESHOLD, labels) if geoms else labels
            base, ref = PQAccumulator(), PQAccumulator()
            for ann in it.annotations.values():
                gt = label_map(ann)
                base.update(gt, labels)
                ref.update(gt, out)
            rows.append((k, stats(base), stats(ref)))
        print(f"fold {k} done ({sum(r[0] == k for r in rows)} images)", flush=True)
        del model
        torch.cuda.empty_cache()

    fold = np.array([r[0] for r in rows])
    base = np.array([r[1] for r in rows])
    ref = np.array([r[2] for r in rows])
    fold_pq = [pq(ref[fold == k].sum(0)) for k in range(5)]
    fold_pq_rules = [pq(base[fold == k].sum(0)) for k in range(5)]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), size=(N_BOOT, len(rows)))
    boot_ref = np.array([pq(ref[i].sum(0)) for i in idx])
    boot_base = np.array([pq(base[i].sum(0)) for i in idx])
    gain = boot_ref - boot_base
    res = {
        "n_images": len(rows),
        "pq_refined": pq(ref.sum(0)), "pq_rules": pq(base.sum(0)),
        "fold_pq_refined": fold_pq, "fold_pq_rules": fold_pq_rules,
        "fold_mean": float(np.mean(fold_pq)), "fold_std": float(np.std(fold_pq, ddof=1)),
        "fold_mean_rules": float(np.mean(fold_pq_rules)), "fold_std_rules": float(np.std(fold_pq_rules, ddof=1)),
        "boot_ci_refined": [float(np.percentile(boot_ref, 2.5)), float(np.percentile(boot_ref, 97.5))],
        "boot_ci_rules": [float(np.percentile(boot_base, 2.5)), float(np.percentile(boot_base, 97.5))],
        "boot_ci_gain": [float(np.percentile(gain, 2.5)), float(np.percentile(gain, 97.5))],
        "gain_positive_share": float((gain > 0).mean()),
        "n_boot": N_BOOT,
    }
    (ROOT / "report" / "uncertainty.json").write_text(json.dumps(res, indent=2))
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    tex = {
        "uncPQ": f3(res["pq_refined"]), "uncRulesPQ": f3(res["pq_rules"]),
        "uncFoldMean": f3(res["fold_mean"]), "uncFoldStd": f3(res["fold_std"]),
        "uncFoldMeanRules": f3(res["fold_mean_rules"]), "uncFoldStdRules": f3(res["fold_std_rules"]),
        "uncFoldMin": f3(min(fold_pq)), "uncFoldMax": f3(max(fold_pq)),
        "uncCILo": f3(res["boot_ci_refined"][0]), "uncCIHi": f3(res["boot_ci_refined"][1]),
        "uncRulesCILo": f3(res["boot_ci_rules"][0]), "uncRulesCIHi": f3(res["boot_ci_rules"][1]),
        "uncGainLo": f"{res['boot_ci_gain'][0]:+.4f}", "uncGainHi": f"{res['boot_ci_gain'][1]:+.4f}",
        "uncGainPos": f"{100 * res['gain_positive_share']:.2f}",  # never round 99.95 up to 100
        "uncNBoot": f"{N_BOOT:,}".replace(",", "{,}"),
    }
    (ROOT / "report" / "uncertainty_numbers.tex").write_text(
        "% generated by report/uncertainty.py\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in tex.items()))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
