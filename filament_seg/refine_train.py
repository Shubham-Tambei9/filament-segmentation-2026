"""Train and evaluate the instance refiner (see :mod:`refine`).

Example (cross-fitted evaluation on the out-of-fold maps, then a final model on all crops):
    python -m filament_seg.refine_train --crops work/refine/crops_r34_1024 \
        --prob-dir work/oof/resnet34_s1024_d4 --out weights/refiner.pt

Cross-fitting: the refiner applied to the images of fold k is trained only on crops from the other
folds, so the reported PQ gain is out-of-fold.  The refined result is compared with the unrefined
rules on exactly the same images and probability maps.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from .annotations import label_map, load_annotations, make_folds
from .dataset import PIXEL_MEAN, PIXEL_STD
from .imaging import read_gray, upsample_prob
from .inference import amp_dtype
from .metrics import PQAccumulator
from .paths import TRAIN_IMAGES, TRAIN_JSON, WEIGHTS_DIR, configure_caches
from .postprocess import PostprocessParams, probs_to_labels
from .refine import crop_geometry, crop_inputs, paste_back


def build_refiner(pretrained: bool = True) -> torch.nn.Module:
    configure_caches()
    return smp.Unet("resnet18", encoder_weights="imagenet" if pretrained else None, in_channels=4, classes=1)


def _prep(x: torch.Tensor) -> torch.Tensor:
    """uint8 crops (B, 4, H, W) -> float network input (image channel standardised)."""
    x = x.float() / 255.0
    x[:, 0] = (x[:, 0] - PIXEL_MEAN) / PIXEL_STD
    return x


def crop_store(crop_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All crops as memory-mapped arrays (x, y) plus the image stem of every row.

    The per-image ``.npz`` files are consolidated once into uncompressed ``.npy`` files next to them;
    memory-mapping keeps them out of the process's committed memory (the training of the main
    networks runs at the same time on this machine).
    """
    xf, yf, sf = crop_dir.with_suffix(".x.npy"), crop_dir.with_suffix(".y.npy"), crop_dir.with_suffix(".stems.npy")
    if not sf.exists():
        files = [f for f in sorted(crop_dir.glob("*.npz"))]
        counts = [len(np.load(f)["drawn"]) for f in files]
        n = sum(counts)
        h, w = next(np.load(f)["y"].shape[1:] for f, c in zip(files, counts) if c)
        x = np.lib.format.open_memmap(xf, mode="w+", dtype=np.uint8, shape=(n, 4, h, w))
        y = np.lib.format.open_memmap(yf, mode="w+", dtype=np.uint8, shape=(n, h, w))
        stems, i = [], 0
        for f, c in zip(files, counts):
            if c:
                d = np.load(f)
                x[i:i + c], y[i:i + c] = d["x"], d["y"]
                stems += [f.stem] * c
                i += c
        x.flush(); y.flush()
        del x, y
        np.save(sf, np.array(stems))
    return np.load(xf, mmap_mode="r"), np.load(yf, mmap_mode="r"), np.load(sf)


def train_refiner(x: np.ndarray, y: np.ndarray, rows: np.ndarray, device, epochs: int = 12, batch: int = 32,
                  lr: float = 1e-3, seed: int = 0) -> torch.nn.Module:
    """Train on crops ``x[rows]`` / ``y[rows]`` (arrays may be memory-mapped)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_refiner().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(rows) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    amp = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16 and device.type == "cuda")
    for epoch in range(epochs):
        order = rng.permutation(rows)
        t0, losses = time.time(), []
        for i in range(len(rows) // batch):
            idx = np.sort(order[i * batch:(i + 1) * batch])
            xb = _prep(torch.from_numpy(x[idx]).to(device))
            yb = torch.from_numpy(y[idx]).to(device).float()[:, None] / 255.0
            if rng.random() < 0.5:  # the rotated crop is symmetric under both flips
                xb, yb = xb.flip(3), yb.flip(3)
            if rng.random() < 0.5:
                xb, yb = xb.flip(2), yb.flip(2)
            xb[:, 0] = xb[:, 0] * (1 + 0.1 * torch.randn(len(xb), 1, 1, device=device))
            with torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == "cuda"):
                logits = model(xb)
            logits = logits.float()
            p = torch.sigmoid(logits)
            dice = 1 - (2 * (p * yb).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + yb.sum((1, 2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(logits, yb) + 0.5 * dice.mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        print(f"  epoch {epoch}: loss {np.mean(losses):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model.eval()


@torch.no_grad()
def refine_crops(model, image: np.ndarray, prob: np.ndarray, labels: np.ndarray, device):
    """Refiner output (flip TTA) for every instance of ``labels``: (crop geometries, crop probabilities)."""
    n = int(labels.max())
    if n == 0:
        return [], []
    geoms = [crop_geometry(labels == k) for k in range(1, n + 1)]
    inp = np.stack([crop_inputs(image, prob, labels, k, g) for k, g in enumerate(geoms, start=1)])
    xb = _prep(torch.from_numpy(np.round(inp * 255).astype(np.uint8)).to(device))
    acc = torch.zeros(xb.shape[0], 1, *xb.shape[2:], device=device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype(device), enabled=device.type == "cuda"):
        for dims in ([], [3], [2], [2, 3]):
            xi = xb.flip(dims) if dims else xb
            out = torch.sigmoid(model(xi).float())
            acc += out.flip(dims) if dims else out
    return geoms, list((acc / 4)[:, 0].cpu().numpy())


def refine_labels(model, image: np.ndarray, prob: np.ndarray, labels: np.ndarray, threshold: float, device) -> np.ndarray:
    """Refine every instance of ``labels`` (flip TTA) and paste back into a disjoint label map."""
    geoms, crops = refine_crops(model, image, prob, labels, device)
    if not geoms:
        return labels
    return paste_back(labels.shape, geoms, crops, threshold, labels)


def _load_params(spec: str) -> PostprocessParams:
    """A rule set from a JSON file or an inline JSON object."""
    text = spec if spec.lstrip().startswith("{") else Path(spec).read_text()
    return PostprocessParams(**json.loads(text))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", required=True)
    p.add_argument("--prob-dir", required=True)
    p.add_argument("--params", default=str(WEIGHTS_DIR / "postprocess.json"))
    p.add_argument("--extra-params", nargs="*", default=[],
                   help="further rule sets (JSON files or inline JSON) whose instances are refined by the same "
                        "cross-fitted models: the rules that suit the refiner need not be the best rules alone")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.5])
    p.add_argument("--out", default=None, help="also train a final refiner on all crops and save it here")
    p.add_argument("--params-out", default=None, help="with --out: write the chosen rule set here")
    p.add_argument("--save-folds", default=None, help="save the cross-fitted fold models here (fold<k>.pt)")
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_annotations(TRAIN_JSON)
    folds = make_folds(items, 5, 42)
    variants = [_load_params(args.params)] + [_load_params(e) for e in args.extra_params]
    crop_dir, prob_dir = Path(args.crops), Path(args.prob_dir)
    base = [PQAccumulator() for _ in variants]
    refined = {(v, t): PQAccumulator() for v in range(len(variants)) for t in args.thresholds}
    x, y, row_stem = crop_store(crop_dir)
    fold_of = {it.stem: int(f) for it, f in zip(items, folds)}
    row_fold = np.array([fold_of[s] for s in row_stem])
    pool = ThreadPoolExecutor(max_workers=min(len(variants), 4))
    for k in range(5):
        rows = np.flatnonzero(row_fold != k)
        print(f"fold {k}: training on {len(rows)} crops", flush=True)
        model = train_refiner(x, y, rows, device, epochs=args.epochs)
        if args.save_folds:
            Path(args.save_folds).mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {n: w.half() for n, w in model.state_dict().items()}, "fold": k},
                       Path(args.save_folds) / f"fold{k}.pt")
        for it in (it for it, f in zip(items, folds) if f == k):
            prob = upsample_prob(np.load(prob_dir / f"{it.stem}.npy"))
            image = read_gray(TRAIN_IMAGES / it.file_name)
            records = [label_map(f) for f in it.annotations.values()]

            def evaluate(v, prob=prob, image=image, records=records, model=model):
                labels = probs_to_labels(prob, variants[v])
                geoms, crops = refine_crops(model, image, prob, labels, device)
                for gt in records:
                    base[v].update(gt, labels)
                for t in args.thresholds:
                    out = paste_back(labels.shape, geoms, crops, t, labels) if geoms else labels
                    for gt in records:
                        refined[v, t].update(gt, out)

            list(pool.map(evaluate, range(len(variants))))  # rule sets in parallel threads
        for v in range(len(variants)):
            head = f"after fold {k}:" if v == 0 else f"  rules {v} after fold {k}:"
            print(f"{head} rules PQ {base[v].summary()['PQ']:.4f} | " + " | ".join(
                f"refined@{t} PQ {refined[v, t].summary()['PQ']:.4f}" for t in args.thresholds), flush=True)
        del model
        torch.cuda.empty_cache()

    for v, params in enumerate(variants):
        s = base[v].summary()
        ind = "" if v == 0 else "  "
        if v:
            print(f"\nrules {v}: {json.dumps(params.as_dict())}")
        else:
            print()
        print(f"{ind}rules:   PQ {s['PQ']:.4f} SQ {s['SQ']:.3f} RQ {s['RQ']:.3f} TP {s['TP']} FP {s['FP']} FN {s['FN']}")
        for t in args.thresholds:
            r = refined[v, t].summary()
            print(f"{ind}refined@{t}: PQ {r['PQ']:.4f} SQ {r['SQ']:.3f} RQ {r['RQ']:.3f} "
                  f"TP {r['TP']} FP {r['FP']} FN {r['FN']}")
    best_v, best_t = max(refined, key=lambda vt: refined[vt].summary()["PQ"])
    best_pq = refined[best_v, best_t].summary()["PQ"]
    print(f"\nbest: rules {best_v} threshold {best_t} PQ {best_pq:.4f} {json.dumps(variants[best_v].as_dict())}",
          flush=True)

    if args.out:
        model = train_refiner(x, y, np.arange(len(row_stem)), device, epochs=args.epochs)
        torch.save({"state_dict": {k: v.half() for k, v in model.state_dict().items()},
                    "threshold": best_t, "params": variants[best_v].as_dict(),
                    "cv_pq": best_pq, "rules_pq": base[best_v].summary()["PQ"],
                    "cv_summary": refined[best_v, best_t].summary(), "rules_summary": base[best_v].summary()},
                   args.out)
        if args.params_out:
            Path(args.params_out).write_text(json.dumps(variants[best_v].as_dict(), indent=2))
        print("saved", args.out, "threshold", best_t, "rules", best_v)


if __name__ == "__main__":
    main()
