"""Train the filament U-Net on one cross-validation fold (or on all data with ``--fold -1``).

Example:
    python -m filament_seg.train --fold 0 --size 1024 --encoder resnet34 --epochs 40

Outputs go to ``work/runs/<name>/``: ``best.pt`` (highest validation PQ), ``last.pt``,
``log.jsonl``, and the best model's TTA validation probabilities (``val_probs/*.npy``,
float16 at network resolution) used later to tune post-processing.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .annotations import load_annotations, make_folds
from .dataset import FilamentDataset, build_cache, read_gray
from .evaluation import evaluate_probs
from .inference import predict_prob, upsample_prob
from .model import build_model, load_checkpoint, save_checkpoint
from .paths import TRAIN_IMAGES, TRAIN_JSON, WORK_DIR
from .postprocess import PostprocessParams


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", type=int, default=0, help="validation fold; -1 = train on everything")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--size", type=int, default=1024, help="network input resolution (full disk)")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--arch", default="unet", choices=["unet", "unetplusplus"])
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16"], help="fp16 (+ grad scaling) for GPUs without bf16, e.g. Kaggle T4/P100")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default=None, help="run name (default: <encoder>_s<size>_f<fold>)")
    p.add_argument("--limit", type=int, default=0, help="debug: use only N train / N//4 val images")
    return p.parse_args(argv)


def loss_fn(logits: torch.Tensor, target: torch.Tensor, dice_weight: float) -> torch.Tensor:
    """BCE on the soft agreement target + batch soft-Dice.

    BCE alone is dominated by the ~99.7% background pixels; Dice counters the imbalance and
    directly optimises overlap, which is what SQ/RQ reward.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)
    return bce + dice_weight * dice


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def validate(model, items, size, device, tta=False, params=PostprocessParams(), save_dir: Path | None = None):
    model.eval()

    def get_prob(stem: str) -> np.ndarray:  # called once per image by evaluate_probs
        prob = predict_prob(model, read_gray(TRAIN_IMAGES / f"{stem}.jpeg"), size, tta, device)
        if save_dir is not None:
            np.save(save_dir / f"{stem}.npy", prob.astype(np.float16))
        return upsample_prob(prob)

    return evaluate_probs(get_prob, items, params)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix = args.encoder if args.arch == "unet" else f"{args.arch}_{args.encoder}"
    name = args.name or f"{prefix}_s{args.size}_f{args.fold}"
    out = WORK_DIR / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    items = load_annotations(TRAIN_JSON)
    folds = make_folds(items, args.n_folds, args.seed)
    train_items = [it for it, f in zip(items, folds) if f != args.fold]
    val_items = [it for it, f in zip(items, folds) if f == args.fold]
    if args.limit:
        train_items, val_items = train_items[: args.limit], val_items[: max(args.limit // 4, 1)]
    cache_dir = WORK_DIR / "cache" / f"size{args.size}"
    build_cache(train_items + val_items, TRAIN_IMAGES, cache_dir, args.size)
    print(f"train {len(train_items)} images | val {len(val_items)} images | device {device}")

    train_ds = FilamentDataset([it.stem for it in train_items], cache_dir, train=True)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,  # re-spawning workers costs seconds per epoch on Windows
    )

    model = build_model(args.encoder, arch=args.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, total, warmup=len(loader)))
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and device.type == "cuda")
    meta = {"encoder": args.encoder, "arch": args.arch, "size": args.size, "fold": args.fold}

    best_pq, log = -1.0, open(out / "log.jsonl", "a")
    for epoch in range(args.epochs):
        model.train()
        t0, losses = time.time(), []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), y, args.dice_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"],
               "time_s": round(time.time() - t0, 1)}

        is_last = epoch == args.epochs - 1
        if val_items and ((epoch + 1) % args.val_every == 0 or is_last):
            rec.update({f"val_{k}": v for k, v in validate(model, val_items, args.size, device).items()})
            if rec["val_PQ"] > best_pq:
                best_pq = rec["val_PQ"]
                save_checkpoint(out / "best.pt", model, {**meta, "epoch": epoch, "val_PQ": best_pq})
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n")
        log.flush()
    save_checkpoint(out / "last.pt", model, {**meta, "epoch": args.epochs - 1})

    if val_items:  # TTA probabilities of the best model, for post-processing tuning
        best, _ = load_checkpoint(out / "best.pt", device)
        prob_dir = out / "val_probs"
        prob_dir.mkdir(exist_ok=True)
        summary = validate(best, val_items, args.size, device, tta="d4", save_dir=prob_dir)
        print("best model, D4 TTA, default post-processing:", json.dumps(summary))
        (out / "val_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
