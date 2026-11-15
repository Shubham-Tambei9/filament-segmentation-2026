"""Segmentation network: a U-Net with an ImageNet-pretrained encoder (segmentation_models_pytorch).

The input is single-channel H-alpha; smp adapts the pretrained RGB stem by summing its weights
over the colour channels.  The output is one logit per pixel ("is filament").
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch

from .paths import configure_caches
from .weights_store import ensure_weight


ARCHS = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}


def build_model(encoder: str = "resnet34", pretrained: bool = True, arch: str = "unet") -> torch.nn.Module:
    configure_caches()
    return ARCHS[arch](
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=1,
        classes=1,
    )


def save_checkpoint(path, model: torch.nn.Module, meta: dict) -> None:
    """Weights are stored in fp16 to keep the files small enough for a Git repo."""
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "meta": meta}, path)


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ensure_weight(path), map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["encoder"], pretrained=False, arch=meta.get("arch", "unet"))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval(), meta
