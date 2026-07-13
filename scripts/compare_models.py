#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from vim_kd.config import load_config
from vim_kd.data.build import build_dataloaders
from vim_kd.models.factory import build_model
from vim_kd.utils.metrics import evaluate_classifier
from vim_kd.utils.seed import resolve_device


DEFAULT_RUNS = [
    ("vit_scratch", "configs/cifar10_vit_scratch.yaml", "runs/cifar10_vit_scratch/best.pt"),
    ("mamba_scratch", "configs/cifar10_mamba_scratch.yaml", "runs/cifar10_mamba_scratch/best.pt"),
    (
        "vit_pretrained_finetuned",
        "configs/cifar10_vit_pretrained_finetune.yaml",
        "runs/cifar10_vit_pretrained_finetuned/best.pt",
    ),
    (
        "mamba_kd_from_scratch_vit",
        "configs/cifar10_vit_scratch_to_mamba_kd.yaml",
        "runs/cifar10_vit_scratch_to_mamba_kd/best.pt",
    ),
    (
        "mamba_kd_from_finetuned_vit",
        "configs/cifar10_vit_pretrained_to_mamba_kd.yaml",
        "runs/cifar10_vit_pretrained_to_mamba_kd/best.pt",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare teacher, Mamba, and Mamba-KD checkpoints.")
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("NAME", "CONFIG", "CHECKPOINT"),
        help="Add a model to compare. Can be passed multiple times.",
    )
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state, dict):
        raise TypeError(f"No model state dictionary found in {path}")
    return state


def build_eval_model(cfg: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    state = load_state(checkpoint)
    model_cfg = dict(cfg.get("student") or cfg.get("model"))
    if (
        model_cfg.get("name", "").lower() == "vision_mamba"
        and any(".mixer.A_log" in key for key in state)
    ):
        model_cfg["mamba_backend"] = "mamba_reference"
    if model_cfg.get("name", "").lower() == "timm":
        model_cfg["pretrained"] = False
    model = build_model(model_cfg, cfg["dataset"]).to(device)
    model.load_state_dict(state, strict=True)
    return model


def main() -> None:
    args = parse_args()
    runs = args.run if args.run else DEFAULT_RUNS

    rows = []
    val_loader = None
    device = None
    for name, config_path, checkpoint_path in runs:
        cfg = load_config(config_path)
        if device is None:
            device = resolve_device(cfg.get("device", "auto"))
        if val_loader is None:
            _, val_loader = build_dataloaders(cfg["dataset"])

        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            rows.append((name, config_path, str(checkpoint), "missing", "-", "-", "-"))
            continue

        model = build_eval_model(cfg, checkpoint, device)
        metrics = evaluate_classifier(model, val_loader, device)
        rows.append(
            (
                name,
                config_path,
                str(checkpoint),
                "ok",
                f"{count_parameters(model) / 1_000_000:.2f}",
                f"{metrics['loss']:.4f}",
                f"{metrics['acc1']:.2f}",
            )
        )

    headers = ("model", "config", "checkpoint", "status", "params_m", "loss", "acc1")
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
