#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from vim_kd.config import load_config
from vim_kd.data.build import build_dataloaders
from vim_kd.engine.trainer import build_student_or_teacher
from vim_kd.utils.checkpoint import load_model_checkpoint
from vim_kd.utils.metrics import evaluate_classifier
from vim_kd.utils.seed import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    _, val_loader = build_dataloaders(cfg["dataset"])
    model = build_student_or_teacher(cfg).to(device)
    load_model_checkpoint(model, args.checkpoint, map_location=device)
    metrics = evaluate_classifier(model, val_loader, device)
    print(f"loss={metrics['loss']:.4f} acc1={metrics['acc1']:.2f} acc5={metrics['acc5']:.2f}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
