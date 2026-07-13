#!/usr/bin/env python
"""Evaluate a saved checkpoint on held-out CIFAR validation or official test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vim_kd.config import load_config  # noqa: E402
from vim_kd.data.build import build_dataloaders  # noqa: E402
from vim_kd.models.factory import build_model  # noqa: E402
from vim_kd.utils.metrics import evaluate_classifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--section", choices=("model", "teacher", "student"), default="model")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        help="Fail when top-1 accuracy (percentage) is below this threshold.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability()
    required = f"sm_{major}{minor}"
    compiled = set(torch.cuda.get_arch_list())
    if required not in compiled:
        message = (
            f"GPU requires {required}, but this PyTorch build supports "
            f"{', '.join(sorted(compiled))}."
        )
        if requested == "cuda":
            raise RuntimeError(message)
        print(message + " Using CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state, dict):
        raise TypeError(f"No model state dictionary found in {path}")
    return state


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = load_config(config_path)
    if args.section not in cfg:
        raise KeyError(f"Config has no {args.section!r} section.")

    dataset_cfg = dict(cfg["dataset"])
    dataset_cfg["batch_size"] = args.batch_size
    dataset_cfg["eval_batch_size"] = args.batch_size
    dataset_cfg["num_workers"] = args.num_workers
    if args.split == "test":
        dataset_cfg["validation_fraction"] = 0.0
    elif not float(dataset_cfg.get("validation_fraction", 0.0)):
        raise ValueError("Validation evaluation requires dataset.validation_fraction in the config.")

    state = load_state(checkpoint_path)
    model_cfg = dict(cfg[args.section])
    if model_cfg.get("name", "").lower() == "timm":
        model_cfg["pretrained"] = False
    if (
        model_cfg.get("name", "").lower() == "vision_mamba"
        and any(".mixer.A_log" in key for key in state)
    ):
        model_cfg["mamba_backend"] = "mamba_reference"

    device = resolve_device(args.device)
    print(f"Evaluating {args.section} on CIFAR-10 {args.split} split using {device}.")
    _, loader = build_dataloaders(dataset_cfg, logger=print)
    model = build_model(model_cfg, dataset_cfg).to(device)
    model.load_state_dict(state, strict=True)
    metrics = evaluate_classifier(model.eval(), loader, device)

    result = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "section": args.section,
        "split": args.split,
        "images": len(loader.dataset),
        "loss": metrics["loss"],
        "accuracy_top1_percent": metrics["acc1"],
        "accuracy_top5_percent": metrics["acc5"],
        "device": str(device),
    }
    print(
        f"images={result['images']} loss={metrics['loss']:.4f} "
        f"acc1={metrics['acc1']:.2f}% acc5={metrics['acc5']:.2f}%"
    )

    output = (
        Path(args.output)
        if args.output
        else checkpoint_path.parent / f"{args.section}_{args.split}_metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved metrics: {output}")

    if args.min_accuracy is not None and metrics["acc1"] < args.min_accuracy:
        raise SystemExit(
            f"Accuracy gate failed: {metrics['acc1']:.2f}% < {args.min_accuracy:.2f}%."
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()