#!/usr/bin/env python
"""Run the corrected teacher, baseline, and KD experiment in a safe order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYTHON = sys.executable

SCRATCH_VIT_CONFIG = "configs/cifar10_vit_scratch.yaml"
SCRATCH_VIT_CHECKPOINT = "runs/cifar10_vit_scratch/best.pt"
PRETRAINED_TEACHER_CONFIG = "configs/cifar10_vit_pretrained_finetune.yaml"
PRETRAINED_TEACHER_CHECKPOINT = "runs/cifar10_vit_pretrained_finetuned/best.pt"
PLAIN_CONFIG = "configs/cifar10_mamba_scratch.yaml"
PLAIN_CHECKPOINT = "runs/cifar10_mamba_scratch/best.pt"
SCRATCH_VIT_KD_CONFIG = "configs/cifar10_vit_scratch_to_mamba_kd.yaml"
SCRATCH_VIT_KD_CHECKPOINT = "runs/cifar10_vit_scratch_to_mamba_kd/best.pt"
PRETRAINED_TEACHER_KD_CONFIG = "configs/cifar10_vit_pretrained_to_mamba_kd.yaml"
PRETRAINED_TEACHER_KD_CHECKPOINT = "runs/cifar10_vit_pretrained_to_mamba_kd/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--scratch-vit-min-accuracy", type=float, default=75.0)
    parser.add_argument("--pretrained-teacher-min-accuracy", type=float, default=90.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-plain-baseline", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def run(arguments: list[str]) -> None:
    print()
    print(">", " ".join(arguments), flush=True)
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(SRC) + os.pathsep + existing_pythonpath
    subprocess.run(arguments, cwd=ROOT, check=True, env=environment)


def train(config: str, checkpoint: str, skip_existing: bool) -> None:
    if skip_existing and (ROOT / checkpoint).is_file():
        print()
        print(f"Skipping existing checkpoint: {checkpoint}")
        return
    run([PYTHON, "scripts/train.py", "--config", config])


def verify(
    config: str,
    checkpoint: str,
    section: str,
    split: str,
    device: str,
    workers: int,
    minimum: float | None = None,
) -> None:
    command = [
        PYTHON,
        "scripts/verify_checkpoint.py",
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--section",
        section,
        "--split",
        split,
        "--device",
        device,
        "--num-workers",
        str(workers),
    ]
    if minimum is not None:
        command.extend(["--min-accuracy", str(minimum)])
    run(command)


def main() -> None:
    args = parse_args()

    train(SCRATCH_VIT_CONFIG, SCRATCH_VIT_CHECKPOINT, args.skip_existing)
    verify(
        SCRATCH_VIT_CONFIG,
        SCRATCH_VIT_CHECKPOINT,
        "model",
        "validation",
        args.device,
        args.num_workers,
        args.scratch_vit_min_accuracy,
    )
    verify(
        SCRATCH_VIT_CONFIG,
        SCRATCH_VIT_CHECKPOINT,
        "model",
        "test",
        args.device,
        args.num_workers,
    )

    train(PRETRAINED_TEACHER_CONFIG, PRETRAINED_TEACHER_CHECKPOINT, args.skip_existing)
    verify(
        PRETRAINED_TEACHER_CONFIG,
        PRETRAINED_TEACHER_CHECKPOINT,
        "model",
        "validation",
        args.device,
        args.num_workers,
        args.pretrained_teacher_min_accuracy,
    )
    verify(
        PRETRAINED_TEACHER_CONFIG,
        PRETRAINED_TEACHER_CHECKPOINT,
        "model",
        "test",
        args.device,
        args.num_workers,
    )

    if not args.skip_plain_baseline:
        train(PLAIN_CONFIG, PLAIN_CHECKPOINT, args.skip_existing)
        verify(
            PLAIN_CONFIG,
            PLAIN_CHECKPOINT,
            "model",
            "test",
            args.device,
            args.num_workers,
        )

    # These stages are reached only if both teachers passed their gates.
    train(SCRATCH_VIT_KD_CONFIG, SCRATCH_VIT_KD_CHECKPOINT, args.skip_existing)
    verify(
        SCRATCH_VIT_KD_CONFIG,
        SCRATCH_VIT_KD_CHECKPOINT,
        "student",
        "test",
        args.device,
        args.num_workers,
    )

    train(PRETRAINED_TEACHER_KD_CONFIG, PRETRAINED_TEACHER_KD_CHECKPOINT, args.skip_existing)
    verify(
        PRETRAINED_TEACHER_KD_CONFIG,
        PRETRAINED_TEACHER_KD_CHECKPOINT,
        "student",
        "test",
        args.device,
        args.num_workers,
    )

    print()
    print("Corrected experiment completed successfully.")


if __name__ == "__main__":
    main()
