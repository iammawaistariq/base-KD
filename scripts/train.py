#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vim_kd.config import load_config
from vim_kd.engine.trainer import train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train teacher or distill Vision Mamba student.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_from_config(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
