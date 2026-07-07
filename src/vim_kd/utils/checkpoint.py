from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_model_checkpoint(model: nn.Module, path: str | Path, map_location: str | torch.device = "cpu") -> None:
    payload = torch.load(path, map_location=map_location)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
