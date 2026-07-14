from __future__ import annotations

from copy import deepcopy
from typing import Any


def evaluation_dataset_config(
    dataset_cfg: dict[str, Any],
    split: str,
) -> tuple[dict[str, Any], str]:
    """Return an isolated dataset config and a truthful display label."""
    eval_cfg = deepcopy(dataset_cfg)
    dataset_name = str(eval_cfg.get("name", "")).lower()
    validation_fraction = float(eval_cfg.get("validation_fraction", 0.0))

    if split == "test" and dataset_name in {"cifar10", "cifar100"}:
        eval_cfg["validation_fraction"] = 0.0
        return eval_cfg, "test"
    if split == "config":
        return eval_cfg, "validation" if validation_fraction else "test"

    # ImageFolder has a configured validation directory but no standard test split.
    return eval_cfg, "validation"
