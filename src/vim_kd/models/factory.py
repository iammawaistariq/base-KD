from __future__ import annotations

from typing import Any

from torch import nn

from .tiny_vit import TinyVisionTransformer
from .vision_mamba import VisionMamba


class TimmClassifier(nn.Module):
    def __init__(self, timm_name: str, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

    def _logits_from_features(self, features_raw, x):
        if hasattr(self.model, "forward_head"):
            return self.model.forward_head(features_raw)
        return self.model(x)

    def forward(self, x, return_features: bool = False):
        if return_features and hasattr(self.model, "forward_features"):
            features_raw = self.model.forward_features(x)
            if isinstance(features_raw, dict):
                tokens = features_raw.get("x", None)
                if tokens is None:
                    tokens = features_raw.get("tokens", None)
                if tokens is None:
                    raise ValueError("Could not find token tensor in timm feature dict.")
            elif isinstance(features_raw, (tuple, list)):
                tokens = features_raw[-1]
            else:
                tokens = features_raw
            if tokens.ndim == 4:
                tokens = tokens.flatten(2).transpose(1, 2)
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(1)
            if tokens.ndim != 3:
                raise ValueError(f"Expected timm features to be rank 2, 3, or 4; got {tokens.shape}.")
            cls = tokens[:, 0]
            logits = self._logits_from_features(features_raw, x)
            return logits, {"tokens": tokens, "cls": cls, "patch_tokens": tokens[:, 1:]}
        return self.model(x)


def build_model(cfg: dict[str, Any], dataset_cfg: dict[str, Any]):
    name = cfg["name"].lower()
    num_classes = int(dataset_cfg["num_classes"])
    img_size = int(dataset_cfg.get("image_size", 224))
    common = {"img_size": img_size, "num_classes": num_classes}

    if name == "tiny_vit":
        keys = ["patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio", "drop_rate"]
        return TinyVisionTransformer(**common, **{k: cfg[k] for k in keys if k in cfg})
    if name == "vision_mamba":
        keys = [
            "patch_size",
            "embed_dim",
            "depth",
            "state_dim",
            "conv_kernel",
            "expand",
            "bidirectional",
            "drop_rate",
            "mamba_backend",
            "pool",
        ]
        return VisionMamba(**common, **{k: cfg[k] for k in keys if k in cfg})
    if name == "timm":
        return TimmClassifier(cfg["timm_name"], num_classes=num_classes, pretrained=cfg.get("pretrained", True))
    raise ValueError(f"Unknown model name: {name}")
