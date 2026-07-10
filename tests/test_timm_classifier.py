import sys
import types

import torch

from vim_kd.models.factory import TimmClassifier


class _FakeTimmModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.forward_features_calls = 0
        self.forward_head_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return torch.zeros(x.shape[0], 10)

    def forward_features(self, x):
        self.forward_features_calls += 1
        return torch.ones(x.shape[0], 5, 10)

    def forward_head(self, features):
        self.forward_head_calls += 1
        return features[:, 0, :10]


def test_timm_classifier_reuses_features_for_logits(monkeypatch):
    fake_model = _FakeTimmModel()
    fake_timm = types.SimpleNamespace(create_model=lambda *args, **kwargs: fake_model)
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    model = TimmClassifier("fake_vit", num_classes=10, pretrained=False)
    logits, features = model(torch.randn(2, 3, 224, 224), return_features=True)

    assert logits.shape == (2, 10)
    assert features["tokens"].shape == (2, 5, 10)
    assert fake_model.forward_features_calls == 1
    assert fake_model.forward_head_calls == 1
    assert fake_model.forward_calls == 0
