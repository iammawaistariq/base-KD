from __future__ import annotations

import torch
from torch.nn import functional as F


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

    def update(self, value: float, n: int) -> None:
        self.sum += value * n
        self.count += n


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[torch.Tensor]:
    maxk = min(max(topk), output.shape[1])
    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    out = []
    for k in topk:
        k = min(k, output.shape[1])
        out.append(correct[:k].reshape(-1).float().sum(0).mul_(100.0 / target.numel()))
    return out


@torch.no_grad()
def evaluate_classifier(model, loader, device) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        loss_meter.update(float(loss), images.size(0))
        acc1_meter.update(float(acc1), images.size(0))
        acc5_meter.update(float(acc5), images.size(0))
    return {"loss": loss_meter.avg, "acc1": acc1_meter.avg, "acc5": acc5_meter.avg}
