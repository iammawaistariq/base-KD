from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_target_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    log_p = F.log_softmax(student_logits / temperature, dim=-1)
    q = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean") * temperature * temperature


def token_relation_loss(student_tokens: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
    student_tokens = F.normalize(student_tokens, dim=-1)
    teacher_tokens = F.normalize(teacher_tokens, dim=-1)
    s_rel = student_tokens @ student_tokens.transpose(1, 2)
    t_rel = teacher_tokens @ teacher_tokens.transpose(1, 2)
    return F.mse_loss(s_rel, t_rel)


class DistillationCriterion(nn.Module):
    def __init__(
        self,
        temperature: float,
        ce_weight: float,
        kd_weight: float,
        feature_weight: float,
        relation_weight: float,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.kd_weight = kd_weight
        self.feature_weight = feature_weight
        self.relation_weight = relation_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_tokens: torch.Tensor | None = None,
        teacher_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce = F.cross_entropy(student_logits, labels)
        kd = soft_target_kl(student_logits, teacher_logits.detach(), self.temperature)
        loss = self.ce_weight * ce + self.kd_weight * kd
        feature = student_logits.new_tensor(0.0)
        relation = student_logits.new_tensor(0.0)

        if student_tokens is not None and teacher_tokens is not None:
            feature = F.mse_loss(student_tokens, teacher_tokens.detach())
            relation = token_relation_loss(student_tokens, teacher_tokens.detach())
            loss = loss + self.feature_weight * feature + self.relation_weight * relation

        return loss, {
            "ce": float(ce.detach()),
            "kd": float(kd.detach()),
            "feature": float(feature.detach()),
            "relation": float(relation.detach()),
        }
