from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from tqdm import tqdm

from vim_kd.data.build import build_dataloaders
from vim_kd.engine.losses import DistillationCriterion
from vim_kd.models.factory import build_model
from vim_kd.utils.checkpoint import load_model_checkpoint, save_checkpoint
from vim_kd.utils.metrics import AverageMeter, accuracy, evaluate_classifier
from vim_kd.utils.seed import resolve_device, set_seed


def log_step(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_student_or_teacher(cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("student") or cfg.get("model")
    if model_cfg is None:
        raise ValueError("Config needs either `student` or `model` section.")
    return build_model(model_cfg, cfg["dataset"])


class FeatureAdapter(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _distillation_uses_tokens(cfg: dict[str, Any]) -> bool:
    return (
        float(cfg.get("feature_weight", 0.0)) > 0.0
        or float(cfg.get("relation_weight", 0.0)) > 0.0
    )


def _cosine_scheduler(base_lr: float, min_lr: float, epoch: int, total: int, warmup: int) -> float:
    if epoch < warmup:
        return base_lr * float(epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _load_optional_checkpoint(model: nn.Module, cfg: dict[str, Any], device: torch.device) -> None:
    checkpoint = cfg.get("checkpoint")
    if checkpoint:
        load_model_checkpoint(model, checkpoint, map_location=device)


def _match_token_count(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if student.shape[1] == teacher.shape[1]:
        return student, teacher
    n = min(student.shape[1], teacher.shape[1])
    return student[:, :n], teacher[:, :n]


def _train_teacher_epoch(
    model, loader, optimizer, scaler, device, amp, grad_clip_norm, epoch: int, epochs: int
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    progress = tqdm(loader, desc=f"train {epoch}/{epochs}", dynamic_ncols=True)
    for step, (images, labels) in enumerate(progress, start=1):
        if step == 1:
            log_step(f"train: first batch loaded shape={tuple(images.shape)}")
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        if grad_clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        acc1 = accuracy(logits, labels, topk=(1,))[0]
        loss_meter.update(float(loss.detach()), images.size(0))
        acc_meter.update(float(acc1), images.size(0))
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}", acc1=f"{acc_meter.avg:.2f}")
    return {"loss": loss_meter.avg, "acc1": acc_meter.avg}


def _train_kd_epoch(
    student,
    teacher,
    adapter,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    amp,
    grad_clip_norm,
    epoch: int,
    epochs: int,
    use_token_distillation: bool,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    meters = {name: AverageMeter() for name in ["loss", "acc1", "ce", "kd", "feature", "relation"]}

    progress = tqdm(loader, desc=f"distill {epoch}/{epochs}", dynamic_ncols=True)
    for step, (images, labels) in enumerate(progress, start=1):
        if step == 1:
            log_step(f"distill: first batch loaded shape={tuple(images.shape)}")
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast(enabled=amp):
            teacher_out = teacher(images, return_features=use_token_distillation)
            if use_token_distillation:
                teacher_logits, teacher_features = teacher_out
            else:
                teacher_logits = teacher_out
                teacher_features = None
        with autocast(enabled=amp):
            student_out = student(images, return_features=use_token_distillation)
            if use_token_distillation:
                student_logits, student_features = student_out
                s_tokens, t_tokens = _match_token_count(
                    student_features["tokens"], teacher_features["tokens"]
                )
                s_tokens = adapter(s_tokens)
            else:
                student_logits = student_out
                s_tokens = None
                t_tokens = None
            loss, parts = criterion(student_logits, teacher_logits, labels, s_tokens, t_tokens)
        scaler.scale(loss).backward()
        if grad_clip_norm:
            scaler.unscale_(optimizer)
            parameters = list(student.parameters())
            if use_token_distillation:
                parameters += list(adapter.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        acc1 = accuracy(student_logits, labels, topk=(1,))[0]
        meters["loss"].update(float(loss.detach()), images.size(0))
        meters["acc1"].update(float(acc1), images.size(0))
        for key, value in parts.items():
            meters[key].update(value, images.size(0))
        progress.set_postfix(
            loss=f"{meters['loss'].avg:.4f}",
            acc1=f"{meters['acc1'].avg:.2f}",
            kd=f"{meters['kd'].avg:.4f}",
        )
    return {key: meter.avg for key, meter in meters.items()}


def _raise_with_oom_hint(exc: torch.OutOfMemoryError, cfg: dict[str, Any]) -> None:
    batch_size = cfg["dataset"].get("batch_size", 64)
    image_size = cfg["dataset"].get("image_size", 224)
    raise RuntimeError(
        "CUDA ran out of memory. For this pure PyTorch Vision-Mamba KD baseline, lower "
        "`dataset.batch_size` first, then `dataset.image_size` only if you are willing to "
        "retrain the teacher with the same image size. Current values: "
        f"batch_size={batch_size}, image_size={image_size}."
    ) from exc


def train_from_config(cfg: dict[str, Any], resume: str | None = None) -> None:
    set_seed(int(cfg.get("seed", 42)))
    torch.set_float32_matmul_precision("high")
    device = resolve_device(cfg.get("device", "auto"))
    output_dir = Path(cfg.get("output_dir", "runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = cfg.get("run_name", output_dir.name)
    log_step(f"Starting run: {run_name}")
    log_step(f"Device: {device}")
    log_step(f"Output dir: {output_dir}")
    log_step(
        "Building dataloaders "
        f"dataset={cfg['dataset']['name']} image_size={cfg['dataset'].get('image_size', 224)} "
        f"batch_size={cfg['dataset'].get('batch_size', 64)}"
    )
    data_start = time.perf_counter()
    train_loader, val_loader = build_dataloaders(cfg["dataset"], logger=log_step)
    log_step(
        f"Data ready: train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"({time.perf_counter() - data_start:.1f}s)"
    )
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp)
    best_acc = 0.0
    start_epoch = 0

    if cfg.get("distillation", {}).get("enabled", False):
        log_step("Building student and teacher for distillation")
        student = build_model(cfg["student"], cfg["dataset"]).to(device)
        teacher = build_model(cfg["teacher"], cfg["dataset"]).to(device)
        _load_optional_checkpoint(teacher, cfg["teacher"], device)
        teacher.requires_grad_(False)
        teacher.eval()
        distill_cfg = {
            key: value for key, value in cfg["distillation"].items() if key != "enabled"
        }
        use_token_distillation = _distillation_uses_tokens(distill_cfg)
        adapter = None
        if use_token_distillation:
            with torch.no_grad():
                log_step("Probing one batch to size the feature adapter")
                sample = next(iter(train_loader))[0][:1].to(device)
                _, sf = student(sample, return_features=True)
                _, tf = teacher(sample, return_features=True)
            adapter = FeatureAdapter(sf["tokens"].shape[-1], tf["tokens"].shape[-1]).to(device)
        criterion = DistillationCriterion(**distill_cfg)
        parameters = list(student.parameters())
        if use_token_distillation:
            parameters += list(adapter.parameters())
        model_for_eval = student
    else:
        log_step(f"Building model: {cfg['model']['name']}")
        student = build_model(cfg["model"], cfg["dataset"]).to(device)
        teacher = None
        adapter = None
        criterion = None
        use_token_distillation = False
        parameters = student.parameters()
        model_for_eval = student

    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.05)),
    )

    if resume:
        log_step(f"Loading resume checkpoint: {resume}")
        state = torch.load(resume, map_location=device)
        model_for_eval.load_state_dict(state["model"])
        if adapter is not None:
            adapter_state = state.get("extra", {}).get("adapter")
            if adapter_state is not None:
                adapter.load_state_dict(adapter_state)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", -1)) + 1
        best_acc = float(state.get("best_acc", 0.0))

    epochs = int(cfg["train"]["epochs"])
    log_step(f"Training for {epochs - start_epoch} epoch(s), amp={amp}")
    for epoch in range(start_epoch, epochs):
        lr = _cosine_scheduler(
            float(cfg["train"]["lr"]),
            float(cfg["train"].get("min_lr", 1e-6)),
            epoch,
            epochs,
            int(cfg["train"].get("warmup_epochs", 0)),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        log_step(f"Epoch {epoch + 1}/{epochs} started lr={lr:.2e}")

        if teacher is None:
            try:
                train_metrics = _train_teacher_epoch(
                    student,
                    train_loader,
                    optimizer,
                    scaler,
                    device,
                    amp,
                    cfg["train"].get("grad_clip_norm"),
                    epoch + 1,
                    epochs,
                )
            except torch.OutOfMemoryError as exc:
                _raise_with_oom_hint(exc, cfg)
        else:
            try:
                train_metrics = _train_kd_epoch(
                    student,
                    teacher,
                    adapter,
                    train_loader,
                    optimizer,
                    criterion,
                    scaler,
                    device,
                    amp,
                    cfg["train"].get("grad_clip_norm"),
                    epoch + 1,
                    epochs,
                    use_token_distillation,
                )
            except torch.OutOfMemoryError as exc:
                _raise_with_oom_hint(exc, cfg)

        log_step(f"Epoch {epoch + 1}/{epochs}: evaluating validation split")
        val_metrics = evaluate_classifier(model_for_eval, val_loader, device)
        is_best = val_metrics["acc1"] > best_acc
        best_acc = max(best_acc, val_metrics["acc1"])
        save_checkpoint(
            output_dir / "last.pt",
            model_for_eval,
            optimizer,
            epoch=epoch,
            best_acc=best_acc,
            extra={"adapter": adapter.state_dict() if adapter is not None else None},
        )
        if is_best:
            save_checkpoint(
                output_dir / "best.pt",
                model_for_eval,
                optimizer,
                epoch=epoch,
                best_acc=best_acc,
                extra={"adapter": adapter.state_dict() if adapter is not None else None},
            )
        log_step(
            f"epoch={epoch + 1}/{epochs} lr={lr:.2e} "
            f"train_loss={train_metrics['loss']:.4f} train_acc1={train_metrics['acc1']:.2f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc1={val_metrics['acc1']:.2f} "
            f"best={best_acc:.2f}"
        )
