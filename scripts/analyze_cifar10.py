#!/usr/bin/env python
"""Audit all CIFAR-10 classifiers on the untouched test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vim_kd.config import load_config  # noqa: E402
from vim_kd.audit import build_cross_model_rows  # noqa: E402
from vim_kd.data.build import build_dataloaders  # noqa: E402
from vim_kd.models.factory import build_model  # noqa: E402

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    config: str
    section: str
    checkpoint: str | None
    note: str
    valid_comparison: bool = True


MODEL_SPECS = {
    "vit_scratch": ModelSpec(
        "vit_scratch",
        "Scratch-trained compact ViT",
        "configs/cifar10_vit_scratch.yaml",
        "model",
        "runs/cifar10_vit_scratch/best.pt",
        "Compact ViT trained directly on CIFAR-10.",
    ),
    "vit_pretrained_finetuned": ModelSpec(
        "vit_pretrained_finetuned",
        "Fine-tuned ViT-B/16 teacher",
        "configs/cifar10_vit_pretrained_finetune.yaml",
        "model",
        "runs/cifar10_vit_pretrained_finetuned/best.pt",
        "ViT-B/16 fine-tuned on CIFAR-10 with a trained classification head.",
    ),
    "mamba_kd_from_scratch_vit": ModelSpec(
        "mamba_kd_from_scratch_vit",
        "Mamba distilled from scratch ViT",
        "configs/cifar10_vit_scratch_to_mamba_kd.yaml",
        "student",
        "runs/cifar10_vit_scratch_to_mamba_kd/best.pt",
        "Student distilled from the saved scratch-trained compact ViT.",
    ),
    "mamba_scratch": ModelSpec(
        "mamba_scratch",
        "Scratch-trained Mamba",
        "configs/cifar10_mamba_scratch.yaml",
        "model",
        "runs/cifar10_mamba_scratch/best.pt",
        "Vision Mamba trained directly on CIFAR-10.",
    ),
    "mamba_kd_from_pretrained_vit": ModelSpec(
        "mamba_kd_from_pretrained_vit",
        "Mamba distilled from fine-tuned teacher",
        "configs/cifar10_vit_pretrained_to_mamba_kd.yaml",
        "student",
        "runs/cifar10_vit_pretrained_to_mamba_kd/best.pt",
        "Student distilled from the saved, fine-tuned ViT-B/16 teacher.",
    ),
}

DEFAULT_MODELS = [
    "vit_scratch",
    "mamba_scratch",
    "vit_pretrained_finetuned",
    "mamba_kd_from_scratch_vit",
    "mamba_kd_from_pretrained_vit",
]

TEACHER_STUDENT_PAIRS = {
    "scratch_pair": ("vit_scratch", "mamba_kd_from_scratch_vit"),
    "pretrained_pair": (
        "vit_pretrained_finetuned",
        "mamba_kd_from_pretrained_vit",
    ),
}


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, max_samples: int | None = None) -> None:
        self.dataset = dataset
        self.length = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every model on unseen CIFAR-10 test images and export audit reports."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODEL_SPECS),
        help="Model to audit; repeat for multiple models. Defaults to all.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="reports/cifar10_audit")
    parser.add_argument("--max-samples", type=int, help="Optional smoke-test limit; omit for all 10,000 images.")
    parser.add_argument(
        "--teacher-checkpoint",
        help="Optional fine-tuned timm teacher checkpoint. Makes its audit a valid comparison.",
    )
    parser.add_argument(
        "--export-failure-images",
        action="store_true",
        help="Save original 32x32 CIFAR images for every misclassification.",
    )
    return parser.parse_args()


def resolve_compatible_device(requested: str) -> tuple[torch.device, str]:
    if requested == "cpu":
        return torch.device("cpu"), "CPU selected."
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cpu"), "CUDA unavailable; using CPU."

    major, minor = torch.cuda.get_device_capability()
    required = f"sm_{major}{minor}"
    compiled = set(torch.cuda.get_arch_list())
    gpu = torch.cuda.get_device_name()
    if required not in compiled:
        message = (
            f"{gpu} requires {required}, but PyTorch {torch.__version__} supports "
            f"{', '.join(sorted(compiled)) or 'no CUDA architectures'}."
        )
        if requested == "cuda":
            raise RuntimeError(message)
        return torch.device("cpu"), message + " Using CPU."
    return torch.device("cuda"), f"Using {gpu} ({required})."


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a model state dictionary: {path}")
    return state


def build_audit_model(
    spec: ModelSpec,
    device: torch.device,
    teacher_checkpoint: str | None,
) -> tuple[torch.nn.Module, dict[str, Any], str | None, bool, str]:
    cfg = load_config(ROOT / spec.config)
    model_cfg = dict(cfg[spec.section])
    checkpoint_value = teacher_checkpoint if spec.key == "timm_teacher" else spec.checkpoint
    state = None

    if checkpoint_value:
        checkpoint = Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = load_checkpoint_state(checkpoint)
        if (
            model_cfg.get("name", "").lower() == "vision_mamba"
            and any(".mixer.A_log" in key for key in state)
        ):
            model_cfg["mamba_backend"] = "mamba_reference"
        checkpoint_display = str(checkpoint)
    else:
        checkpoint_display = None

    # Make the randomly initialized timm classification head reproducible.
    torch.manual_seed(0)
    model = build_model(model_cfg, cfg["dataset"])
    if state is not None:
        model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    valid = spec.valid_comparison or bool(teacher_checkpoint)
    note = spec.note
    if teacher_checkpoint:
        note = f"Fine-tuned teacher loaded from {teacher_checkpoint}."
    return model, cfg["dataset"], checkpoint_display, valid, note


@torch.inference_mode()
def collect_predictions(model, loader, device: torch.device):
    indices: list[int] = []
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    top5_hits: list[bool] = []
    total_loss = 0.0

    for images, labels, sample_indices in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        probabilities = logits.softmax(dim=1)
        predicted_confidence, predicted = probabilities.max(dim=1)
        top5 = probabilities.topk(min(5, probabilities.shape[1]), dim=1).indices
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))

        indices.extend(sample_indices.tolist())
        targets.extend(labels.cpu().tolist())
        predictions.extend(predicted.cpu().tolist())
        confidences.extend(predicted_confidence.float().cpu().tolist())
        top5_hits.extend(top5.eq(labels.unsqueeze(1)).any(dim=1).cpu().tolist())

    return {
        "indices": indices,
        "targets": targets,
        "predictions": predictions,
        "confidences": confidences,
        "top5_hits": top5_hits,
        "loss": total_loss / max(1, len(targets)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    rows = []
    for class_name, values in zip(CLASS_NAMES, matrix):
        row = {"true_class": class_name}
        row.update({name: value for name, value in zip(CLASS_NAMES, values)})
        rows.append(row)
    write_csv(path, rows, ["true_class", *CLASS_NAMES])


def save_confusion_plot(path: Path, matrix: np.ndarray, title: str, normalized: bool) -> None:
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = f"{value:.2f}" if normalized else str(int(value))
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def per_class_rows(matrix: np.ndarray, report: dict[str, Any]) -> list[dict[str, Any]]:
    total = int(matrix.sum())
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = total - tp - fn - fp
        metrics = report[name]
        rows.append(
            {
                "class": name,
                "precision": metrics["precision"],
                "recall_sensitivity": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": int(metrics["support"]),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
            }
        )
    return rows


def source_reference(dataset, index: int) -> tuple[str, int | str]:
    if hasattr(dataset, "samples"):
        return str(Path(dataset.samples[index][0]).resolve()), ""
    root = Path(getattr(dataset, "root", "data"))
    base_folder = getattr(dataset, "base_folder", "cifar-10-batches-py")
    return str((root / base_folder / "test_batch").resolve()), index


def export_failure_image(dataset, output_dir: Path, index: int, true_name: str, pred_name: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{index:05d}_true-{true_name}_pred-{pred_name}.png"
    destination = output_dir / filename
    if hasattr(dataset, "data"):
        image = Image.fromarray(dataset.data[index])
    elif hasattr(dataset, "samples"):
        image = Image.open(dataset.samples[index][0]).convert("RGB")
    else:
        raise TypeError("Cannot export an image from this dataset type.")
    image.save(destination)
    return str(destination.resolve())


def audit_model(
    spec: ModelSpec,
    loader: DataLoader,
    raw_dataset,
    device: torch.device,
    output_root: Path,
    teacher_checkpoint: str | None,
    export_images: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print()
    print(f"[{spec.key}] {spec.label}")
    start = time.perf_counter()
    model, dataset_cfg, checkpoint, valid, note = build_audit_model(
        spec, device, teacher_checkpoint
    )
    if str(dataset_cfg["name"]).lower() != "cifar10":
        raise ValueError(f"{spec.key} is not configured for CIFAR-10.")

    result = collect_predictions(model, loader, device)
    y_true = np.asarray(result["targets"], dtype=np.int64)
    y_pred = np.asarray(result["predictions"], dtype=np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(10),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    model_dir = output_root / spec.key
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    failure_rows = []
    failure_image_dir = model_dir / "failure_images"

    for index, true_index, pred_index, confidence, top5_hit in zip(
        result["indices"],
        result["targets"],
        result["predictions"],
        result["confidences"],
        result["top5_hits"],
    ):
        true_name = CLASS_NAMES[true_index]
        pred_name = CLASS_NAMES[pred_index]
        correct = true_index == pred_index
        source_path, source_record_index = source_reference(raw_dataset, index)
        saved_image = (
            export_failure_image(
                raw_dataset, failure_image_dir, index, true_name, pred_name
            )
            if export_images and not correct
            else ""
        )
        row = {
            "test_index": index,
            "image_id": f"cifar10_test_{index:05d}",
            "source_path": source_path,
            "source_record_index": source_record_index,
            "true_class": true_name,
            "predicted_class": pred_name,
            "confidence": confidence,
            "top1_correct": correct,
            "top5_correct": bool(top5_hit),
            "saved_image": saved_image,
            "error_type": (
                "correct"
                if correct
                else f"false_negative_for_{true_name};false_positive_for_{pred_name}"
            ),
        }
        prediction_rows.append(row)
        if not correct:
            failure_rows.append(dict(row))

    write_csv(model_dir / "predictions_all.csv", prediction_rows)
    write_csv(
        model_dir / "failures.csv",
        failure_rows,
        list(prediction_rows[0]),
    )
    write_matrix_csv(model_dir / "confusion_matrix_counts.csv", matrix)
    write_matrix_csv(model_dir / "confusion_matrix_normalized.csv", normalized)
    write_csv(model_dir / "per_class_metrics.csv", per_class_rows(matrix, report))
    save_confusion_plot(
        model_dir / "confusion_matrix_counts.png",
        matrix,
        f"{spec.label} - confusion matrix (counts)",
        normalized=False,
    )
    save_confusion_plot(
        model_dir / "confusion_matrix_normalized.png",
        normalized,
        f"{spec.label} - confusion matrix (row normalized)",
        normalized=True,
    )

    count = len(y_true)
    summary = {
        "model": spec.key,
        "label": spec.label,
        "checkpoint": checkpoint or "",
        "valid_comparison": valid,
        "warning": "" if valid else note,
        "test_images": count,
        "correct": int((y_true == y_pred).sum()),
        "failures": int((y_true != y_pred).sum()),
        "accuracy_top1": float((y_true == y_pred).mean()),
        "accuracy_top5": float(np.mean(result["top5_hits"])),
        "cross_entropy_loss": result["loss"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - start,
        "note": note,
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"  top1={summary['accuracy_top1']:.2%} "
        f"top5={summary['accuracy_top5']:.2%} "
        f"failures={summary['failures']}/{count}"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, prediction_rows


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    torch.set_float32_matmul_precision("high")
    device, device_message = resolve_compatible_device(args.device)
    print(device_message)

    selected = args.model or DEFAULT_MODELS
    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    predictions_by_model = {}
    errors = []
    audited_images = None
    for key in selected:
        try:
            spec = MODEL_SPECS[key]
            data_cfg = dict(load_config(ROOT / spec.config)["dataset"])
            data_cfg.update(
                {
                    "batch_size": args.batch_size,
                    "eval_batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "download": True,
                    "validation_fraction": 0.0,
                }
            )
            _, base_test_loader = build_dataloaders(data_cfg, logger=print)
            raw_dataset = base_test_loader.dataset
            loader = DataLoader(
                IndexedDataset(raw_dataset, args.max_samples),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
            audited_images = len(loader.dataset)
            summary, prediction_rows = audit_model(
                spec,
                loader,
                raw_dataset,
                device,
                output_root,
                args.teacher_checkpoint,
                args.export_failure_images,
            )
            summaries.append(summary)
            predictions_by_model[key] = prediction_rows
        except Exception as exc:
            errors.append({"model": key, "error": str(exc)})
            print(f"  ERROR: {exc}")

    if summaries:
        write_csv(output_root / "model_summary.csv", summaries)
    combined_rows, transition_rows = build_cross_model_rows(
        predictions_by_model,
        TEACHER_STUDENT_PAIRS,
    )
    if combined_rows:
        write_csv(output_root / "sample_comparison.csv", combined_rows)
        write_csv(
            output_root / "failed_by_three_or_more.csv",
            [row for row in combined_rows if row["failed_three_or_more"]],
            list(combined_rows[0]),
        )
        write_csv(
            output_root / "failed_by_all_models.csv",
            [row for row in combined_rows if row["failed_all_models"]],
            list(combined_rows[0]),
        )
        write_csv(
            output_root / "teacher_student_transitions.csv",
            transition_rows,
            list(combined_rows[0]),
        )
    metadata = {
        "dataset": "CIFAR-10 official test split",
        "test_images": audited_images,
        "classes": CLASS_NAMES,
        "device": str(device),
        "selected_models": selected,
        "teacher_student_pairs": TEACHER_STUDENT_PAIRS,
        "errors": errors,
        "teacher_warning": (
        "Legacy random-head pretrained-teacher runs are stored under runs/legacy/ "
        "and are excluded from the standard audit."
        ),
        "error_semantics": (
            "For a multiclass mistake true=A, predicted=B: the image is a false "
            "negative for A and a false positive for B."
        ),
    }
    (output_root / "audit_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print()
    print(f"Reports written to: {output_root}")
    if errors:
        print(f"{len(errors)} model(s) failed; see audit_manifest.json.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
