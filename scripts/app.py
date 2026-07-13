#!/usr/bin/env python
"""Streamlit UI for comparing the classifiers trained by this repository."""

# streamlit run .\scripts\app.py

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from PIL import Image

# Allow `streamlit run scripts/app.py` to work before an editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vim_kd.config import load_config  # noqa: E402
from vim_kd.data.build import _build_transforms  # noqa: E402
from vim_kd.models.factory import build_model  # noqa: E402


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

MODEL_SPECS = {
    "Fine-tuned ViT-B/16 teacher": {
        "config": "configs/cifar10_timm_teacher_finetune.yaml",
        "section": "model",
        "checkpoint": "runs/cifar10_timm_teacher_finetuned/best.pt",
        "note": "ViT-B/16 with a CIFAR-10-trained classification head.",
    },
    "Trained compact ViT": {
        "config": "configs/cifar10_teacher_vit.yaml",
        "section": "model",
        "checkpoint": "runs/cifar10_teacher_vit/best.pt",
        "note": "Compact ViT trained directly on CIFAR-10 labels.",
    },
    "Corrected trained Mamba": {
        "config": "configs/cifar10_mamba_corrected.yaml",
        "section": "model",
        "checkpoint": "runs/cifar10_mamba_corrected/best.pt",
        "note": "Fair plain baseline using the corrected split and normalization.",
    },
    "Corrected distilled Mamba": {
        "config": "configs/cifar10_timm_teacher_to_mamba_corrected_kd.yaml",
        "section": "student",
        "checkpoint": "runs/cifar10_timm_teacher_to_mamba_corrected_kd/best.pt",
        "note": "Mamba distilled from the saved, fine-tuned ViT-B/16 teacher.",
    },
}


def _absolute(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def _resolve_inference_device(requested: str) -> tuple[str, str]:
    """Select CUDA only when this PyTorch binary contains the GPU kernel architecture."""

    if requested == "cpu":
        return "cpu", "CPU selected manually."
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested, but PyTorch reports that CUDA is unavailable.")
        return "cpu", "CUDA is unavailable; using CPU."

    major, minor = torch.cuda.get_device_capability()
    required_arch = f"sm_{major}{minor}"
    compiled_arches = set(torch.cuda.get_arch_list())
    gpu_name = torch.cuda.get_device_name()

    if required_arch not in compiled_arches:
        supported = ", ".join(sorted(compiled_arches)) or "none"
        message = (
            f"{gpu_name} requires {required_arch}, but PyTorch {torch.__version__} "
            f"was compiled for: {supported}."
        )
        if requested == "cuda":
            raise RuntimeError(message)
        return "cpu", message + " Using CPU to avoid a CUDA kernel crash."

    return "cuda", f"Using {gpu_name} ({required_arch})."


def _class_names(dataset_cfg: dict[str, Any]) -> list[str]:
    name = str(dataset_cfg["name"]).lower()
    if name == "cifar10":
        return list(CIFAR10_CLASSES)
    if name == "cifar100":
        from torchvision.datasets import CIFAR100

        return list(CIFAR100.classes)
    if name == "imagefolder":
        train_dir = _absolute(str(dataset_cfg["train_dir"]))
        names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
        if names:
            return names
    return [f"class_{index}" for index in range(int(dataset_cfg["num_classes"]))]


@st.cache_resource(show_spinner=False)
def _load_model(config_path: str, section: str, checkpoint_path: str, device_name: str):
    cfg = load_config(_absolute(config_path))
    device = torch.device(device_name)
    model_cfg = dict(cfg[section])
    state = None

    if checkpoint_path:
        checkpoint = _absolute(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

        # Optimized mamba-ssm and the educational fallback use different
        # parameters. Reconstruct a checkpoint-compatible reference backend.
        uses_classic_ssm = any(".mixer.A_log" in key for key in state)
        if model_cfg.get("name", "").lower() == "vision_mamba" and uses_classic_ssm:
            model_cfg["mamba_backend"] = "mamba_reference"

    model = build_model(model_cfg, cfg["dataset"]).to(device)
    if state is not None:
        model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg["dataset"], device


def _predict(model, image: Image.Image, dataset_cfg: dict[str, Any], device, top_k: int):
    dataset_name = str(dataset_cfg["name"]).lower()
    image_size = int(dataset_cfg.get("image_size", 224))
    transform = _build_transforms(
        dataset_name,
        image_size,
        train=False,
        normalization=dataset_cfg.get("normalization"),
    )
    batch = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(batch).softmax(dim=-1)[0].float().cpu()
    names = _class_names(dataset_cfg)
    count = min(top_k, probabilities.numel())
    values, indices = probabilities.topk(count)
    return [(names[index], float(value)) for value, index in zip(values, indices)]


def main() -> None:
    st.set_page_config(page_title="ViT to Mamba KD", page_icon="microscope", layout="wide")
    st.title("ViT to Mamba classification comparison")
    st.caption("Upload one image and compare the teacher, compact ViT, plain Mamba, and distilled Mamba.")

    with st.sidebar:
        st.header("Inference settings")
        selected = st.multiselect("Models", list(MODEL_SPECS), default=list(MODEL_SPECS))
        device_request = st.selectbox("Device", ("auto", "cpu", "cuda"))
        top_k = st.slider("Top predictions", 1, 10, 5)
        try:
            device, device_message = _resolve_inference_device(device_request)
            st.caption(device_message)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        st.info("Models are loaded once and cached. Run the corrected experiment pipeline first.")

    uploaded = st.file_uploader("Upload a JPG, JPEG, PNG, BMP, or WebP image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Upload an image to run classification.")
        return

    image = Image.open(uploaded).convert("RGB")
    preview, results = st.columns((1, 2))
    with preview:
        st.image(image, caption=uploaded.name, width="stretch")

    if not selected:
        results.warning("Select at least one model in the sidebar.")
        return

    with results:
        columns = st.columns(min(2, len(selected)))
        for position, name in enumerate(selected):
            spec = MODEL_SPECS[name]
            with columns[position % len(columns)]:
                st.subheader(name)
                try:
                    with st.spinner(f"Running {name}..."):
                        model, dataset_cfg, model_device = _load_model(
                            spec["config"], spec["section"], spec["checkpoint"], device
                        )
                        predictions = _predict(model, image, dataset_cfg, model_device, top_k)
                    label, confidence = predictions[0]
                    st.metric("Prediction", label, f"{confidence:.2%} confidence")
                    st.bar_chart({label: probability for label, probability in predictions})
                    st.caption(spec["note"])
                except Exception as exc:
                    st.error(str(exc))


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
            )
        )
    main()