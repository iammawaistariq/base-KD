# Vision Transformer to Mamba Knowledge Distillation for Image Classification

This repository is a practical baseline for learning and extending knowledge
distillation from a Transformer teacher to a Mamba-style vision student.

It is intentionally not a dummy demo:

- Teacher: ViT from `timm` or the included compact ViT.
- Student: pure PyTorch Vision-Mamba style model with patch tokens, selective
  state-space mixing, residual blocks, and classification head.
- Distillation: hard labels, softened logits, feature alignment, and token
  relation distillation.
- Datasets: CIFAR-10/100 and generic `ImageFolder`.
- Training: config files, AMP, checkpoints, metrics, and resume support.
- Dashboard: uploaded-image comparison across ViT and Mamba classifiers.

The student uses `mamba-ssm` automatically when it is installed, and falls back
to a readable pure PyTorch mixer otherwise. The fallback is useful for study, but
the optimized backend is the practical choice for training speed.

## Setup

For a clean, reproducible CUDA 12.8 installation on another matching Linux or
WSL2 computer, follow [the second-PC installation guide](docs/INSTALL_SECOND_PC.md).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For faster Mamba training on a CUDA machine, install the optional optimized
backend after PyTorch is already installed:

```bash
pip install -r requirements.txt
pip install -e .
pip install "mamba-ssm==2.2.6.post3" --no-build-isolation
```

`mamba-ssm` builds against the installed PyTorch package, so installing it before
`torch` is available can fail with `ModuleNotFoundError: No module named
'torch'`.

If the PyPI package fails with `nvcc was not found` or `bare_metal_version is not
defined`, use the latest upstream source install:

```bash
pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation
```

If `from mamba_ssm import Mamba` fails inside `tilelang` or `tvm`, you likely
installed the newer Mamba3 package path. This repo only needs the classic Mamba
block, so remove it and use the pinned 2.2.x package:

```bash
pip uninstall -y mamba-ssm tilelang tvm tvm-ffi tvm_ffi
pip install "mamba-ssm==2.2.6.post3" --no-build-isolation
```

If installation still fails with `nvcc was not found` followed by
`NameError: name 'bare_metal_version' is not defined`, the CUDA compiler is not
available. `nvidia-smi` only proves the driver is installed; building Mamba also
needs `nvcc`:

```bash
which nvcc
nvcc --version
```

Install a CUDA compiler that matches your PyTorch CUDA runtime, or use a PyTorch
environment with a Mamba wheel available for it. For a conda environment using
PyTorch CUDA 13.0, the compiler can usually be added with:

```bash
conda install -c nvidia cuda-nvcc
pip install "mamba-ssm==2.2.6.post3" --no-build-isolation
```

The optimized CUDA extension also needs a visible NVIDIA driver and CUDA build
toolchain. Check these before expecting speedups:

```bash
nvidia-smi
which nvcc
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

If `nvidia-smi` fails or `torch.cuda.is_available()` prints `False`, this project
will fall back to CPU/PyTorch execution and training will be slow regardless of
whether `mamba-ssm` is installed.

The CIFAR configs download the real torchvision datasets into `data/` by
default. For your own data, use the ImageFolder config layout:

```text
dataset_root/
  train/class_a/*.jpg
  train/class_b/*.jpg
  val/class_a/*.jpg
  val/class_b/*.jpg
```

## RTX 50-series / Blackwell setup

RTX 50-series GPUs use compute capability sm_120. They require a PyTorch build
compiled with CUDA 12.8 or newer. If an older cu121 environment is already
installed, replace its PyTorch packages before running this project:

    pip uninstall -y torch torchvision torchaudio
    pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

Verify that sm_120 appears in the compiled architecture list:

    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"

The optimized Mamba extension must then be rebuilt against the new PyTorch/CUDA
installation. The upstream mamba-ssm project officially targets Linux; WSL2 or
Linux is recommended for optimized Mamba on an RTX 50-series GPU.

## Clean CIFAR-10 Experiment Setup

The final comparison should use these five experiments:

- ViT from scratch: `configs/cifar10_vit_scratch.yaml`
- Mamba from scratch: `configs/cifar10_mamba_scratch.yaml`
- Pretrained ViT fine-tuned on CIFAR-10: `configs/cifar10_vit_pretrained_finetune.yaml`
- Mamba distilled from scratch ViT: `configs/cifar10_vit_scratch_to_mamba_kd.yaml`
- Mamba distilled from fine-tuned pretrained ViT: `configs/cifar10_vit_pretrained_to_mamba_kd.yaml`

Run the full ordered pipeline:

```bash
python scripts/run_corrected_experiment.py --device cuda
```

Resume after interruption:

```bash
python scripts/run_corrected_experiment.py --device cuda --skip-existing
```

Compare whichever checkpoints already exist:

```bash
python scripts/compare_models.py
```

Create final CIFAR-10 test reports:

```bash
python scripts/analyze_cifar10.py --device cuda --output-dir reports/cifar10_experiment_audit --export-failure-images
```

See `docs/experiment_plan.md` for the expected configs, checkpoints, and legacy
run warning.

## Uploaded-image dashboard

Install dependencies and launch from the repository root:

    streamlit run scripts/app.py

Upload a JPG, PNG, BMP, or WebP image to compare top-k predictions from the
pretrained ViT-B/16 teacher definition, trained compact ViT, normally trained
Mamba, and distilled Mamba. The app uses checkpoints under runs/, caches models,
selects CUDA automatically, and lets one missing model fail without stopping the
other comparisons.

The compact ViT and timm ViT are separate teacher choices. The supplied KD
config uses pretrained timm ViT-B/16 and does not load the compact ViT
checkpoint. KD training stores only the student, so the dashboard timm teacher
has a new CIFAR-10 head unless you supply a fine-tuned teacher checkpoint in
MODEL_SPECS in scripts/app.py. Treat its output as diagnostic until then. The
default classifiers expect CIFAR-10-like images.

Default checkpoints:

- Scratch compact ViT: runs/cifar10_vit_scratch/best.pt
- Scratch Mamba: runs/cifar10_mamba_scratch/best.pt
- Legacy distilled Mamba diagnostic: runs/legacy/cifar10_timm_random_head_to_mamba_kd/best.pt

## Corrected fine-tuned-teacher experiment

The original timm KD run used a random frozen CIFAR-10 head and is retained only
as a diagnostic legacy result. The corrected experiment uses a deterministic
45,000/5,000 training/validation split, ImageNet normalization for every model,
saved teacher checkpoints, teacher-accuracy gates, and fresh plain/KD Mamba
comparisons. The official 10,000-image test split is used only for final
reporting.

Run the complete pipeline:

    python scripts/run_corrected_experiment.py --device cuda

The default teacher gate requires at least 90% validation accuracy. Existing
successful stages can be reused after interruption:

    python scripts/run_corrected_experiment.py --device cuda --skip-existing

Manual stages are:

    python scripts/train.py --config configs/cifar10_vit_scratch.yaml
    python scripts/train.py --config configs/cifar10_vit_pretrained_finetune.yaml
    python scripts/verify_checkpoint.py --config configs/cifar10_vit_pretrained_finetune.yaml --checkpoint runs/cifar10_vit_pretrained_finetuned/best.pt --section model --split validation --min-accuracy 90 --device cuda
    python scripts/verify_checkpoint.py --config configs/cifar10_vit_pretrained_finetune.yaml --checkpoint runs/cifar10_vit_pretrained_finetuned/best.pt --section model --split test --device cuda
    python scripts/train.py --config configs/cifar10_mamba_scratch.yaml
    python scripts/train.py --config configs/cifar10_vit_scratch_to_mamba_kd.yaml
    python scripts/train.py --config configs/cifar10_vit_pretrained_to_mamba_kd.yaml

After all checkpoints exist, create an isolated corrected audit:

    python scripts/analyze_cifar10.py --model vit_pretrained_finetuned --model mamba_scratch --model mamba_kd_from_pretrained_vit --device cuda --output-dir reports/cifar10_experiment_audit --export-failure-images

Outputs are stored separately in:

- runs/cifar10_vit_pretrained_finetuned/
- runs/cifar10_mamba_scratch/
- runs/cifar10_vit_pretrained_to_mamba_kd/
- reports/cifar10_experiment_audit/

## Full CIFAR-10 test audit

Evaluate all four model entries on the official unseen 10,000-image CIFAR-10 test
split:

    python scripts/analyze_cifar10.py --device auto

Reports are written to reports/cifar10_audit/. Each model receives count and
row-normalized confusion matrices in CSV and PNG form, per-class precision,
recall, F1, TP, FP, FN, and TN counts, a CSV row for every test image, and a
failures-only CSV. Every row includes the stable CIFAR test index, true class,
predicted class, confidence, and error interpretation.

For a mistake whose true class is cat and prediction is dog, the same image is a
false negative for cat and a false positive for dog. To also save the original
32x32 failed images, run:

    python scripts/analyze_cifar10.py --export-failure-images

The default timm teacher has no trained CIFAR-10 head and is flagged as an
invalid comparison. Supply a real fine-tuned teacher checkpoint with
--teacher-checkpoint PATH to obtain meaningful teacher metrics. Use
--max-samples N only for a quick smoke test; omit it for the final thesis report.

## What to Study

1. `src/vim_kd/models/vision_mamba.py`
   The Mamba-style student. Start by tracing tensor shapes through
   `PatchEmbed`, `VisionMambaBlock`, and `MambaMixer`.

2. `src/vim_kd/engine/losses.py`
   The distillation objective:
   `CE + KL(logits/T) + feature alignment + relation matching`.

3. `src/vim_kd/engine/trainer.py`
   How teacher outputs are detached, how feature adapters are learned, and how
   checkpointing/resume works.

## Future Enhancements

- Add VMamba/Vim blocks as additional vision-student options.
- Add segmentation heads and dense KD for medical images.
- Add cross-domain KD inspired by the PDF in this folder: alignment networks,
  semantic autoencoders, and graph/relation distillation.
- Add DeiT-style pretrained teachers from `timm`.
- Add Weights & Biases or TensorBoard logging.
