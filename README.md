# Transformer to Mamba Knowledge Distillation for Vision

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

The student uses `mamba-ssm` automatically when it is installed, and falls back
to a readable pure PyTorch mixer otherwise. The fallback is useful for study, but
the optimized backend is the practical choice for training speed.

## Setup

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

## Quick Start

Train a ViT teacher first:

```bash
python scripts/train.py --config configs/cifar10_teacher_vit.yaml
```

Then distill into the Mamba student:

```bash
python scripts/train.py --config configs/cifar10_vit_to_mamba_kd.yaml
```

Train the same Mamba student without distillation for a fair baseline:

```bash
python scripts/train.py --config configs/cifar10_mamba.yaml
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/cifar10_vit_to_mamba_kd.yaml \
  --checkpoint runs/cifar10_vit_to_mamba_kd/best.pt
```

Compare teacher, plain Mamba, and Mamba-KD:

```bash
python scripts/compare_models.py
```

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
