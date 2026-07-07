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

The included student is written for clarity. Later, you can replace
`MambaMixer` with `mamba-ssm` or VMamba/Vim blocks while keeping the training
and distillation pipeline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

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

Evaluate a checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/cifar10_vit_to_mamba_kd.yaml \
  --checkpoint runs/cifar10_vit_to_mamba_kd/best.pt
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

- Swap the pure PyTorch scan for `mamba-ssm` selective scan kernels.
- Add segmentation heads and dense KD for medical images.
- Add cross-domain KD inspired by the PDF in this folder: alignment networks,
  semantic autoencoders, and graph/relation distillation.
- Add DeiT-style pretrained teachers from `timm`.
- Add Weights & Biases or TensorBoard logging.
