# CIFAR-10 Experiment Plan

This project should use five clean CIFAR-10 experiments.

## Models to train

| Role | Config | Checkpoint |
| --- | --- | --- |
| ViT from scratch | `configs/cifar10_vit_scratch.yaml` | `runs/cifar10_vit_scratch/best.pt` |
| Mamba from scratch | `configs/cifar10_mamba_scratch.yaml` | `runs/cifar10_mamba_scratch/best.pt` |
| Pretrained ViT fine-tuned on CIFAR-10 | `configs/cifar10_vit_pretrained_finetune.yaml` | `runs/cifar10_vit_pretrained_finetuned/best.pt` |
| Mamba distilled from scratch ViT | `configs/cifar10_vit_scratch_to_mamba_kd.yaml` | `runs/cifar10_vit_scratch_to_mamba_kd/best.pt` |
| Mamba distilled from fine-tuned pretrained ViT | `configs/cifar10_vit_pretrained_to_mamba_kd.yaml` | `runs/cifar10_vit_pretrained_to_mamba_kd/best.pt` |

## Recommended command

Run the full sequence:

```bash
python scripts/run_corrected_experiment.py --device cuda
```

Resume without retraining completed stages:

```bash
python scripts/run_corrected_experiment.py --device cuda --skip-existing
```

Compare available checkpoints:

```bash
python scripts/compare_models.py
```

Create final test-set reports:

```bash
python scripts/analyze_cifar10.py --device cuda --output-dir reports/cifar10_experiment_audit --export-failure-images
```
