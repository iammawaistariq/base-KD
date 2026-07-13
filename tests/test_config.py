from vim_kd.config import load_config


def test_load_kd_config():
    cfg = load_config("configs/cifar10_vit_scratch_to_mamba_kd.yaml")
    assert cfg["distillation"]["enabled"] is True
    assert cfg["student"]["name"] == "vision_mamba"
    assert cfg["teacher"]["checkpoint"] == "runs/cifar10_vit_scratch/best.pt"

def test_clean_experiment_configs():
    scratch_teacher = load_config("configs/cifar10_vit_scratch.yaml")
    pretrained_teacher = load_config("configs/cifar10_vit_pretrained_finetune.yaml")
    plain = load_config("configs/cifar10_mamba_scratch.yaml")
    kd = load_config("configs/cifar10_vit_pretrained_to_mamba_kd.yaml")

    assert scratch_teacher["output_dir"] == "runs/cifar10_vit_scratch"
    assert plain["output_dir"] == "runs/cifar10_mamba_scratch"
    assert pretrained_teacher["dataset"]["validation_fraction"] == 0.1
    assert pretrained_teacher["dataset"]["normalization"] == "imagenet"
    assert kd["dataset"]["normalization"] == "imagenet"
    assert kd["teacher"]["pretrained"] is False
    assert kd["teacher"]["checkpoint"] == "runs/cifar10_vit_pretrained_finetuned/best.pt"
    assert kd["distillation"]["enabled"] is True
