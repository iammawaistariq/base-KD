from vim_kd.config import load_config


def test_load_kd_config():
    cfg = load_config("configs/cifar10_vit_to_mamba_kd.yaml")
    assert cfg["distillation"]["enabled"] is True
    assert cfg["student"]["name"] == "vision_mamba"

def test_corrected_kd_loads_finetuned_teacher_checkpoint():
    teacher = load_config("configs/cifar10_timm_teacher_finetune.yaml")
    plain = load_config("configs/cifar10_mamba_corrected.yaml")
    kd = load_config("configs/cifar10_timm_teacher_to_mamba_corrected_kd.yaml")

    assert teacher["dataset"]["validation_fraction"] == 0.1
    assert teacher["dataset"]["normalization"] == "imagenet"
    assert plain["dataset"]["split_seed"] == teacher["dataset"]["split_seed"]
    assert kd["dataset"]["normalization"] == "imagenet"
    assert kd["teacher"]["pretrained"] is False
    assert kd["teacher"]["checkpoint"] == "runs/cifar10_timm_teacher_finetuned/best.pt"
    assert kd["distillation"]["enabled"] is True
