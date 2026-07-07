from vim_kd.config import load_config


def test_load_kd_config():
    cfg = load_config("configs/cifar10_vit_to_mamba_kd.yaml")
    assert cfg["distillation"]["enabled"] is True
    assert cfg["student"]["name"] == "vision_mamba"
