from vim_kd.data.evaluation import evaluation_dataset_config


def test_test_split_removes_cifar_validation_fraction_without_mutating_config():
    original = {
        "name": "cifar10",
        "normalization": "imagenet",
        "validation_fraction": 0.1,
    }

    result, label = evaluation_dataset_config(original, "test")

    assert label == "test"
    assert result["validation_fraction"] == 0.0
    assert result["normalization"] == "imagenet"
    assert original["validation_fraction"] == 0.1


def test_config_split_preserves_validation_settings():
    original = {"name": "cifar10", "validation_fraction": 0.1, "split_seed": 42}

    result, label = evaluation_dataset_config(original, "config")

    assert label == "validation"
    assert result == original
    assert result is not original


def test_imagefolder_uses_its_configured_validation_directory():
    result, label = evaluation_dataset_config(
        {"name": "imagefolder", "val_dir": "data/validation"},
        "test",
    )

    assert label == "validation"
    assert result["val_dir"] == "data/validation"
