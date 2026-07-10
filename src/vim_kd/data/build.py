from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR_MEAN_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}
CIFAR_STORAGE = {
    "cifar10": ("cifar-10-batches-py", "cifar-10-python.tar.gz"),
    "cifar100": ("cifar-100-python", "cifar-100-python.tar.gz"),
}
CIFAR_IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".ppm", ".webp"}
Logger = Callable[[str], None]


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _require_torchvision():
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment.
        raise ModuleNotFoundError(
            "torchvision is required for dataset loading. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return datasets, transforms


def _require_dataloader():
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment.
        raise ModuleNotFoundError(
            "torch is required for dataloaders. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return DataLoader


def _as_path(value: str | Path | None, field: str) -> Path:
    if value is None:
        raise ValueError(f"dataset.{field} is required.")
    return Path(value).expanduser()


class FlatCifarImageDataset:
    """Read CIFAR images named like `1234_frog.png` from a flat split folder."""

    def __init__(self, split_dir: Path, labels_path: Path, transform=None) -> None:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment.
            raise ModuleNotFoundError(
                "Pillow is required for flat CIFAR image folders. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        self.image_cls = Image
        self.split_dir = split_dir
        self.transform = transform
        self.classes = [
            line.strip()
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = self._find_samples(split_dir)
        if not self.samples:
            raise FileNotFoundError(f"No CIFAR images found in {split_dir}")

    def _find_samples(self, split_dir: Path) -> list[tuple[Path, int]]:
        samples = []
        for path in sorted(split_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in CIFAR_IMAGE_EXTENSIONS:
                continue
            label = path.stem.split("_", 1)[-1]
            if label not in self.class_to_idx:
                raise ValueError(
                    f"Could not map label {label!r} from file {path.name}. "
                    f"Expected one of: {', '.join(self.classes)}"
                )
            samples.append((path, self.class_to_idx[label]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.image_cls.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _build_transforms(dataset_name: str, image_size: int, train: bool):
    _, transforms = _require_torchvision()
    mean, std = CIFAR_MEAN_STD.get(dataset_name, (IMAGENET_MEAN, IMAGENET_STD))

    if train:
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
        ]
        # Apply stronger augmentation for CIFAR datasets (RandAugment)
        if dataset_name in {"cifar10", "cifar100"}:
            try:
                ops.insert(0, transforms.RandAugment(num_ops=2, magnitude=9))
            except Exception:
                # Fall back silently if torchvision version lacks RandAugment
                pass
        ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(ops)

    resize_size = max(image_size, int(round(image_size / 0.875)))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _build_flat_cifar_images(cfg: dict[str, Any], name: str, root: Path, logger: Logger | None):
    image_size = int(cfg.get("image_size", 224))
    dataset_dir = Path(cfg.get("flat_dir", root / "cifar")).expanduser()
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "test"
    labels_path = dataset_dir / "labels.txt"
    if not train_dir.is_dir() or not val_dir.is_dir() or not labels_path.is_file():
        return None

    _log(logger, f"data: found flat CIFAR image dataset folder={dataset_dir}")
    start = time.perf_counter()
    train_dataset = FlatCifarImageDataset(
        train_dir,
        labels_path,
        transform=_build_transforms(name, image_size, train=True),
    )
    _log(
        logger,
        f"data: flat train split ready samples={len(train_dataset)} "
        f"classes={len(train_dataset.classes)} ({time.perf_counter() - start:.1f}s)",
    )

    start = time.perf_counter()
    val_dataset = FlatCifarImageDataset(
        val_dir,
        labels_path,
        transform=_build_transforms(name, image_size, train=False),
    )
    _log(
        logger,
        f"data: flat test split ready samples={len(val_dataset)} "
        f"({time.perf_counter() - start:.1f}s)",
    )
    return train_dataset, val_dataset


def _build_cifar(cfg: dict[str, Any], name: str, logger: Logger | None):
    datasets, _ = _require_torchvision()
    root = _as_path(cfg.get("root", "data"), "root")
    image_size = int(cfg.get("image_size", 224))
    download = bool(cfg.get("download", True))
    dataset_cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    base_folder, archive_name = CIFAR_STORAGE[name]
    extracted_dir = root / base_folder
    archive_path = root / archive_name

    _log(
        logger,
        f"data: preparing {name} root={root} download={download} image_size={image_size}",
    )
    flat_datasets = _build_flat_cifar_images(cfg, name, root, logger)
    if flat_datasets is not None:
        return flat_datasets

    if extracted_dir.is_dir():
        _log(logger, f"data: found extracted {name} folder={extracted_dir}")
    elif archive_path.is_file():
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        _log(
            logger,
            f"data: found archive={archive_path} size={size_mb:.1f}MB; checking/extracting",
        )
    elif download:
        _log(
            logger,
            f"data: {name} not found locally; torchvision will download archive={archive_name}",
        )
    else:
        _log(logger, f"data: {name} not found locally and download=False")

    start = time.perf_counter()
    train_dataset = dataset_cls(
        root=str(root),
        train=True,
        transform=_build_transforms(name, image_size, train=True),
        download=download,
    )
    _log(
        logger,
        f"data: train split ready samples={len(train_dataset)} "
        f"({time.perf_counter() - start:.1f}s)",
    )

    start = time.perf_counter()
    val_dataset = dataset_cls(
        root=str(root),
        train=False,
        transform=_build_transforms(name, image_size, train=False),
        download=download,
    )
    _log(
        logger,
        f"data: val split ready samples={len(val_dataset)} "
        f"({time.perf_counter() - start:.1f}s)",
    )
    return train_dataset, val_dataset


def _build_imagefolder(cfg: dict[str, Any], logger: Logger | None):
    datasets, _ = _require_torchvision()
    image_size = int(cfg.get("image_size", 224))
    train_dir = _as_path(cfg.get("train_dir"), "train_dir")
    val_dir = _as_path(cfg.get("val_dir"), "val_dir")
    _log(logger, f"data: scanning ImageFolder train_dir={train_dir}")
    if not train_dir.is_dir():
        raise FileNotFoundError(f"ImageFolder train_dir does not exist: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"ImageFolder val_dir does not exist: {val_dir}")

    start = time.perf_counter()
    train_dataset = datasets.ImageFolder(
        str(train_dir),
        transform=_build_transforms("imagefolder", image_size, train=True),
    )
    _log(
        logger,
        f"data: train ImageFolder ready classes={len(train_dataset.classes)} "
        f"samples={len(train_dataset)} ({time.perf_counter() - start:.1f}s)",
    )

    start = time.perf_counter()
    _log(logger, f"data: scanning ImageFolder val_dir={val_dir}")
    val_dataset = datasets.ImageFolder(
        str(val_dir),
        transform=_build_transforms("imagefolder", image_size, train=False),
    )
    _log(
        logger,
        f"data: val ImageFolder ready samples={len(val_dataset)} "
        f"({time.perf_counter() - start:.1f}s)",
    )

    expected_classes = cfg.get("num_classes")
    if expected_classes is not None and int(expected_classes) != len(train_dataset.classes):
        raise ValueError(
            "dataset.num_classes does not match ImageFolder classes: "
            f"configured={expected_classes}, found={len(train_dataset.classes)}"
        )
    return train_dataset, val_dataset


def _worker_count(cfg: dict[str, Any]) -> int:
    return max(0, int(cfg.get("num_workers", 4)))


def build_dataloaders(
    cfg: dict[str, Any],
    logger: Logger | None = None,
) -> tuple["DataLoader", "DataLoader"]:
    """Build realistic train/validation dataloaders from the YAML dataset section."""

    DataLoader = _require_dataloader()
    name = str(cfg.get("name", "")).lower()
    if name in {"cifar10", "cifar100"}:
        train_dataset, val_dataset = _build_cifar(cfg, name, logger)
    elif name == "imagefolder":
        train_dataset, val_dataset = _build_imagefolder(cfg, logger)
    else:
        raise ValueError(
            f"Unsupported dataset.name: {name!r}. Use cifar10, cifar100, or imagefolder."
        )

    batch_size = int(cfg.get("batch_size", 64))
    num_workers = _worker_count(cfg)
    persistent_workers = num_workers > 0
    pin_memory = bool(cfg.get("pin_memory", True))

    _log(
        logger,
        "data: creating loaders "
        f"batch_size={batch_size} eval_batch_size={int(cfg.get('eval_batch_size', batch_size))} "
        f"num_workers={num_workers} pin_memory={pin_memory} "
        f"drop_last={bool(cfg.get('drop_last', True))}",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=bool(cfg.get("drop_last", True)),
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.get("eval_batch_size", batch_size)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )
    _log(
        logger,
        f"data: loaders ready train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)}",
    )
    return train_loader, val_loader
