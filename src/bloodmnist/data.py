"""BloodMNIST loading and tensor conversion helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


def _load_split(split: str, root: str | Path, download: bool) -> Dataset:
    """Load one official MedMNIST split with the pilot's RGB tensor transform."""
    # Importar MedMNIST aquí permite mostrar un error claro si falta la dependencia.
    try:
        from medmnist import BloodMNIST
    except ImportError as exc:
        raise RuntimeError("Falta MedMNIST. Instala las dependencias con `pip install -r requirements.txt`.") from exc

    # Asegurar que la carpeta de datos exista antes de solicitar descarga o lectura.
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    return BloodMNIST(
        split=split,
        root=str(root_path),
        download=download,
        size=28,
        as_rgb=True,
        transform=ToTensor(),
    )


def load_train_val(root: str | Path = "data", download: bool = True) -> tuple[Dataset, Dataset]:
    """Load only train and validation; test remains untouched during development."""
    # Cargar solo entrenamiento y validación durante EDA y ajuste del modelo.
    return (
        _load_split("train", root, download),
        _load_split("val", root, download),
    )


def load_test(root: str | Path = "data", download: bool = True) -> Dataset:
    """Explicit final-evaluation entry point for the held-out official test split."""
    # Exponer test por separado para reservarlo a la evaluación final.
    return _load_split("test", root, download)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 image into normalized CHW float tensor."""
    # Convertir la entrada a un arreglo y agregar canal si la imagen es en escala de grises.
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Se esperaba imagen HWC o HW; recibido shape={array.shape}.")
    # PyTorch espera canales primero (CHW), mientras MedMNIST entrega píxeles HWC.
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float32).div_(255.0)
    return tensor.to(torch.float32)


def make_loader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Create a reproducible DataLoader without changing the dataset transform."""
    # Sembrar el generador solo al mezclar datos para reproducir el orden de entrenamiento.
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )
