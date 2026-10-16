"""Exploratory summaries and visualizations using train and validation only."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _label_name(label: int, class_names: dict[str, str]) -> str:
    return class_names.get(str(label), str(label))


def _display_name(name: str) -> str:
    if name.startswith("immature granulocytes"):
        return "immature granulocytes"
    return name


def _summarize(dataset, split: str, class_names: dict[str, str]) -> tuple[dict, Counter]:
    # Contar etiquetas y registrar la forma de la primera imagen del split.
    counts: Counter = Counter()
    image_shape = None
    for index in range(len(dataset)):
        image, target = dataset[index]
        image_shape = tuple(image.shape)
        counts[int(np.asarray(target).reshape(-1)[0])] += 1
    summary = {
        "split": split,
        "samples": sum(counts.values()),
        "image_shape_chw": image_shape,
        "classes_present": len(counts),
        "class_counts": {
            _label_name(label, class_names): count for label, count in sorted(counts.items())
        },
    }
    return summary, counts


def _sample_grid(train, class_names: dict[str, str], output_path: Path, seed: int = 42) -> None:
    """Display two reproducibly selected train examples per class."""
    # Usar una semilla fija para que la cuadrícula muestre siempre los mismos ejemplos.
    rng = np.random.default_rng(seed)
    indices_by_class: dict[int, list[int]] = {}
    # Recorrer train en orden aleatorio reproducible y conservar dos ejemplos por clase.
    for index in rng.permutation(len(train)):
        _, target = train[int(index)]
        label = int(np.asarray(target).reshape(-1)[0])
        picks = indices_by_class.setdefault(label, [])
        if len(picks) < 2:
            picks.append(int(index))
        if len(indices_by_class) == len(class_names) and all(len(items) >= 2 for items in indices_by_class.values()):
            break

    # Crear una cuadrícula uniforme usando únicamente ejemplos de train.
    labels = sorted(indices_by_class)
    figure, axes = plt.subplots(len(labels), 2, figsize=(7, 2.1 * len(labels)), squeeze=False)
    for row, label in enumerate(labels):
        for column, index in enumerate(indices_by_class[label]):
            image, _ = train[index]
            pixels = image.permute(1, 2, 0).cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
            axes[row, column].imshow(np.clip(pixels, 0, 1))
            axes[row, column].axis("off")
            display_name = _display_name(_label_name(label, class_names))
            axes[row, column].set_title(f"{display_name} · ejemplo {column + 1}", fontsize=9)
    figure.suptitle("BloodMNIST · ejemplos del split train", y=1.002)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_eda(train, val, class_names: dict[str, str], output_dir: str | Path) -> pd.DataFrame:
    """Write class counts, split summary, and train-only image examples."""
    # Preparar la carpeta y acumular resúmenes separados para train y validation.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries, rows = [], []
    for name, dataset in (("train", train), ("val", val)):
        summary, counts = _summarize(dataset, name, class_names)
        summaries.append(summary)
        rows.extend(
            {"split": name, "label": label, "class_name": _label_name(label, class_names), "count": n}
            for label, n in sorted(counts.items())
        )

    # Guardar conteos y dimensiones en formatos reutilizables por el informe.
    (output / "dataset_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    counts_df = pd.DataFrame(rows)
    counts_df.to_csv(output / "class_distribution.csv", index=False)
    # Simplificar etiquetas largas solo para mejorar la lectura de la gráfica.
    plot_df = counts_df.assign(display_class=counts_df["class_name"].map(_display_name))
    pivot = plot_df.pivot(index="display_class", columns="split", values="count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(11, 5), rot=25)
    ax.tick_params(axis="x", labelsize=9)
    ax.set(title="BloodMNIST: muestras por clase", xlabel="Clase", ylabel="Número de imágenes")
    ax.legend(title="Split")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "class_distribution.png", dpi=160)
    plt.close(ax.figure)
    # Guardar ejemplos visuales de train; validation y test no se dibujan aquí.
    _sample_grid(train, class_names, output / "train_examples.png")
    return counts_df
