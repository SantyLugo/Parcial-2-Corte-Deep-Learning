"""Training-history plots for a single experiment run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history: list[dict], output_path: str | Path, scenario_id: str) -> None:
    # Preparar el eje de épocas y comparar curvas de entrenamiento con validación.
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train"]["loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation"]["loss"] for row in history], label="validation")
    axes[0].set(title="Pérdida", xlabel="Época", ylabel="Cross entropy")
    axes[1].plot(epochs, [row["train"]["accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["validation"]["accuracy"] for row in history], label="validation")
    axes[1].set(title="Accuracy", xlabel="Época", ylabel="Proporción correcta", ylim=(0, 1))
    # Añadir referencias visuales y leyenda para facilitar la lectura.
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"BloodMNIST · escenario {scenario_id}")
    figure.tight_layout()
    # Exportar la figura y cerrar la ventana para liberar memoria.
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
