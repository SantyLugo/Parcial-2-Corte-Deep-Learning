"""Classification metrics derived from a multiclass confusion matrix."""

from __future__ import annotations

import torch


def summarize(confusion: torch.Tensor, loss_sum: float, sample_count: int, class_names: list[str]) -> dict:
    # Evitar métricas inválidas si el cargador no entregó ninguna imagen.
    if sample_count <= 0:
        raise ValueError("No se procesaron muestras.")
    # Derivar precisión, recall y F1 a partir de la matriz de confusión.
    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    # Conservar soporte y métricas individuales para analizar clases minoritarias.
    per_class = {
        class_names[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(len(class_names))
    }
    # Reportar promedios macro para dar el mismo peso a todas las clases.
    return {
        "loss": loss_sum / sample_count,
        "accuracy": float(true_positive.sum() / sample_count),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
