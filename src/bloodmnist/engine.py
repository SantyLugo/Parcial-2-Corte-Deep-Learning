"""Training and validation engine with best-validation-loss checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from bloodmnist.metrics import summarize
from bloodmnist.plots import plot_history


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    # La presencia del optimizador distingue entrenamiento de evaluación.
    training = optimizer is not None
    model.train(training)
    class_count = len(class_names)
    confusion = torch.zeros(class_count, class_count, dtype=torch.long)
    total_loss = 0.0
    sample_count = 0
    # Activar gradientes solo al entrenar; evaluación no necesita almacenarlos.
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        # Procesar el split por lotes para limitar el uso de memoria.
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.long).view(-1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # Calcular predicciones y pérdida; actualizar pesos únicamente en train.
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            predicted = logits.argmax(dim=1)
            total_loss += float(loss.detach()) * labels.numel()
            sample_count += labels.numel()
            # Codificar pares real/predicho para acumular la matriz de confusión del lote.
            flat = labels.detach().cpu() * class_count + predicted.detach().cpu()
            confusion += torch.bincount(flat, minlength=class_count**2).reshape(class_count, class_count)
    return summarize(confusion, total_loss, sample_count, class_names)


def fit_scenario(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    defaults: dict,
    scenario: dict,
    class_names: list[str],
    output_dir: str | Path,
    run_metadata: dict,
    device: torch.device,
) -> dict:
    """Fit one scenario, save every best-validation-loss checkpoint, and summarize."""
    # Crear la carpeta donde se guardarán checkpoint, historial y resumen.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenario_id = scenario["id"]
    checkpoint_path = output / f"{scenario_id}_best.pt"
    history_path = output / "history.json"
    summary_path = output / "summary.json"
    criterion = nn.CrossEntropyLoss()
    # Mantener el mismo optimizador y regularización; solo cambia el learning rate por escenario.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scenario["learning_rate"],
        weight_decay=defaults["weight_decay"],
    )
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []
    start_all = time.perf_counter()

    # Entrenar una época y medir validation para decidir si mejora el checkpoint.
    for epoch in range(1, defaults["max_epochs"] + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, class_names, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, class_names)
        improved = val_metrics["loss"] < best_val_loss
        # Guardar el estado con menor pérdida de validación; nunca se selecciona con test.
        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "scenario_id": scenario_id,
                    "scenario": scenario,
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "class_names": class_names,
                    "model_config": {
                        "batch_norm": scenario["batch_norm"],
                        "dropout": scenario["dropout"],
                        "num_classes": len(class_names),
                    },
                    "run_metadata": run_metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "epoch_seconds": time.perf_counter() - start,
            "checkpoint_saved": improved,
        }
        history.append(row)
        print(
            f"{scenario_id} epoch {epoch:02d}/{defaults['max_epochs']} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macro-F1={val_metrics['macro_f1']:.4f} | "
            f"{'best checkpoint' if improved else 'no improvement'}",
            flush=True,
        )
        # Aplicar parada anticipada solo si el escenario la tiene configurada.
        patience = scenario["early_stopping_patience"]
        if patience is not None and epochs_without_improvement >= patience:
            print(f"{scenario_id} early stopping: {patience} épocas sin mejora.", flush=True)
            break

    if best_epoch == 0 or not checkpoint_path.is_file():
        raise RuntimeError(f"No se generó checkpoint para el escenario {scenario_id}.")

    # Restaurar el mejor checkpoint antes de calcular las métricas finales del escenario.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_eval = run_epoch(model, train_loader, criterion, device, class_names)
    val_eval = run_epoch(model, val_loader, criterion, device, class_names)
    # Persistir historial y figura para comparar escenarios sin repetir el entrenamiento.
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_history(history, output / "training_curves.png", scenario_id)
    summary = {
        "scenario": scenario,
        "run_metadata": run_metadata,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "stopped_early": len(history) < defaults["max_epochs"],
        "train_at_best_checkpoint": train_eval,
        "validation_at_best_checkpoint": val_eval,
        "generalization_gap_accuracy": train_eval["accuracy"] - val_eval["accuracy"],
        "elapsed_seconds": time.perf_counter() - start_all,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "curve": str(output / "training_curves.png"),
        "test_used": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
