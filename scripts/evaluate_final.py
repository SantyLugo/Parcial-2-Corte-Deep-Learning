"""Evaluate exactly one selected checkpoint on the held-out official test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

# Resolver la raíz para localizar checkpoint, datos y resultados de salida.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bloodmnist.data import load_test, make_loader  # noqa: E402
from bloodmnist.engine import run_epoch  # noqa: E402
from bloodmnist.models import BloodCNN  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Recibir el checkpoint escogido con validation; test no elige el modelo.
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    # Recuperar pesos y configuración guardados durante entrenamiento.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    scenario_id = checkpoint["scenario_id"]
    output_path = ROOT / "outputs" / "final_evaluation" / "test_metrics.json"
    # Bloquear una segunda evaluación para mantener test como evaluación única.
    if output_path.exists():
        raise FileExistsError(
            f"Ya existe {output_path}; el protocolo permite una sola evaluación final del test."
        )

    model_config = checkpoint["model_config"]
    model = BloodCNN(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Cargar el split test solo después de elegir el checkpoint final.
    test_data = load_test(ROOT / "data", download=True)
    loader = make_loader(test_data, batch_size=128, shuffle=False)
    # Calcular una vez las métricas del checkpoint sobre el split test oficial.
    metrics = run_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        checkpoint["class_names"],
    )
    result = {
        "scenario_id": scenario_id,
        "checkpoint": str(checkpoint_path),
        "test_samples": len(test_data),
        "metrics": metrics,
        "test_evaluation_count": 1,
    }
    # Guardar métricas y matriz de confusión que leerá la sección de informe.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluación final guardada en {output_path}")


if __name__ == "__main__":
    main()
