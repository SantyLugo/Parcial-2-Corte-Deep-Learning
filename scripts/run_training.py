"""Train one approved BloodMNIST scenario or all five sequentially."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolver la raíz para que las rutas funcionen desde cualquier terminal.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bloodmnist.config import load_config  # noqa: E402
from bloodmnist.train import run_all, run_scenario  # noqa: E402


def main() -> None:
    # Usar la configuración activa para validar los IDs disponibles.
    config = load_config()
    scenario_ids = [item["id"] for item in config["scenarios"]]
    parser = argparse.ArgumentParser(description=__doc__)
    # Permitir ejecutar un ID o todos los activos, pero no ambas opciones.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=scenario_ids)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    # Delegar el entrenamiento; este archivo solo interpreta opciones y resume.
    results = run_all(ROOT) if args.all else [run_scenario(args.scenario, ROOT)]
    print("\nResumen de validación:")
    # Mostrar métricas de validation, usadas para comparar sin consultar test.
    for result in results:
        metrics = result["validation_at_best_checkpoint"]
        print(
            f"{result['scenario']['id']}: best_epoch={result['best_epoch']}, "
            f"val_loss={result['best_validation_loss']:.4f}, "
            f"val_accuracy={metrics['accuracy']:.4f}, val_macro_f1={metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
