"""Load and validate the project's experiment configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    # Leer el archivo de configuración conservando correctamente los caracteres en español.
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    # Rechazar configuraciones incompletas antes de iniciar cualquier ejecución.
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    # Comprobar cantidad e identidad de escenarios para evitar ejecuciones ambiguas.
    scenarios = config.get("scenarios", [])
    ids = [item.get("id") for item in scenarios]
    if len(ids) != 5 or len(set(ids)) != 5:
        raise ValueError("La configuración debe definir cinco escenarios con IDs únicos.")
    # Verificar que los valores acordados y la reserva del test sigan vigentes.
    defaults = config["training_defaults"]
    if defaults["max_epochs"] != 40 or defaults["batch_size"] != 128:
        raise ValueError("Los valores acordados son 40 épocas máximas y batch size 128.")
    if config["dataset"]["test_usage"] != "final_evaluation_only":
        raise ValueError("El split test debe reservarse para la evaluación final.")


def get_scenario(config: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    # Buscar únicamente entre los escenarios activos definidos en el JSON.
    for scenario in config["scenarios"]:
        if scenario["id"] == scenario_id:
            return dict(scenario)
    raise KeyError(f"Escenario desconocido: {scenario_id}")
