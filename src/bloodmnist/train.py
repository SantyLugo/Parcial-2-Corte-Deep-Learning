"""Run one or all approved training scenarios; this module never loads test."""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bloodmnist.config import get_scenario, load_config
from bloodmnist.data import load_train_val, make_loader
from bloodmnist.engine import fit_scenario
from bloodmnist.models import BloodCNN


def _seed_everything(seed: int, num_threads: int) -> None:
    # Fijar generadores y algoritmos para que las corridas sean comparables.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def _merge_comparison(root: Path, results: list[dict], active_ids: list[str]) -> None:
    # Leer lo existente para conservar metadatos, incluido B1 como antecedente histórico.
    summary_path = root / "outputs" / "runs" / "comparison.json"
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    by_id = {item["scenario"]["id"]: item for item in existing}
    # Sustituir solo los escenarios que acaban de entrenarse y conservar los demás.
    by_id.update({item["scenario"]["id"]: item for item in results})
    active_order = [scenario_id for scenario_id in active_ids if scenario_id in by_id]
    retired_order = [scenario_id for scenario_id in by_id if scenario_id not in active_ids]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Presentar primero los escenarios activos y al final los registros retirados.
    summary_path.write_text(
        json.dumps([by_id[scenario_id] for scenario_id in active_order + retired_order], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_scenario(scenario_id: str, project_root: str | Path | None = None) -> dict:
    # Resolver configuración, escenario y recursos antes de construir el modelo.
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "experiments.json")
    defaults = config["training_defaults"]
    scenario = get_scenario(config, scenario_id)
    seed = config["reproducibility"]["seed"]
    _seed_everything(seed, defaults["num_threads"])
    device = torch.device(config["reproducibility"]["device"])
    data_start = time.perf_counter()
    # Cargar exclusivamente train y validation; esta ruta nunca abre test.
    train_data, val_data = load_train_val(root / "data", download=config["dataset"]["download"])
    data_seconds = time.perf_counter() - data_start
    class_names_map = train_data.info["label"]
    class_names = [class_names_map[str(index)] for index in range(len(class_names_map))]
    # Mezclar train de forma reproducible y recorrer validation sin mezclar.
    train_loader = make_loader(
        train_data,
        batch_size=defaults["batch_size"],
        shuffle=True,
        seed=seed,
        num_workers=defaults["num_workers"],
    )
    val_loader = make_loader(
        val_data,
        batch_size=defaults["batch_size"],
        shuffle=False,
        num_workers=defaults["num_workers"],
    )
    # Construir la CNN con BatchNorm y Dropout según el escenario seleccionado.
    model = BloodCNN(
        num_classes=len(class_names),
        batch_norm=scenario["batch_norm"],
        dropout=scenario["dropout"],
    ).to(device)
    # Registrar entorno y parámetros para interpretar y reproducir la corrida.
    metadata = {
        "run_name": f"BloodMNIST-{scenario_id}",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "device": str(device),
        "batch_size": defaults["batch_size"],
        "max_epochs": defaults["max_epochs"],
        "optimizer": defaults["optimizer"],
        "learning_rate": scenario["learning_rate"],
        "weight_decay": defaults["weight_decay"],
        "num_workers": defaults["num_workers"],
        "num_threads": defaults["num_threads"],
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "class_names": class_names,
        "dataset": config["dataset"]["name"],
        "input_size": [28, 28],
        "input_channels": 3,
        "image_transform": "ToTensor; uint8 RGB to float32 [0,1]",
        "deterministic_algorithms": config["reproducibility"]["deterministic"],
        "checkpoint_selection": defaults["checkpoint_selection"],
        "resolved_config": config,
        "data_setup_seconds": data_seconds,
    }
    result = fit_scenario(
        model,
        train_loader,
        val_loader,
        defaults,
        scenario,
        class_names,
        root / "outputs" / "runs" / scenario_id,
        metadata,
        device,
    )
    # Actualizar la tabla consolidada sin borrar escenarios retirados.
    _merge_comparison(root, [result], [item["id"] for item in config["scenarios"]])
    return result


def run_all(project_root: str | Path | None = None) -> list[dict]:
    config = load_config(
        (Path(project_root) if project_root else Path(__file__).resolve().parents[2])
        / "configs"
        / "experiments.json"
    )
    # Cada corrida actualiza comparison.json y conserva entradas históricas.
    return [run_scenario(scenario["id"], project_root) for scenario in config["scenarios"]]
