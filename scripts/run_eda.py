"""Download/load official data splits and write EDA artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

# Resolver la raíz del proyecto desde la ubicación del script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bloodmnist.config import load_config  # noqa: E402
from bloodmnist.data import load_train_val  # noqa: E402
from bloodmnist.eda import build_eda  # noqa: E402


def main() -> None:
    # Cargar configuración y splits de desarrollo para preparar el EDA.
    config = load_config()
    train, val = load_train_val(ROOT / "data", download=config["dataset"]["download"])
    class_names = dict(train.info["label"])
    # Guardar los resúmenes y figuras que luego presenta el notebook.
    build_eda(train, val, class_names, ROOT / "outputs" / "eda")
    print(f"EDA guardada en: {ROOT / 'outputs' / 'eda'}")


if __name__ == "__main__":
    main()
