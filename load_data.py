import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional

import pandas as pd


# Cambia este valor si luego quieres cargar otro kernel/dataset
KAGGLE_KERNEL = "bryanlh27/clasificaci-n-de-precios-de-telefon-a-m-vil"
DATA_DIR = Path("data/kaggle")


def ensure_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "No está instalado el cliente de Kaggle. "
            "Instálalo con: pip install kaggle"
        )


def download_kaggle_kernel(kernel_slug: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "kernels", "pull", kernel_slug, "--path", str(output_dir)]
    print(f"Ejecutando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def find_csv_files(root_dir: Path) -> List[Path]:
    return sorted(root_dir.rglob("*.csv"))


def find_zip_files(root_dir: Path) -> List[Path]:
    return sorted(root_dir.rglob("*.zip"))


def load_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No se encontró ningún CSV dentro del ZIP: {zip_path}")

        csv_name = csv_files[0]
        print(f"Cargando CSV desde ZIP: {csv_name}")
        with zf.open(csv_name) as f:
            return pd.read_csv(f)


def load_data() -> pd.DataFrame:
    ensure_kaggle_cli()

    # Si la carpeta no existe, descarga el kernel de Kaggle
    if not DATA_DIR.exists():
        print(f"No existe la carpeta {DATA_DIR}. Descargando kernel desde Kaggle...")
        download_kaggle_kernel(KAGGLE_KERNEL, DATA_DIR)

    csv_files = find_csv_files(DATA_DIR)
    if csv_files:
        data_path = csv_files[0]
        print(f"Cargando dataset desde CSV: {data_path}")
        return pd.read_csv(data_path)

    zip_files = find_zip_files(DATA_DIR)
    if zip_files:
        print(f"Se encontró un ZIP. Cargando el primer CSV dentro de: {zip_files[0]}")
        return load_csv_from_zip(zip_files[0])

    raise FileNotFoundError(
        f"No se encontró ningún CSV ni ZIP en {DATA_DIR}. "
        "Verifica si el kernel tiene la data dentro del proyecto o cambia el slug del kernel."
    )


if __name__ == "__main__":
    df = load_data()

    print("\nForma del dataset:", df.shape)
    print("Columnas:", list(df.columns))
    print("\nVista previa:")
    print(df.head())
