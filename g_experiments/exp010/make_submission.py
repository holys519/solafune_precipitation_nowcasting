#!/usr/bin/env python3
"""Create submission zip for exp010."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def main() -> None:
    config = load_config(SCRIPT_DIR / "config.yaml")
    output_dir = resolve_path(config["paths"]["output_dir"])
    zip_path = resolve_path(config["paths"]["submission_zip"])
    csv_path = output_dir / "evaluation_target.csv"
    test_files_dir = output_dir / "test_files"

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    tif_files = sorted(test_files_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No tif files under {test_files_dir}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(csv_path, "evaluation_target.csv")
        for idx, path in enumerate(tif_files, 1):
            zf.write(path, f"test_files/{path.name}")
            if idx % 5000 == 0:
                print(f"zipped {idx}/{len(tif_files)}", flush=True)
    print(f"created submission: {zip_path} files={len(tif_files) + 1}")


if __name__ == "__main__":
    main()
