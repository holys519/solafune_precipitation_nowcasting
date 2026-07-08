#!/usr/bin/env python3
"""Quick data readability check for exp001."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from dataset import PrecipDataset, parse_observation_filenames, read_rows


SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def select_row(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if parse_observation_filenames(row["last_30_minutes_observation_filename"]):
            return row
    if not rows:
        raise RuntimeError("CSV has no rows")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config((SCRIPT_DIR / args.config).resolve())
    train_csv = resolve_path(config["data"]["train_csv"])
    eval_csv = resolve_path(config["data"]["evaluation_csv"])
    train_dir = resolve_path(config["data"]["train_dir"])
    eval_dir = resolve_path(config["data"]["evaluation_dir"])
    sample_dir = resolve_path(config["data"]["sample_submission_dir"])

    required = [
        train_csv,
        eval_csv,
        train_dir / "goes",
        train_dir / "himawari",
        train_dir / "meteosat",
        train_dir / "gpm_imerg",
        eval_dir / "goes",
        eval_dir / "himawari",
        eval_dir / "meteosat",
        eval_dir / "test_files",
        sample_dir / "test_files",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required data paths:\n" + "\n".join(missing))

    target_size = (int(config["data"]["target_height"]), int(config["data"]["target_width"]))
    common_kwargs = {
        "max_observations": int(config["data"]["max_observations"]),
        "satellite_channels": int(config["data"]["satellite_channels"]),
        "target_size": target_size,
    }

    train_rows = read_rows(train_csv)
    eval_rows = read_rows(eval_csv)
    train_ds = PrecipDataset([select_row(train_rows)], train_dir, has_target=True, **common_kwargs)
    eval_ds = PrecipDataset([select_row(eval_rows)], eval_dir, has_target=False, **common_kwargs)

    train_item = train_ds[0]
    eval_item = eval_ds[0]
    print(
        "data check ok: "
        f"train_rows={len(train_rows)} eval_rows={len(eval_rows)} "
        f"x_train={tuple(train_item['x'].shape)} y={tuple(train_item['y'].shape)} "
        f"x_eval={tuple(eval_item['x'].shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
