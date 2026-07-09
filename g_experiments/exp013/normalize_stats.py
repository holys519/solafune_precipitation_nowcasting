#!/usr/bin/env python3
"""Compute per-satellite, per-band normalization stats for exp013.

CPU-only. Samples a deterministic subset of train rows, reads the most recent
observation file per row (already resized to the 16-channel/uint8 layout, minus
the small share of anomalous channel counts handled the same way training does),
and accumulates per-channel mean/std per satellite. Writes norm_stats.json next
to this script, consumed by dataset.py at train/inference time.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tiff_utils import read_tiff_array

SCRIPT_DIR = Path(__file__).resolve().parent
SATELLITES = ("goes", "himawari", "meteosat")


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def read_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def channel_stats(
    path: Path,
    channels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-channel (sum, sumsq, count) for the file's available channels only."""

    arr, _ = read_tiff_array(path)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.astype(np.float64)
    n_chan = min(arr.shape[2], channels)
    arr = arr[:, :, :n_chan]
    flat = arr.reshape(-1, n_chan)
    sums = np.zeros(channels, dtype=np.float64)
    sumsqs = np.zeros(channels, dtype=np.float64)
    counts = np.zeros(channels, dtype=np.float64)
    sums[:n_chan] = flat.sum(axis=0)
    sumsqs[:n_chan] = np.square(flat).sum(axis=0)
    counts[:n_chan] = flat.shape[0]
    return sums, sumsqs, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--samples-per-satellite", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(SCRIPT_DIR / "norm_stats.json"))
    args = parser.parse_args()

    config = load_config(Path(args.config))
    train_csv = resolve_path(config["data"]["train_csv"])
    train_dir = resolve_path(config["data"]["train_dir"])
    channels = int(config["data"]["satellite_channels"])

    rows = read_rows(train_csv)
    by_satellite: dict[str, list[dict[str, str]]] = {sat: [] for sat in SATELLITES}
    for row in rows:
        sat = row["satellite_target"]
        if sat in by_satellite:
            by_satellite[sat].append(row)

    rng = random.Random(args.seed)
    stats: dict[str, Any] = {}
    for sat in SATELLITES:
        sat_rows = by_satellite[sat]
        rng.shuffle(sat_rows)
        sample = sat_rows[: args.samples_per_satellite]

        sums = np.zeros(channels, dtype=np.float64)
        sumsqs = np.zeros(channels, dtype=np.float64)
        counts = np.zeros(channels, dtype=np.float64)
        used_files = 0
        for row in sample:
            import ast

            obs_names = ast.literal_eval(row["last_30_minutes_observation_filename"])
            if not obs_names:
                continue
            path = train_dir / sat / str(obs_names[-1])
            if not path.exists():
                continue
            s, sq, c = channel_stats(path, channels)
            sums += s
            sumsqs += sq
            counts += c
            used_files += 1

        counts = np.maximum(counts, 1.0)
        mean = (sums / counts) / 255.0
        var = (sumsqs / counts) / (255.0**2) - np.square(mean)
        std = np.sqrt(np.clip(var, 1e-8, None))
        stats[sat] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "files_used": used_files,
        }
        print(f"{sat}: files_used={used_files} mean[:4]={mean[:4].round(4).tolist()}")

    output_path = Path(args.output)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
