#!/usr/bin/env python3
"""Compute per-satellite, per-band normalization stats for exp016 (copied from exp009).

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-channel (sum, sumsq, count, zero_count) for available channels only.

    zero_count is the number of pixels exactly equal to 0 per band, computed on the NATIVE
    (pre-resize) raster. exp059 uses this to classify bands as emissive-fill vs reflective-night
    (peppamint's discussion diagnostic): a hard spike of exact-0 pixels is the fingerprint of a
    NODATA fill value in an emissive (IR/WV) band, whereas a large 0 fraction in a
    reflective/visible band is genuine night darkness that must be preserved."""

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
    zero_counts = np.zeros(channels, dtype=np.float64)
    sums[:n_chan] = flat.sum(axis=0)
    sumsqs[:n_chan] = np.square(flat).sum(axis=0)
    counts[:n_chan] = flat.shape[0]
    zero_counts[:n_chan] = (flat == 0).sum(axis=0)
    return sums, sumsqs, counts, zero_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--samples-per-satellite", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emissive-zero-frac-threshold", type=float, default=0.10)
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
        zero_counts = np.zeros(channels, dtype=np.float64)
        used_files = 0
        for row in sample:
            import ast

            obs_names = ast.literal_eval(row["last_30_minutes_observation_filename"])
            if not obs_names:
                continue
            path = train_dir / sat / str(obs_names[-1])
            if not path.exists():
                continue
            s, sq, c, z = channel_stats(path, channels)
            sums += s
            sumsqs += sq
            counts += c
            zero_counts += z
            used_files += 1

        counts = np.maximum(counts, 1.0)
        mean = (sums / counts) / 255.0
        var = (sumsqs / counts) / (255.0**2) - np.square(mean)
        std = np.sqrt(np.clip(var, 1e-8, None))
        zero_frac = zero_counts / counts
        # Emissive-fill bands: low exact-0 fraction (a spike at the bottom = NODATA fill), as
        # opposed to reflective/visible bands where a large 0 fraction is night. Threshold 0.10
        # cleanly separates the two regimes reported in the discussion (night bands 30-43% exact-0,
        # emissive fill a couple % at most). Recorded per band; dataset.py masks emissive zeros only.
        emissive = (zero_frac < args.emissive_zero_frac_threshold).astype(int)
        stats[sat] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "zero_frac": zero_frac.tolist(),
            "emissive": emissive.tolist(),
            "files_used": used_files,
        }
        print(
            f"{sat}: files_used={used_files} mean[:4]={mean[:4].round(4).tolist()} "
            f"zero_frac={np.round(zero_frac, 3).tolist()} emissive={emissive.tolist()}"
        )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
