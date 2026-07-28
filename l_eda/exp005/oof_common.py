"""l_eda/exp005 shared loading/alignment helpers for the robust-validation toolkit.

Reads the same `outputs/analysis/{exp}/oof_sample_metrics.csv` per-tile cache that
`l_eda/exp003` (E-3, CV->LB calibration) and `l_eda/exp004` (E-4, fold anatomy) already
established as this project's finest-grained OOF artifact. Nothing here retrains or
re-infers anything -- it is pure post-hoc analysis of existing caches.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_DIR / "outputs" / "analysis"
OUT_DIR = PROJECT_DIR / "outputs" / "l_eda" / "exp005"

# l_eda/exp003 (E-3, 12-16 submitted pairs): 5-fold OOF tile_rmse residual std vs public LB is
# 0.0033-0.0044 depending on pair count at time of fitting. Treat deltas smaller than this as
# indistinguishable from fold-composition noise regardless of what the point estimate says.
NOISE_FLOOR = 0.004


class OOFLoadError(RuntimeError):
    pass


def sample_metrics_path(exp: str) -> Path:
    return ANALYSIS_DIR / exp / "oof_sample_metrics.csv"


def load_sample_metrics(exp: str) -> list[dict]:
    path = sample_metrics_path(exp)
    if not path.exists():
        raise OOFLoadError(
            f"{path} not found -- run this experiment's analyze_oof.py first, or check the name "
            f"(available: {sorted(p.name for p in ANALYSIS_DIR.iterdir() if (p / 'oof_sample_metrics.csv').exists())})"
        )
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise OOFLoadError(f"{path} is empty")
    return rows


def metric_key(rows: list[dict]) -> str:
    return "tile_rmse" if "tile_rmse" in rows[0] else "rmse"


def align_pair(rows_a: list[dict], rows_b: list[dict], metric: str) -> tuple[dict[str, list[tuple[float, float]]], dict]:
    """Inner-join two experiments' per-tile metrics on unique_id, grouped by name_location.

    Returns (loc_tiles, diagnostics) where loc_tiles[location] is a list of (a_value, b_value)
    pairs for every unique_id present in both experiments and assigned to that location by
    experiment A. diagnostics reports how many rows were dropped from each side and whether A/B
    disagree on the location a shared unique_id belongs to (would indicate the two runs used a
    different fold/location split and are not directly comparable).
    """
    map_a = {row["unique_id"]: row for row in rows_a}
    map_b = {row["unique_id"]: row for row in rows_b}
    shared = sorted(set(map_a) & set(map_b))
    if not shared:
        raise OOFLoadError("no shared unique_id between the two experiments -- not comparable")

    mismatches = 0
    loc_tiles: dict[str, list[tuple[float, float]]] = {}
    for uid in shared:
        ra, rb = map_a[uid], map_b[uid]
        if ra["name_location"] != rb["name_location"]:
            mismatches += 1
            continue
        loc = ra["name_location"]
        loc_tiles.setdefault(loc, []).append((float(ra[metric]), float(rb[metric])))

    diagnostics = {
        "n_shared_tiles": len(shared),
        "n_dropped_a_only": len(map_a) - len(shared),
        "n_dropped_b_only": len(map_b) - len(shared),
        "n_location_mismatches": mismatches,
        "n_locations": len(loc_tiles),
    }
    return loc_tiles, diagnostics


def location_cluster_sums(loc_tiles: dict[str, list[tuple[float, float]]]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-location (sum_a, sum_b, n) so cluster bootstrap is O(n_locations) per draw."""
    locations = sorted(loc_tiles)
    sum_a = np.array([sum(a for a, _ in loc_tiles[loc]) for loc in locations], dtype=np.float64)
    sum_b = np.array([sum(b for _, b in loc_tiles[loc]) for loc in locations], dtype=np.float64)
    n = np.array([len(loc_tiles[loc]) for loc in locations], dtype=np.float64)
    return locations, sum_a, sum_b, n
