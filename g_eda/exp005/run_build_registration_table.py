#!/usr/bin/env python3
"""Build a merged train+eval parallax registration table (Round 5 plan H3 -> exp045).

Only Himawari gets a real registration shift: it's the only satellite whose geometric
model was validated against empirical shifts (direction cosine 0.81, g_eda/exp005). GOES
(cosine 0.70, R^2 0.41) and Meteosat (R^2 negative) are left at zero shift for BOTH splits
-- applying an unvalidated geometric extrapolation only on the eval side (while leaving
train unregistered) would introduce a train/eval processing mismatch, which is worse than
no registration at all.

- Train Himawari locations: empirical median (dy, dx) from l_eda's phase-correlation
  measurement (outputs/g_eda/exp001/parallax_shift_by_location.csv) -- more accurate than
  the geometric formula for locations we've actually measured.
- Eval Himawari locations: geometric-model prediction (outputs/g_eda/exp005/
  eval_parallax_prior.csv), the only source available since eval has no GPM truth to
  measure an empirical shift against.

Output: outputs/g_eda/exp005/registration_shifts.json, {location: {satellite: [dy, dx]}}
covering every location in both train_dataset.csv and evaluation_target.csv (zero for all
non-Himawari rows).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
EMPIRICAL_CSV = PROJECT_DIR / "outputs" / "g_eda" / "exp001" / "parallax_shift_by_location.csv"
EVAL_PRIOR_CSV = PROJECT_DIR / "outputs" / "g_eda" / "exp005" / "eval_parallax_prior.csv"
TRAIN_CSV = PROJECT_DIR / "data" / "train_dataset" / "train_dataset.csv"
EVAL_CSV = PROJECT_DIR / "data" / "evaluation_dataset" / "evaluation_target.csv"
OUT_PATH = PROJECT_DIR / "outputs" / "g_eda" / "exp005" / "registration_shifts.json"

REGISTERED_SATELLITES = {"himawari"}  # only the validated one; goes/meteosat stay at 0,0


def location_satellite_pairs(csv_path: Path) -> set[tuple[str, str]]:
    with csv_path.open(newline="") as f:
        return {(row["name_location"], row["satellite_target"]) for row in csv.DictReader(f)}


def main() -> None:
    all_pairs = location_satellite_pairs(TRAIN_CSV) | location_satellite_pairs(EVAL_CSV)

    empirical: dict[tuple[str, str], tuple[float, float]] = {}
    with EMPIRICAL_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            empirical[(row["name_location"], row["satellite_target"])] = (
                float(row["median_dy"]), float(row["median_dx"]))

    geometric: dict[tuple[str, str], tuple[float, float]] = {}
    with EVAL_PRIOR_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            geometric[(row["name_location"], row["satellite_target"])] = (
                float(row["pred_dy"]), float(row["pred_dx"]))

    table: dict[str, dict[str, list[float]]] = {}
    source_log = []
    for location, satellite in sorted(all_pairs):
        dy, dx, source = 0.0, 0.0, "unregistered_satellite"
        if satellite in REGISTERED_SATELLITES:
            if (location, satellite) in empirical:
                dy, dx = empirical[(location, satellite)]
                source = "empirical_train"
            elif (location, satellite) in geometric:
                dy, dx = geometric[(location, satellite)]
                source = "geometric_eval"
            else:
                source = "missing_no_shift"
        table.setdefault(location, {})[satellite] = [dy, dx]
        source_log.append({"location": location, "satellite": satellite,
                           "dy": dy, "dx": dx, "source": source})

    OUT_PATH.write_text(json.dumps(table, indent=2), encoding="utf-8")
    log_path = OUT_PATH.with_name("registration_shifts_log.csv")
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["location", "satellite", "dy", "dx", "source"])
        writer.writeheader()
        writer.writerows(source_log)

    n_registered = sum(1 for r in source_log if r["source"] in ("empirical_train", "geometric_eval"))
    print(f"wrote {OUT_PATH} ({len(all_pairs)} location/satellite pairs, "
          f"{n_registered} with a nonzero registration source)", flush=True)
    for row in source_log:
        if row["source"] != "unregistered_satellite":
            print(row, flush=True)


if __name__ == "__main__":
    main()
