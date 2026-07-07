#!/usr/bin/env python3
"""Inspect satellite files whose metadata differs from the common shape."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/solafune_precipitation_nowcasting_mplconfig")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "eda" / "outputs"
TRAIN_DIR = PROJECT_ROOT / "data" / "train_dataset"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation_dataset"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eda01 = load_module("eda01", PROJECT_ROOT / "eda" / "01_data_overview.py")
eda02 = load_module("eda02", PROJECT_ROOT / "eda" / "02_deep_dive.py")


EXPECTED = {
    "goes": {"width": 141, "height": 141, "samples_per_pixel": 16, "dtype": "uint8"},
    "himawari": {"width": 81, "height": 81, "samples_per_pixel": 16, "dtype": "uint8"},
    "meteosat": {"width": 144, "height": 144, "samples_per_pixel": 16, "dtype": "uint8"},
}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_reference_index(rows: list[dict[str, str]], split: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        satellite = row["satellite_target"]
        for filename in eda01.parse_obs(row["last_30_minutes_observation_filename"]):
            index[(satellite, filename)].append(
                {
                    "split": split,
                    "unique_id": row["unique_id"],
                    "name_location": row["name_location"],
                    "satellite_target": satellite,
                    "datetime": row["datetime"],
                    "obs_count": str(len(eda01.parse_obs(row["last_30_minutes_observation_filename"]))),
                }
            )
    return index


def is_expected(satellite: str, meta: dict[str, Any]) -> bool:
    expected = EXPECTED[satellite]
    return all(meta[key] == value for key, value in expected.items())


def inspect_array(path: Path) -> dict[str, Any]:
    arr, _ = eda01.read_tiff_array(path)
    return {
        "array_shape": "x".join(str(x) for x in arr.shape),
        "array_min": float(np.nanmin(arr)),
        "array_max": float(np.nanmax(arr)),
        "array_mean": float(np.nanmean(arr)),
        "array_std": float(np.nanstd(arr)),
    }


def scan_split(split: str, split_dir: Path, reference_index: dict[tuple[str, str], list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for satellite in ["goes", "himawari", "meteosat"]:
        for path in sorted((split_dir / satellite).glob("*.tif")):
            meta = eda02.read_tiff_meta_light(path)
            if is_expected(satellite, meta):
                continue

            refs = reference_index.get((satellite, path.name), [])
            ref_locations = sorted({ref["name_location"] for ref in refs})
            ref_datetimes = [ref["datetime"] for ref in refs[:10]]
            ref_unique_ids = [ref["unique_id"] for ref in refs[:10]]
            ref_obs_counts = sorted(Counter(ref["obs_count"] for ref in refs).items())
            row = {
                "split": split,
                "satellite": satellite,
                "filename": path.name,
                "path": eda01.rel(path),
                "width": meta["width"],
                "height": meta["height"],
                "samples_per_pixel": meta["samples_per_pixel"],
                "bits_per_sample": meta["bits_per_sample"],
                "sample_format": meta["sample_format"],
                "compression": meta["compression"],
                "rows_per_strip": meta["rows_per_strip"],
                "dtype": meta["dtype"],
                "file_size_bytes": path.stat().st_size,
                "referenced_rows": len(refs),
                "referenced_locations": "|".join(ref_locations),
                "referenced_obs_counts": json.dumps(ref_obs_counts),
                "first_unique_ids": "|".join(ref_unique_ids),
                "first_datetimes": "|".join(ref_datetimes),
            }
            row.update(inspect_array(path))
            rows.append(row)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = eda01.read_csv_rows(TRAIN_DIR / "train_dataset.csv")
    eval_rows = eda01.read_csv_rows(EVAL_DIR / "evaluation_target.csv")

    train_index = build_reference_index(train_rows, "train")
    eval_index = build_reference_index(eval_rows, "evaluation")

    anomaly_rows = scan_split("train", TRAIN_DIR, train_index) + scan_split("evaluation", EVAL_DIR, eval_index)
    if anomaly_rows:
        write_csv(
            OUTPUT_DIR / "satellite_anomaly_files.csv",
            anomaly_rows,
            list(anomaly_rows[0].keys()),
        )

    summary = Counter(
        (
            row["split"],
            row["satellite"],
            row["width"],
            row["height"],
            row["samples_per_pixel"],
            row["referenced_rows"],
        )
        for row in anomaly_rows
    )
    summary_rows = [
        {
            "split": key[0],
            "satellite": key[1],
            "width": key[2],
            "height": key[3],
            "samples_per_pixel": key[4],
            "referenced_rows_per_file": key[5],
            "file_count": count,
        }
        for key, count in sorted(summary.items())
    ]
    write_csv(
        OUTPUT_DIR / "satellite_anomaly_summary.csv",
        summary_rows,
        ["split", "satellite", "width", "height", "samples_per_pixel", "referenced_rows_per_file", "file_count"],
    )

    report = f"""# Satellite Anomaly Inspection

Generated by `eda/03_satellite_anomalies.py`.

Expected satellite shapes:

- GOES: `141x141x16`
- Himawari: `81x81x16`
- Meteosat: `144x144x16`

Anomaly files found: `{len(anomaly_rows)}`.

## Summary

| split | satellite | width | height | samples_per_pixel | referenced_rows_per_file | file_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
"""
    for row in summary_rows:
        report += (
            f"| {row['split']} | {row['satellite']} | {row['width']} | {row['height']} | "
            f"{row['samples_per_pixel']} | {row['referenced_rows_per_file']} | {row['file_count']} |\n"
        )
    report += """
## Outputs

- `eda/outputs/satellite_anomaly_files.csv`
- `eda/outputs/satellite_anomaly_summary.csv`

## Modeling Note

These files are referenced by CSV rows, so the dataset loader must handle variable channel counts and occasional GOES `282x282x4` files. A robust baseline should allocate a fixed 16-channel tensor and copy available bands into the expected positions, or implement explicit per-satellite adapters with missing-band masks.
"""
    (OUTPUT_DIR / "SATELLITE_ANOMALIES.md").write_text(report, encoding="utf-8")
    print(f"Satellite anomaly files: {len(anomaly_rows)}")
    print(f"Wrote {OUTPUT_DIR / 'SATELLITE_ANOMALIES.md'}")


if __name__ == "__main__":
    main()
