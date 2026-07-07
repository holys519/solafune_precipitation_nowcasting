#!/usr/bin/env python3
"""Dataset overview EDA without pandas/rasterio dependencies."""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import random
import struct
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/solafune_precipitation_nowcasting_mplconfig")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train_dataset"
EVAL_DIR = DATA_DIR / "evaluation_dataset"
SAMPLE_DIR = DATA_DIR / "sample_submission"
OUTPUT_DIR = PROJECT_ROOT / "eda" / "outputs"

TRAIN_CSV = TRAIN_DIR / "train_dataset.csv"
EVAL_CSV = EVAL_DIR / "evaluation_target.csv"
SAMPLE_CSV = SAMPLE_DIR / "evaluation_target.csv"

RANDOM_SEED = 42
SAMPLES_PER_SATELLITE = 24
TARGET_SAMPLE_SIZE = 600


TIFF_TYPE_INFO = {
    1: ("B", 1),
    2: ("c", 1),
    3: ("H", 2),
    4: ("I", 4),
    5: ("II", 8),
    11: ("f", 4),
    12: ("d", 8),
}

TIFF_TAG_NAMES = {
    256: "width",
    257: "height",
    258: "bits_per_sample",
    259: "compression",
    262: "photometric",
    273: "strip_offsets",
    277: "samples_per_pixel",
    278: "rows_per_strip",
    279: "strip_byte_counts",
    284: "planar_config",
    317: "predictor",
    339: "sample_format",
}


@dataclass
class TiffMeta:
    path: str
    width: int
    height: int
    samples_per_pixel: int
    bits_per_sample: tuple[int, ...]
    sample_format: tuple[int, ...]
    compression: int
    rows_per_strip: int
    strip_count: int
    dtype: str


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def parse_obs(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed)!r}")
    return [str(item) for item in parsed]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def counter_rows(counter: Counter, key_name: str, value_name: str = "count") -> list[dict[str, Any]]:
    return [{key_name: key, value_name: value} for key, value in counter.most_common()]


def split_summary(rows: list[dict[str, str]], split_name: str) -> dict[str, Any]:
    satellite = Counter()
    location = Counter()
    year = Counter()
    month = Counter()
    location_satellite = Counter()
    obs_len = Counter()
    unique_ids = Counter()
    parse_errors = 0
    min_dt: datetime | None = None
    max_dt: datetime | None = None

    for row in rows:
        sat = row["satellite_target"]
        loc = row["name_location"]
        satellite[sat] += 1
        location[loc] += 1
        location_satellite[(loc, sat)] += 1
        unique_ids[row["unique_id"]] += 1
        dt = parse_dt(row["datetime"])
        min_dt = dt if min_dt is None else min(min_dt, dt)
        max_dt = dt if max_dt is None else max(max_dt, dt)
        year[str(dt.year)] += 1
        month[dt.strftime("%Y-%m")] += 1
        try:
            obs_len[len(parse_obs(row["last_30_minutes_observation_filename"]))] += 1
        except Exception:
            parse_errors += 1

    return {
        "split": split_name,
        "rows": len(rows),
        "unique_ids": len(unique_ids),
        "duplicate_unique_ids": sum(1 for count in unique_ids.values() if count > 1),
        "locations": len(location),
        "satellite_counts": dict(sorted(satellite.items())),
        "top_locations": dict(location.most_common(20)),
        "year_counts": dict(sorted(year.items())),
        "month_counts": dict(sorted(month.items())),
        "observation_length_counts": dict(sorted((str(k), v) for k, v in obs_len.items())),
        "observation_parse_errors": parse_errors,
        "datetime_min": min_dt.isoformat(sep=" ") if min_dt else None,
        "datetime_max": max_dt.isoformat(sep=" ") if max_dt else None,
        "location_satellite_counts": {
            f"{loc}|{sat}": count for (loc, sat), count in location_satellite.most_common()
        },
    }


def resolve_observation_dir(split_dir: Path, satellite: str) -> Path:
    return split_dir / satellite


def validate_paths(rows: list[dict[str, str]], split_dir: Path, has_target: bool) -> dict[str, Any]:
    missing_obs = 0
    missing_target = 0
    checked_obs = 0
    checked_target = 0
    missing_examples: list[str] = []

    for row in rows:
        satellite = row["satellite_target"]
        obs_dir = resolve_observation_dir(split_dir, satellite)
        try:
            obs_names = parse_obs(row["last_30_minutes_observation_filename"])
        except Exception as exc:
            missing_examples.append(f"observation_parse_error:{row['unique_id']}:{exc}")
            continue

        for name in obs_names:
            checked_obs += 1
            path = obs_dir / name
            if not path.exists():
                missing_obs += 1
                if len(missing_examples) < 20:
                    missing_examples.append(rel(path))

        if has_target:
            checked_target += 1
            path = split_dir / "gpm_imerg" / row["gpm_imerg_filename"]
            if not path.exists():
                missing_target += 1
                if len(missing_examples) < 20:
                    missing_examples.append(rel(path))
        else:
            checked_target += 1
            path = split_dir / "test_files" / row["gpm_imerg_filename"]
            if not path.exists():
                missing_target += 1
                if len(missing_examples) < 20:
                    missing_examples.append(rel(path))

    return {
        "checked_observation_files": checked_obs,
        "missing_observation_files": missing_obs,
        "checked_target_or_template_files": checked_target,
        "missing_target_or_template_files": missing_target,
        "missing_examples": missing_examples,
    }


def tiff_raw_value(data: bytes, endian: str, typ: int, count: int, raw: bytes) -> Any:
    fmt, size = TIFF_TYPE_INFO[typ]
    total = count * size
    if total <= 4:
        buf = raw[:total]
    else:
        offset = struct.unpack(endian + "I", raw)[0]
        buf = data[offset : offset + total]

    if typ == 2:
        return buf.rstrip(b"\0").decode("utf-8", "replace")
    if typ == 5:
        values = []
        for i in range(count):
            numerator, denominator = struct.unpack(endian + "II", buf[i * 8 : i * 8 + 8])
            values.append(numerator / denominator if denominator else math.nan)
        return values[0] if count == 1 else tuple(values)

    values = struct.unpack(endian + (fmt * count), buf)
    return values[0] if count == 1 else tuple(values)


def read_tiff_tags(path: Path) -> tuple[bytes, str, dict[str, Any]]:
    data = path.read_bytes()
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        raise ValueError(f"Not a TIFF file: {path}")

    magic = struct.unpack(endian + "H", data[2:4])[0]
    if magic != 42:
        raise ValueError(f"Unsupported TIFF magic {magic}: {path}")

    ifd_offset = struct.unpack(endian + "I", data[4:8])[0]
    tag_count = struct.unpack(endian + "H", data[ifd_offset : ifd_offset + 2])[0]
    tags: dict[str, Any] = {}
    for i in range(tag_count):
        offset = ifd_offset + 2 + i * 12
        tag, typ, count = struct.unpack(endian + "HHI", data[offset : offset + 8])
        raw = data[offset + 8 : offset + 12]
        name = TIFF_TAG_NAMES.get(tag, f"tag_{tag}")
        if typ in TIFF_TYPE_INFO:
            tags[name] = tiff_raw_value(data, endian, typ, count, raw)
    return data, endian, tags


def as_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    return (int(value),)


def dtype_from_tags(bits_per_sample: tuple[int, ...], sample_format: tuple[int, ...]) -> np.dtype:
    bits = bits_per_sample[0]
    fmt = sample_format[0] if sample_format else 1
    if fmt == 1 and bits == 8:
        return np.dtype("uint8")
    if fmt == 1 and bits == 16:
        return np.dtype("uint16")
    if fmt == 3 and bits == 32:
        return np.dtype("float32")
    if fmt == 3 and bits == 64:
        return np.dtype("float64")
    raise ValueError(f"Unsupported TIFF dtype sample_format={fmt}, bits={bits}")


def read_tiff_array(path: Path) -> tuple[np.ndarray, TiffMeta]:
    data, endian, tags = read_tiff_tags(path)
    width = int(tags["width"])
    height = int(tags["height"])
    samples = int(tags.get("samples_per_pixel", 1))
    bits = as_tuple(tags["bits_per_sample"])
    sample_format = as_tuple(tags.get("sample_format", 1))
    compression = int(tags.get("compression", 1))
    rows_per_strip = int(tags.get("rows_per_strip", height))
    offsets = as_tuple(tags["strip_offsets"])
    byte_counts = as_tuple(tags["strip_byte_counts"])
    dtype = dtype_from_tags(bits, sample_format)

    chunks: list[bytes] = []
    for offset, byte_count in zip(offsets, byte_counts):
        chunk = data[offset : offset + byte_count]
        if compression == 1:
            chunks.append(chunk)
        elif compression == 8:
            chunks.append(zlib.decompress(chunk))
        else:
            raise ValueError(f"Unsupported TIFF compression={compression}: {path}")

    raw = b"".join(chunks)
    endian_dtype = np.dtype(dtype).newbyteorder("<" if endian == "<" else ">")
    arr = np.frombuffer(raw, dtype=endian_dtype)
    expected = width * height * samples
    if arr.size != expected:
        raise ValueError(f"Unexpected TIFF size {arr.size} != {expected}: {path}")
    if samples == 1:
        arr = arr.reshape(height, width)
    else:
        arr = arr.reshape(height, width, samples)
    arr = arr.astype(dtype, copy=False)

    meta = TiffMeta(
        path=rel(path),
        width=width,
        height=height,
        samples_per_pixel=samples,
        bits_per_sample=bits,
        sample_format=sample_format,
        compression=compression,
        rows_per_strip=rows_per_strip,
        strip_count=len(offsets),
        dtype=str(dtype),
    )
    return arr, meta


def sample_rows_by_satellite(rows: list[dict[str, str]], n_per_sat: int) -> list[dict[str, str]]:
    rng = random.Random(RANDOM_SEED)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["satellite_target"]].append(row)
    sampled: list[dict[str, str]] = []
    for satellite, group in sorted(grouped.items()):
        sampled.extend(rng.sample(group, min(n_per_sat, len(group))))
    return sampled


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    result = {}
    for q, value in zip(qs, np.percentile(values, qs)):
        result[f"p{q}"] = float(value)
    return result


def geotiff_sample_eda(train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]]) -> dict[str, Any]:
    rng = random.Random(RANDOM_SEED)
    satellite_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    metadata_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for split_name, split_dir, rows in [
        ("train", TRAIN_DIR, sample_rows_by_satellite(train_rows, SAMPLES_PER_SATELLITE)),
        ("evaluation", EVAL_DIR, sample_rows_by_satellite(eval_rows, SAMPLES_PER_SATELLITE)),
    ]:
        for row in rows:
            obs_names = parse_obs(row["last_30_minutes_observation_filename"])
            # Use the most recent observation in the 30-minute window for satellite value EDA.
            obs_path = split_dir / row["satellite_target"] / obs_names[-1]
            try:
                arr, meta = read_tiff_array(obs_path)
            except Exception as exc:
                errors.append(f"{rel(obs_path)}: {type(exc).__name__}: {exc}")
                continue
            metadata_records.append({"split": split_name, "kind": "satellite", **meta.__dict__})
            flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1, 1)
            band_means = flat.mean(axis=0)
            band_stds = flat.std(axis=0)
            satellite_records.append(
                {
                    "split": split_name,
                    "satellite": row["satellite_target"],
                    "location": row["name_location"],
                    "path": rel(obs_path),
                    "height": int(arr.shape[0]),
                    "width": int(arr.shape[1]),
                    "channels": int(arr.shape[2]) if arr.ndim == 3 else 1,
                    "dtype": str(arr.dtype),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "band_means": [float(x) for x in band_means],
                    "band_stds": [float(x) for x in band_stds],
                }
            )

    target_candidates = train_rows[:]
    target_sample = rng.sample(target_candidates, min(TARGET_SAMPLE_SIZE, len(target_candidates)))
    all_target_values: list[np.ndarray] = []
    for row in target_sample:
        target_path = TRAIN_DIR / "gpm_imerg" / row["gpm_imerg_filename"]
        try:
            arr, meta = read_tiff_array(target_path)
        except Exception as exc:
            errors.append(f"{rel(target_path)}: {type(exc).__name__}: {exc}")
            continue
        metadata_records.append({"split": "train", "kind": "target", **meta.__dict__})
        values = arr.astype("float64").ravel()
        finite = values[np.isfinite(values)]
        all_target_values.append(finite)
        target_records.append(
            {
                "location": row["name_location"],
                "satellite": row["satellite_target"],
                "path": rel(target_path),
                "height": int(arr.shape[0]),
                "width": int(arr.shape[1]),
                "dtype": str(arr.dtype),
                "min": float(finite.min()) if finite.size else math.nan,
                "max": float(finite.max()) if finite.size else math.nan,
                "mean": float(finite.mean()) if finite.size else math.nan,
                "std": float(finite.std()) if finite.size else math.nan,
                "zero_fraction": float((finite == 0).mean()) if finite.size else math.nan,
                "positive_fraction": float((finite > 0).mean()) if finite.size else math.nan,
            }
        )

    target_values = np.concatenate(all_target_values) if all_target_values else np.array([])
    target_summary = {
        "sample_files": len(target_records),
        "sample_pixels": int(target_values.size),
        "mean": float(target_values.mean()) if target_values.size else None,
        "std": float(target_values.std()) if target_values.size else None,
        "zero_fraction": float((target_values == 0).mean()) if target_values.size else None,
        "positive_fraction": float((target_values > 0).mean()) if target_values.size else None,
        "percentiles": percentile_dict(target_values),
    }

    return {
        "satellite_records": satellite_records,
        "target_records": target_records,
        "metadata_records": metadata_records,
        "target_summary": target_summary,
        "errors": errors,
    }


def aggregate_satellite_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["split"], record["satellite"])].append(record)

    rows = []
    for (split, satellite), group in sorted(grouped.items()):
        rows.append(
            {
                "split": split,
                "satellite": satellite,
                "sample_files": len(group),
                "height_values": sorted(set(r["height"] for r in group)),
                "width_values": sorted(set(r["width"] for r in group)),
                "channel_values": sorted(set(r["channels"] for r in group)),
                "dtype_values": sorted(set(r["dtype"] for r in group)),
                "mean_of_file_means": float(np.mean([r["mean"] for r in group])),
                "mean_of_file_stds": float(np.mean([r["std"] for r in group])),
                "min_value": float(np.min([r["min"] for r in group])),
                "max_value": float(np.max([r["max"] for r in group])),
            }
        )
    return rows


def make_plots(train_summary: dict[str, Any], eval_summary: dict[str, Any], geo: dict[str, Any]) -> list[str]:
    plot_paths: list[str] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save_current(name: str) -> None:
        path = OUTPUT_DIR / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths.append(rel(path))

    for summary in [train_summary, eval_summary]:
        counts = summary["satellite_counts"]
        plt.figure(figsize=(6, 4))
        plt.bar(list(counts.keys()), list(counts.values()), color=["#4C78A8", "#F58518", "#54A24B"])
        plt.title(f"{summary['split']} rows by satellite")
        plt.ylabel("rows")
        save_current(f"{summary['split']}_satellite_counts.png")

        top = list(summary["top_locations"].items())[:15]
        plt.figure(figsize=(8, 5))
        plt.barh([x[0] for x in reversed(top)], [x[1] for x in reversed(top)], color="#4C78A8")
        plt.title(f"{summary['split']} top locations")
        plt.xlabel("rows")
        save_current(f"{summary['split']}_top_locations.png")

    target_records = geo["target_records"]
    if target_records:
        target_values = []
        for record in target_records:
            # Re-read only sampled targets for plotting. This keeps memory bounded.
            path = PROJECT_ROOT / record["path"]
            arr, _ = read_tiff_array(path)
            target_values.append(arr.astype("float64").ravel())
        values = np.concatenate(target_values)
        clip_hi = np.percentile(values, 99.5)
        plt.figure(figsize=(7, 4))
        plt.hist(values[values <= clip_hi], bins=80, color="#4C78A8")
        plt.title("Sampled GPM-IMERG target distribution (<= p99.5)")
        plt.xlabel("precipitation")
        plt.ylabel("pixels")
        save_current("target_distribution_sampled.png")

    satellite_records = geo["satellite_records"]
    if satellite_records:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for record in satellite_records:
            grouped[f"{record['split']}:{record['satellite']}"].append(np.array(record["band_means"]))
        plt.figure(figsize=(9, 5))
        for label, values in sorted(grouped.items()):
            mean_band = np.vstack(values).mean(axis=0)
            plt.plot(range(1, len(mean_band) + 1), mean_band, marker="o", label=label)
        plt.title("Sampled satellite band means")
        plt.xlabel("band index")
        plt.ylabel("mean uint8 value")
        plt.legend(fontsize=8)
        save_current("satellite_band_means_sampled.png")

    return plot_paths


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(results: dict[str, Any]) -> None:
    train = results["summaries"]["train"]
    evaluation = results["summaries"]["evaluation"]
    path_validation = results["path_validation"]
    geo = results["geotiff_sample"]
    satellite_agg = aggregate_satellite_records(geo["satellite_records"])
    target_summary = geo["target_summary"]

    report = f"""# EDA Report

Generated by `eda/01_data_overview.py`.

## CSV Overview

| Split | Rows | Unique IDs | Duplicate IDs | Locations | Datetime min | Datetime max |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| train | {train['rows']} | {train['unique_ids']} | {train['duplicate_unique_ids']} | {train['locations']} | {train['datetime_min']} | {train['datetime_max']} |
| evaluation | {evaluation['rows']} | {evaluation['unique_ids']} | {evaluation['duplicate_unique_ids']} | {evaluation['locations']} | {evaluation['datetime_min']} | {evaluation['datetime_max']} |

## Satellite Counts

### Train

{markdown_table(counter_rows(Counter(train['satellite_counts']), 'satellite'), ['satellite', 'count'])}

### Evaluation

{markdown_table(counter_rows(Counter(evaluation['satellite_counts']), 'satellite'), ['satellite', 'count'])}

## Observation Window Length

### Train

{markdown_table(counter_rows(Counter(train['observation_length_counts']), 'observation_count'), ['observation_count', 'count'])}

### Evaluation

{markdown_table(counter_rows(Counter(evaluation['observation_length_counts']), 'observation_count'), ['observation_count', 'count'])}

## File Existence Checks

| Split | Checked observation files | Missing observation files | Checked target/template files | Missing target/template files |
| --- | ---: | ---: | ---: | ---: |
| train | {path_validation['train']['checked_observation_files']} | {path_validation['train']['missing_observation_files']} | {path_validation['train']['checked_target_or_template_files']} | {path_validation['train']['missing_target_or_template_files']} |
| evaluation | {path_validation['evaluation']['checked_observation_files']} | {path_validation['evaluation']['missing_observation_files']} | {path_validation['evaluation']['checked_target_or_template_files']} | {path_validation['evaluation']['missing_target_or_template_files']} |

## GeoTIFF Sample Summary

Satellite samples use the most recent file in each 30-minute observation list. Target samples use `{TARGET_SAMPLE_SIZE}` deterministic train rows.

{markdown_table(satellite_agg, ['split', 'satellite', 'sample_files', 'height_values', 'width_values', 'channel_values', 'dtype_values', 'mean_of_file_means', 'mean_of_file_stds', 'min_value', 'max_value'])}

## Target Distribution Sample

| Metric | Value |
| --- | ---: |
| sampled files | {target_summary['sample_files']} |
| sampled pixels | {target_summary['sample_pixels']} |
| mean | {target_summary['mean']} |
| std | {target_summary['std']} |
| zero fraction | {target_summary['zero_fraction']} |
| positive fraction | {target_summary['positive_fraction']} |

Percentiles:

{markdown_table([{'percentile': k, 'value': v} for k, v in target_summary['percentiles'].items()], ['percentile', 'value'])}

## Plots

{chr(10).join(f'- `{path}`' for path in results['plots'])}

## Notes

- Actual ID column is `unique_id`, not `data_id`.
- `last_30_minutes_observation_filename` is a Python-list string; parse with `ast.literal_eval`.
- Satellite GeoTIFFs are sampled as 16-channel `uint8` images.
- GPM-IMERG target GeoTIFFs are sampled as single-channel `float32` rasters.
- Use location/time-aware CV because locations differ between train and evaluation.
"""
    (OUTPUT_DIR / "EDA_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = read_csv_rows(TRAIN_CSV)
    eval_rows = read_csv_rows(EVAL_CSV)
    sample_rows = read_csv_rows(SAMPLE_CSV)

    summaries = {
        "train": split_summary(train_rows, "train"),
        "evaluation": split_summary(eval_rows, "evaluation"),
        "sample_submission": split_summary(sample_rows, "sample_submission"),
    }

    path_validation = {
        "train": validate_paths(train_rows, TRAIN_DIR, has_target=True),
        "evaluation": validate_paths(eval_rows, EVAL_DIR, has_target=False),
    }

    geo = geotiff_sample_eda(train_rows, eval_rows)
    plots = make_plots(summaries["train"], summaries["evaluation"], geo)

    results = {
        "summaries": summaries,
        "path_validation": path_validation,
        "geotiff_sample": {
            "target_summary": geo["target_summary"],
            "errors": geo["errors"],
            "satellite_sample_count": len(geo["satellite_records"]),
            "target_sample_count": len(geo["target_records"]),
        },
        "plots": plots,
    }
    (OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for split_name, rows in summaries.items():
        write_csv(
            OUTPUT_DIR / f"{split_name}_satellite_counts.csv",
            counter_rows(Counter(rows["satellite_counts"]), "satellite"),
            ["satellite", "count"],
        )
        write_csv(
            OUTPUT_DIR / f"{split_name}_top_locations.csv",
            counter_rows(Counter(rows["top_locations"]), "location"),
            ["location", "count"],
        )
        loc_sat_rows = [
            {"location": key.split("|")[0], "satellite": key.split("|")[1], "count": count}
            for key, count in rows["location_satellite_counts"].items()
        ]
        write_csv(
            OUTPUT_DIR / f"{split_name}_location_satellite_counts.csv",
            loc_sat_rows,
            ["location", "satellite", "count"],
        )

    write_csv(
        OUTPUT_DIR / "satellite_sample_stats.csv",
        [
            {
                **{k: v for k, v in record.items() if k not in {"band_means", "band_stds"}},
                "band_means": json.dumps(record["band_means"]),
                "band_stds": json.dumps(record["band_stds"]),
            }
            for record in geo["satellite_records"]
        ],
        [
            "split",
            "satellite",
            "location",
            "path",
            "height",
            "width",
            "channels",
            "dtype",
            "min",
            "max",
            "mean",
            "std",
            "band_means",
            "band_stds",
        ],
    )
    write_csv(
        OUTPUT_DIR / "target_sample_stats.csv",
        geo["target_records"],
        ["location", "satellite", "path", "height", "width", "dtype", "min", "max", "mean", "std", "zero_fraction", "positive_fraction"],
    )
    write_csv(
        OUTPUT_DIR / "geotiff_metadata_samples.csv",
        geo["metadata_records"],
        [
            "split",
            "kind",
            "path",
            "width",
            "height",
            "samples_per_pixel",
            "bits_per_sample",
            "sample_format",
            "compression",
            "rows_per_strip",
            "strip_count",
            "dtype",
        ],
    )

    write_report(
        {
            "summaries": summaries,
            "path_validation": path_validation,
            "geotiff_sample": geo,
            "plots": plots,
        }
    )

    print(f"Wrote EDA outputs to {rel(OUTPUT_DIR)}")
    print(f"Train rows: {len(train_rows)} | Evaluation rows: {len(eval_rows)}")
    print(f"Path validation: {path_validation}")
    if geo["errors"]:
        print(f"GeoTIFF sample errors: {len(geo['errors'])}")


if __name__ == "__main__":
    main()
