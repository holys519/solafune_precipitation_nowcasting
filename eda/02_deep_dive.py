#!/usr/bin/env python3
"""Additional EDA driven by findings from 01_data_overview.py."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/solafune_precipitation_nowcasting_mplconfig")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDA01_PATH = PROJECT_ROOT / "eda" / "01_data_overview.py"
spec = importlib.util.spec_from_file_location("eda01", EDA01_PATH)
eda01 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eda01"] = eda01
spec.loader.exec_module(eda01)

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train_dataset"
EVAL_DIR = DATA_DIR / "evaluation_dataset"
OUTPUT_DIR = PROJECT_ROOT / "eda" / "outputs"

THRESHOLDS = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]


@dataclass
class PixelStats:
    files: int = 0
    pixels: int = 0
    finite_pixels: int = 0
    zero_pixels: int = 0
    positive_pixels: int = 0
    sum_value: float = 0.0
    sumsq_value: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf
    threshold_counts: dict[float, int] = field(default_factory=lambda: {t: 0 for t in THRESHOLDS})

    def update(self, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)]
        self.files += 1
        self.pixels += values.size
        self.finite_pixels += finite.size
        if finite.size == 0:
            return
        self.zero_pixels += int((finite == 0).sum())
        self.positive_pixels += int((finite > 0).sum())
        self.sum_value += float(finite.sum(dtype="float64"))
        self.sumsq_value += float(np.square(finite, dtype="float64").sum(dtype="float64"))
        self.min_value = min(self.min_value, float(finite.min()))
        self.max_value = max(self.max_value, float(finite.max()))
        for threshold in THRESHOLDS:
            self.threshold_counts[threshold] += int((finite > threshold).sum())

    def as_row(self, **keys: Any) -> dict[str, Any]:
        mean = self.sum_value / self.finite_pixels if self.finite_pixels else math.nan
        variance = self.sumsq_value / self.finite_pixels - mean * mean if self.finite_pixels else math.nan
        std = math.sqrt(max(variance, 0.0)) if self.finite_pixels else math.nan
        row = {
            **keys,
            "files": self.files,
            "pixels": self.pixels,
            "finite_pixels": self.finite_pixels,
            "mean": mean,
            "std": std,
            "min": self.min_value if self.finite_pixels else math.nan,
            "max": self.max_value if self.finite_pixels else math.nan,
            "zero_fraction": self.zero_pixels / self.finite_pixels if self.finite_pixels else math.nan,
            "positive_fraction": self.positive_pixels / self.finite_pixels if self.finite_pixels else math.nan,
        }
        for threshold in THRESHOLDS:
            row[f"gt_{threshold:g}_fraction"] = (
                self.threshold_counts[threshold] / self.finite_pixels if self.finite_pixels else math.nan
            )
        return row


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
    277: "samples_per_pixel",
    278: "rows_per_strip",
    279: "strip_byte_counts",
    284: "planar_config",
    317: "predictor",
    339: "sample_format",
}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def tiff_light_value(f, endian: str, typ: int, count: int, raw: bytes) -> Any:
    fmt, size = TIFF_TYPE_INFO[typ]
    total = count * size
    if total <= 4:
        data = raw[:total]
    else:
        current = f.tell()
        offset = struct.unpack(endian + "I", raw)[0]
        f.seek(offset)
        data = f.read(total)
        f.seek(current)

    if typ == 2:
        return data.rstrip(b"\0").decode("utf-8", "replace")
    if typ == 5:
        values = []
        for i in range(count):
            numerator, denominator = struct.unpack(endian + "II", data[i * 8 : i * 8 + 8])
            values.append(numerator / denominator if denominator else math.nan)
        return values[0] if count == 1 else tuple(values)

    values = struct.unpack(endian + (fmt * count), data)
    return values[0] if count == 1 else tuple(values)


def read_tiff_meta_light(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        header = f.read(8)
        if header[:2] == b"II":
            endian = "<"
        elif header[:2] == b"MM":
            endian = ">"
        else:
            raise ValueError("not a tiff")
        magic = struct.unpack(endian + "H", header[2:4])[0]
        if magic != 42:
            raise ValueError(f"unsupported tiff magic {magic}")
        ifd_offset = struct.unpack(endian + "I", header[4:8])[0]
        f.seek(ifd_offset)
        tag_count = struct.unpack(endian + "H", f.read(2))[0]
        tags: dict[str, Any] = {}
        for _ in range(tag_count):
            entry = f.read(12)
            tag, typ, count = struct.unpack(endian + "HHI", entry[:8])
            raw = entry[8:12]
            if typ not in TIFF_TYPE_INFO:
                continue
            name = TIFF_TAG_NAMES.get(tag)
            if name:
                tags[name] = tiff_light_value(f, endian, typ, count, raw)

    bits = tags.get("bits_per_sample", 0)
    if isinstance(bits, tuple):
        bit_key = ",".join(str(x) for x in bits)
        bit0 = int(bits[0])
    else:
        bit_key = str(bits)
        bit0 = int(bits)
    fmt = tags.get("sample_format", 1)
    fmt0 = int(fmt[0] if isinstance(fmt, tuple) else fmt)
    if fmt0 == 1 and bit0 == 8:
        dtype = "uint8"
    elif fmt0 == 3 and bit0 == 32:
        dtype = "float32"
    else:
        dtype = f"sample_format={fmt0},bits={bit0}"

    return {
        "width": int(tags["width"]),
        "height": int(tags["height"]),
        "samples_per_pixel": int(tags.get("samples_per_pixel", 1)),
        "bits_per_sample": bit_key,
        "sample_format": ",".join(str(x) for x in fmt) if isinstance(fmt, tuple) else str(fmt),
        "compression": int(tags.get("compression", 1)),
        "rows_per_strip": int(tags.get("rows_per_strip", tags["height"])),
        "dtype": dtype,
    }


def adjusted_overall_percentiles(positive_values: np.ndarray, zero_count: int, total_count: int) -> dict[str, float]:
    result: dict[str, float] = {}
    if total_count == 0:
        return result
    positive_values.sort()
    for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:
        rank = q / 100 * (total_count - 1)
        if rank < zero_count or positive_values.size == 0:
            value = 0.0
        else:
            pos_rank = min(max(rank - zero_count, 0), positive_values.size - 1)
            lo = int(math.floor(pos_rank))
            hi = int(math.ceil(pos_rank))
            if lo == hi:
                value = float(positive_values[lo])
            else:
                frac = pos_rank - lo
                value = float(positive_values[lo] * (1 - frac) + positive_values[hi] * frac)
        result[f"p{q:g}"] = value
    return result


def target_full_eda(train_rows: list[dict[str, str]]) -> dict[str, Any]:
    overall = PixelStats()
    by_location: dict[str, PixelStats] = defaultdict(PixelStats)
    by_satellite: dict[str, PixelStats] = defaultdict(PixelStats)
    by_month: dict[str, PixelStats] = defaultdict(PixelStats)
    by_obs_len: dict[int, PixelStats] = defaultdict(PixelStats)
    file_rows: list[dict[str, Any]] = []
    positive_chunks: list[np.ndarray] = []
    target_meta_counts: Counter = Counter()

    for i, row in enumerate(train_rows, 1):
        arr, meta = eda01.read_tiff_array(TRAIN_DIR / "gpm_imerg" / row["gpm_imerg_filename"])
        values = arr.astype("float32", copy=False).ravel()
        finite = values[np.isfinite(values)]
        positive = finite[finite > 0]
        if positive.size:
            positive_chunks.append(positive.copy())

        dt = eda01.parse_dt(row["datetime"])
        month = dt.strftime("%Y-%m")
        obs_count = len(eda01.parse_obs(row["last_30_minutes_observation_filename"]))

        overall.update(values)
        by_location[row["name_location"]].update(values)
        by_satellite[row["satellite_target"]].update(values)
        by_month[month].update(values)
        by_obs_len[obs_count].update(values)
        target_meta_counts[
            (
                meta.width,
                meta.height,
                meta.samples_per_pixel,
                ",".join(str(x) for x in meta.bits_per_sample),
                ",".join(str(x) for x in meta.sample_format),
                meta.compression,
                meta.dtype,
            )
        ] += 1

        file_rows.append(
            {
                "unique_id": row["unique_id"],
                "name_location": row["name_location"],
                "satellite_target": row["satellite_target"],
                "datetime": row["datetime"],
                "month": month,
                "obs_count": obs_count,
                "gpm_imerg_filename": row["gpm_imerg_filename"],
                "mean": float(finite.mean()) if finite.size else math.nan,
                "std": float(finite.std()) if finite.size else math.nan,
                "min": float(finite.min()) if finite.size else math.nan,
                "max": float(finite.max()) if finite.size else math.nan,
                "sum": float(finite.sum(dtype="float64")) if finite.size else math.nan,
                "zero_fraction": float((finite == 0).mean()) if finite.size else math.nan,
                "positive_fraction": float((finite > 0).mean()) if finite.size else math.nan,
                "gt_1_fraction": float((finite > 1.0).mean()) if finite.size else math.nan,
                "gt_5_fraction": float((finite > 5.0).mean()) if finite.size else math.nan,
                "gt_10_fraction": float((finite > 10.0).mean()) if finite.size else math.nan,
                "gt_20_fraction": float((finite > 20.0).mean()) if finite.size else math.nan,
            }
        )

        if i % 10000 == 0:
            print(f"Processed target files: {i}/{len(train_rows)}")

    positive_values = np.concatenate(positive_chunks) if positive_chunks else np.array([], dtype="float32")
    overall_row = overall.as_row(scope="overall")
    overall_row["rmse_if_predict_zero"] = math.sqrt(overall.sumsq_value / overall.finite_pixels)
    overall_row["rmse_if_predict_global_mean"] = math.sqrt(
        max(overall.sumsq_value / overall.finite_pixels - overall_row["mean"] ** 2, 0.0)
    )
    overall_row["percentiles"] = adjusted_overall_percentiles(
        positive_values,
        zero_count=overall.zero_pixels,
        total_count=overall.finite_pixels,
    )
    overall_row["positive_percentiles"] = eda01.percentile_dict(positive_values.astype("float64"))

    target_meta_rows = [
        {
            "width": key[0],
            "height": key[1],
            "samples_per_pixel": key[2],
            "bits_per_sample": key[3],
            "sample_format": key[4],
            "compression": key[5],
            "dtype": key[6],
            "count": count,
        }
        for key, count in target_meta_counts.items()
    ]

    return {
        "overall": overall_row,
        "by_location": [stats.as_row(name_location=key) for key, stats in sorted(by_location.items())],
        "by_satellite": [stats.as_row(satellite_target=key) for key, stats in sorted(by_satellite.items())],
        "by_month": [stats.as_row(month=key) for key, stats in sorted(by_month.items())],
        "by_obs_len": [stats.as_row(obs_count=key) for key, stats in sorted(by_obs_len.items())],
        "file_rows": file_rows,
        "target_meta_rows": target_meta_rows,
        "top_by_max": sorted(file_rows, key=lambda x: x["max"], reverse=True)[:50],
        "top_by_mean": sorted(file_rows, key=lambda x: x["mean"], reverse=True)[:50],
    }


def observation_anomaly_eda(rows: list[dict[str, str]], split: str, target_file_rows: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    anomalies: list[dict[str, Any]] = []
    summary = Counter()
    by_location_sat = Counter()
    by_month = Counter()
    for row in rows:
        obs = eda01.parse_obs(row["last_30_minutes_observation_filename"])
        obs_count = len(obs)
        summary[obs_count] += 1
        if obs_count == 3:
            continue
        dt = eda01.parse_dt(row["datetime"])
        out = {
            "split": split,
            "unique_id": row["unique_id"],
            "name_location": row["name_location"],
            "satellite_target": row["satellite_target"],
            "datetime": row["datetime"],
            "month": dt.strftime("%Y-%m"),
            "obs_count": obs_count,
            "filenames": row["last_30_minutes_observation_filename"],
            "gpm_imerg_filename": row["gpm_imerg_filename"],
        }
        if target_file_rows and row["unique_id"] in target_file_rows:
            stats = target_file_rows[row["unique_id"]]
            out.update(
                {
                    "target_mean": stats["mean"],
                    "target_max": stats["max"],
                    "target_positive_fraction": stats["positive_fraction"],
                }
            )
        anomalies.append(out)
        by_location_sat[(row["name_location"], row["satellite_target"], obs_count)] += 1
        by_month[(dt.strftime("%Y-%m"), obs_count)] += 1

    return {
        "summary": [{"split": split, "obs_count": key, "count": value} for key, value in sorted(summary.items())],
        "anomalies": anomalies,
        "by_location_satellite": [
            {"split": split, "name_location": key[0], "satellite_target": key[1], "obs_count": key[2], "count": value}
            for key, value in sorted(by_location_sat.items())
        ],
        "by_month": [
            {"split": split, "month": key[0], "obs_count": key[1], "count": value}
            for key, value in sorted(by_month.items())
        ],
    }


def split_overlap_eda(train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]]) -> dict[str, Any]:
    train_locations = {row["name_location"] for row in train_rows}
    eval_locations = {row["name_location"] for row in eval_rows}
    train_loc_sat = {(row["name_location"], row["satellite_target"]) for row in train_rows}
    eval_loc_sat = {(row["name_location"], row["satellite_target"]) for row in eval_rows}

    train_count = Counter(row["name_location"] for row in train_rows)
    eval_count = Counter(row["name_location"] for row in eval_rows)
    rows = []
    for loc in sorted(train_locations | eval_locations):
        rows.append(
            {
                "name_location": loc,
                "train_count": train_count.get(loc, 0),
                "evaluation_count": eval_count.get(loc, 0),
                "in_train": loc in train_locations,
                "in_evaluation": loc in eval_locations,
            }
        )

    return {
        "train_locations": len(train_locations),
        "evaluation_locations": len(eval_locations),
        "overlap_locations": sorted(train_locations & eval_locations),
        "train_only_locations": sorted(train_locations - eval_locations),
        "evaluation_only_locations": sorted(eval_locations - train_locations),
        "overlap_location_satellite": sorted(f"{loc}|{sat}" for loc, sat in (train_loc_sat & eval_loc_sat)),
        "rows": rows,
    }


def satellite_metadata_full_scan() -> dict[str, Any]:
    counts: dict[tuple[Any, ...], dict[str, Any]] = {}
    errors: list[str] = []
    jobs = [
        ("train", "goes", TRAIN_DIR / "goes"),
        ("train", "himawari", TRAIN_DIR / "himawari"),
        ("train", "meteosat", TRAIN_DIR / "meteosat"),
        ("evaluation", "goes", EVAL_DIR / "goes"),
        ("evaluation", "himawari", EVAL_DIR / "himawari"),
        ("evaluation", "meteosat", EVAL_DIR / "meteosat"),
    ]
    for split, satellite, directory in jobs:
        files = sorted(directory.glob("*.tif"))
        for i, path in enumerate(files, 1):
            try:
                meta = read_tiff_meta_light(path)
                key = (
                    split,
                    satellite,
                    meta["width"],
                    meta["height"],
                    meta["samples_per_pixel"],
                    meta["bits_per_sample"],
                    meta["sample_format"],
                    meta["compression"],
                    meta["rows_per_strip"],
                    meta["dtype"],
                )
                stat = counts.setdefault(
                    key,
                    {
                        "split": split,
                        "satellite": satellite,
                        "width": meta["width"],
                        "height": meta["height"],
                        "samples_per_pixel": meta["samples_per_pixel"],
                        "bits_per_sample": meta["bits_per_sample"],
                        "sample_format": meta["sample_format"],
                        "compression": meta["compression"],
                        "rows_per_strip": meta["rows_per_strip"],
                        "dtype": meta["dtype"],
                        "count": 0,
                        "total_size_bytes": 0,
                        "min_size_bytes": None,
                        "max_size_bytes": None,
                    },
                )
                size = path.stat().st_size
                stat["count"] += 1
                stat["total_size_bytes"] += size
                stat["min_size_bytes"] = size if stat["min_size_bytes"] is None else min(stat["min_size_bytes"], size)
                stat["max_size_bytes"] = size if stat["max_size_bytes"] is None else max(stat["max_size_bytes"], size)
            except Exception as exc:
                errors.append(f"{eda01.rel(path)}: {type(exc).__name__}: {exc}")
            if i % 30000 == 0:
                print(f"Scanned satellite metadata: {split}/{satellite} {i}/{len(files)}")

    rows = []
    for stat in counts.values():
        row = dict(stat)
        row["mean_size_bytes"] = row["total_size_bytes"] / row["count"]
        rows.append(row)
    rows.sort(key=lambda x: (x["split"], x["satellite"], x["height"], x["width"]))
    return {"rows": rows, "errors": errors}


def make_plots(target: dict[str, Any], obs_train: dict[str, Any], obs_eval: dict[str, Any]) -> list[str]:
    plot_paths: list[str] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        path = OUTPUT_DIR / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths.append(eda01.rel(path))

    by_loc = sorted(target["by_location"], key=lambda r: r["mean"], reverse=True)
    plt.figure(figsize=(9, 6))
    plt.barh([r["name_location"] for r in reversed(by_loc)], [r["mean"] for r in reversed(by_loc)], color="#4C78A8")
    plt.title("Full target mean by train location")
    plt.xlabel("mean precipitation")
    save("target_mean_by_location_full.png")

    by_sat = target["by_satellite"]
    plt.figure(figsize=(6, 4))
    plt.bar([r["satellite_target"] for r in by_sat], [r["mean"] for r in by_sat], color="#F58518")
    plt.title("Full target mean by satellite")
    plt.ylabel("mean precipitation")
    save("target_mean_by_satellite_full.png")

    by_month = target["by_month"]
    plt.figure(figsize=(10, 4))
    plt.plot([r["month"] for r in by_month], [r["mean"] for r in by_month], marker="o")
    plt.title("Full target mean by month")
    plt.ylabel("mean precipitation")
    plt.xticks(rotation=60, ha="right")
    save("target_mean_by_month_full.png")

    obs_rows = obs_train["summary"] + obs_eval["summary"]
    labels = sorted({r["obs_count"] for r in obs_rows})
    x = np.arange(len(labels))
    width = 0.35
    train_counts = [next((r["count"] for r in obs_train["summary"] if r["obs_count"] == label), 0) for label in labels]
    eval_counts = [next((r["count"] for r in obs_eval["summary"] if r["obs_count"] == label), 0) for label in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, train_counts, width, label="train")
    plt.bar(x + width / 2, eval_counts, width, label="evaluation")
    plt.yscale("log")
    plt.xticks(x, labels)
    plt.xlabel("observation file count")
    plt.ylabel("rows (log)")
    plt.title("Observation count per row")
    plt.legend()
    save("observation_count_rows_log.png")

    return plot_paths


def write_report(
    target: dict[str, Any],
    obs_train: dict[str, Any],
    obs_eval: dict[str, Any],
    overlap: dict[str, Any],
    satellite_meta: dict[str, Any],
    plots: list[str],
) -> None:
    overall = target["overall"]
    top_max = target["top_by_max"][:10]
    top_mean = target["top_by_mean"][:10]
    report = f"""# EDA Deep Dive

Generated by `eda/02_deep_dive.py`.

## Why This Pass

The first EDA found zero-inflated targets, rows with fewer than three observations, and disjoint-looking train/evaluation locations. This pass checks those issues more directly.

## Full Target Distribution

All train GPM-IMERG files were read.

| Metric | Value |
| --- | ---: |
| files | {overall['files']} |
| pixels | {overall['finite_pixels']} |
| mean | {overall['mean']} |
| std | {overall['std']} |
| min | {overall['min']} |
| max | {overall['max']} |
| zero fraction | {overall['zero_fraction']} |
| positive fraction | {overall['positive_fraction']} |
| RMSE if predict zero | {overall['rmse_if_predict_zero']} |
| RMSE if predict global mean | {overall['rmse_if_predict_global_mean']} |

Overall percentiles:

{markdown_table([{'percentile': k, 'value': v} for k, v in overall['percentiles'].items()], ['percentile', 'value'])}

Positive-pixel percentiles:

{markdown_table([{'percentile': k, 'value': v} for k, v in overall['positive_percentiles'].items()], ['percentile', 'value'])}

## Target By Satellite

{markdown_table(target['by_satellite'], ['satellite_target', 'files', 'mean', 'std', 'max', 'zero_fraction', 'positive_fraction', 'gt_5_fraction', 'gt_10_fraction'])}

## Target By Observation Count

{markdown_table(target['by_obs_len'], ['obs_count', 'files', 'mean', 'std', 'max', 'zero_fraction', 'positive_fraction'])}

## Rows With Fewer Than 3 Observations

{markdown_table(obs_train['summary'], ['split', 'obs_count', 'count'])}

{markdown_table(obs_eval['summary'], ['split', 'obs_count', 'count'])}

Train anomaly rows include target stats in `observation_anomalies_train.csv`; evaluation anomalies are in `observation_anomalies_evaluation.csv`.

## Train/Evaluation Location Overlap

| Metric | Value |
| --- | ---: |
| train locations | {overlap['train_locations']} |
| evaluation locations | {overlap['evaluation_locations']} |
| overlapping locations | {len(overlap['overlap_locations'])} |
| overlapping location-satellite pairs | {len(overlap['overlap_location_satellite'])} |

Train-only locations: `{', '.join(overlap['train_only_locations'])}`

Evaluation-only locations: `{', '.join(overlap['evaluation_only_locations'])}`

## Satellite Metadata Full Scan

{markdown_table(satellite_meta['rows'], ['split', 'satellite', 'count', 'height', 'width', 'samples_per_pixel', 'bits_per_sample', 'sample_format', 'compression', 'rows_per_strip', 'dtype', 'mean_size_bytes'])}

Metadata scan errors: {len(satellite_meta['errors'])}

## Top Target Files By Max

{markdown_table(top_max, ['unique_id', 'name_location', 'satellite_target', 'datetime', 'obs_count', 'max', 'mean', 'positive_fraction', 'gpm_imerg_filename'])}

## Top Target Files By Mean

{markdown_table(top_mean, ['unique_id', 'name_location', 'satellite_target', 'datetime', 'obs_count', 'max', 'mean', 'positive_fraction', 'gpm_imerg_filename'])}

## Plots

{chr(10).join(f'- `{path}`' for path in plots)}

## Modeling Implications

- Train and evaluation locations have no overlap, so random row CV will be optimistic. Use location holdout or satellite-aware location holdout.
- A zero predictor has a strong baseline because more than half of all pixels are zero; report this baseline with every CV.
- Rows with 0 observations exist in both train and evaluation. The model path must support empty observation lists, likely through fallback climatology/location-satellite priors or a learned missing-input token.
- Target tails are heavy. Track RMSE on positive pixels and high-rain thresholds in addition to global RMSE during validation.
"""
    (OUTPUT_DIR / "EDA_DEEP_DIVE.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = eda01.read_csv_rows(TRAIN_DIR / "train_dataset.csv")
    eval_rows = eda01.read_csv_rows(EVAL_DIR / "evaluation_target.csv")

    target = target_full_eda(train_rows)
    target_by_uid = {row["unique_id"]: row for row in target["file_rows"]}
    obs_train = observation_anomaly_eda(train_rows, "train", target_by_uid)
    obs_eval = observation_anomaly_eda(eval_rows, "evaluation", None)
    overlap = split_overlap_eda(train_rows, eval_rows)
    satellite_meta = satellite_metadata_full_scan()
    plots = make_plots(target, obs_train, obs_eval)

    write_csv(OUTPUT_DIR / "target_full_file_stats.csv", target["file_rows"], list(target["file_rows"][0].keys()))
    write_csv(OUTPUT_DIR / "target_stats_by_location.csv", target["by_location"], list(target["by_location"][0].keys()))
    write_csv(OUTPUT_DIR / "target_stats_by_satellite.csv", target["by_satellite"], list(target["by_satellite"][0].keys()))
    write_csv(OUTPUT_DIR / "target_stats_by_month.csv", target["by_month"], list(target["by_month"][0].keys()))
    write_csv(OUTPUT_DIR / "target_stats_by_obs_count.csv", target["by_obs_len"], list(target["by_obs_len"][0].keys()))
    write_csv(OUTPUT_DIR / "target_top_by_max.csv", target["top_by_max"], list(target["top_by_max"][0].keys()))
    write_csv(OUTPUT_DIR / "target_top_by_mean.csv", target["top_by_mean"], list(target["top_by_mean"][0].keys()))
    write_csv(OUTPUT_DIR / "target_metadata_full_counts.csv", target["target_meta_rows"], list(target["target_meta_rows"][0].keys()))
    write_csv(OUTPUT_DIR / "observation_count_summary.csv", obs_train["summary"] + obs_eval["summary"], ["split", "obs_count", "count"])
    write_csv(OUTPUT_DIR / "observation_anomalies_train.csv", obs_train["anomalies"], list(obs_train["anomalies"][0].keys()))
    write_csv(OUTPUT_DIR / "observation_anomalies_evaluation.csv", obs_eval["anomalies"], list(obs_eval["anomalies"][0].keys()))
    write_csv(
        OUTPUT_DIR / "observation_anomalies_by_location_satellite.csv",
        obs_train["by_location_satellite"] + obs_eval["by_location_satellite"],
        ["split", "name_location", "satellite_target", "obs_count", "count"],
    )
    write_csv(
        OUTPUT_DIR / "observation_anomalies_by_month.csv",
        obs_train["by_month"] + obs_eval["by_month"],
        ["split", "month", "obs_count", "count"],
    )
    write_csv(OUTPUT_DIR / "split_location_comparison.csv", overlap["rows"], list(overlap["rows"][0].keys()))
    write_csv(
        OUTPUT_DIR / "satellite_metadata_full_counts.csv",
        satellite_meta["rows"],
        [
            "split",
            "satellite",
            "width",
            "height",
            "samples_per_pixel",
            "bits_per_sample",
            "sample_format",
            "compression",
            "rows_per_strip",
            "dtype",
            "count",
            "total_size_bytes",
            "min_size_bytes",
            "max_size_bytes",
            "mean_size_bytes",
        ],
    )

    summary = {
        "target_overall": target["overall"],
        "target_by_satellite": target["by_satellite"],
        "target_by_obs_count": target["by_obs_len"],
        "observation_count_train": obs_train["summary"],
        "observation_count_evaluation": obs_eval["summary"],
        "split_overlap": {
            key: value for key, value in overlap.items() if key != "rows"
        },
        "satellite_metadata_errors": satellite_meta["errors"],
        "plots": plots,
    }
    (OUTPUT_DIR / "eda_deep_dive_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(target, obs_train, obs_eval, overlap, satellite_meta, plots)

    print("Wrote additional EDA outputs to eda/outputs")
    print(f"Target files: {target['overall']['files']}, pixels: {target['overall']['finite_pixels']}")
    print(f"Location overlap: {len(overlap['overlap_locations'])}")
    print(f"Satellite metadata scan errors: {len(satellite_meta['errors'])}")


if __name__ == "__main__":
    main()
