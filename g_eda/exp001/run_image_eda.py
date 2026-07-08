#!/usr/bin/env python3
"""Image-processing EDA for Solafune precipitation nowcasting.

This script intentionally produces simple, inspectable artifacts: CSVs, PNGs, and a short Markdown
report. OpenCV is used when available, with NumPy fallbacks for the few operations we need.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import re
import struct
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/solafune_precipitation_nowcasting_mplconfig")

import numpy as np
import yaml

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - depends on the container.
    cv2 = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - report can still be written without figures.
    plt = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SATELLITES = ("goes", "himawari", "meteosat")

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
    273: "strip_offsets",
    277: "samples_per_pixel",
    278: "rows_per_strip",
    279: "strip_byte_counts",
    339: "sample_format",
}


@dataclass(frozen=True)
class TiffMeta:
    width: int
    height: int
    samples_per_pixel: int
    bits_per_sample: tuple[int, ...]
    sample_format: tuple[int, ...]
    compression: int
    rows_per_strip: int
    strip_offsets: tuple[int, ...]
    strip_byte_counts: tuple[int, ...]
    dtype: str
    endian: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_obs(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


def observation_time(filename: str) -> Any | None:
    match = re.search(r"_(\d{8})_(\d{4})\.tif$", filename)
    if not match:
        return None
    return np.datetime64(
        f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:8]}T"
        f"{match.group(2)[:2]}:{match.group(2)[2:4]}"
    )


def _raw_value(data: bytes, endian: str, typ: int, count: int, raw: bytes) -> Any:
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


def _as_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    return (int(value),)


def _dtype_from_tags(bits_per_sample: tuple[int, ...], sample_format: tuple[int, ...]) -> np.dtype:
    bits = bits_per_sample[0]
    fmt = sample_format[0] if sample_format else 1
    if fmt == 1 and bits == 8:
        return np.dtype("uint8")
    if fmt == 3 and bits == 32:
        return np.dtype("float32")
    raise ValueError(f"Unsupported TIFF dtype sample_format={fmt}, bits={bits}")


def read_tiff_tags(path: Path) -> tuple[bytes, TiffMeta]:
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
            tags[name] = _raw_value(data, endian, typ, count, raw)
    bits = _as_tuple(tags["bits_per_sample"])
    sample_format = _as_tuple(tags.get("sample_format", 1))
    dtype = _dtype_from_tags(bits, sample_format)
    return data, TiffMeta(
        width=int(tags["width"]),
        height=int(tags["height"]),
        samples_per_pixel=int(tags.get("samples_per_pixel", 1)),
        bits_per_sample=bits,
        sample_format=sample_format,
        compression=int(tags.get("compression", 1)),
        rows_per_strip=int(tags.get("rows_per_strip", tags["height"])),
        strip_offsets=_as_tuple(tags["strip_offsets"]),
        strip_byte_counts=_as_tuple(tags["strip_byte_counts"]),
        dtype=str(dtype),
        endian=endian,
    )


def read_tiff_array(path: Path) -> np.ndarray:
    data, meta = read_tiff_tags(path)
    dtype = np.dtype(meta.dtype)
    chunks: list[bytes] = []
    for offset, byte_count in zip(meta.strip_offsets, meta.strip_byte_counts):
        chunk = data[offset : offset + byte_count]
        if meta.compression == 1:
            chunks.append(chunk)
        elif meta.compression == 8:
            chunks.append(zlib.decompress(chunk))
        else:
            raise ValueError(f"Unsupported TIFF compression={meta.compression}: {path}")
    arr = np.frombuffer(b"".join(chunks), dtype=dtype.newbyteorder(meta.endian))
    expected = meta.width * meta.height * meta.samples_per_pixel
    if arr.size != expected:
        raise ValueError(f"Unexpected TIFF size {arr.size} != {expected}: {path}")
    if meta.samples_per_pixel == 1:
        return arr.reshape(meta.height, meta.width).astype(dtype, copy=False)
    return arr.reshape(meta.height, meta.width, meta.samples_per_pixel).astype(dtype, copy=False)


def resize_to_target(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    dst_h, dst_w = size
    if arr.shape[:2] == (dst_h, dst_w):
        return arr.astype(np.float32, copy=False)
    if cv2 is not None:
        return cv2.resize(arr.astype(np.float32), (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    src_h, src_w = arr.shape[:2]
    y_idx = np.minimum((np.arange(dst_h) * src_h / dst_h).astype(int), src_h - 1)
    x_idx = np.minimum((np.arange(dst_w) * src_w / dst_w).astype(int), src_w - 1)
    return arr[np.ix_(y_idx, x_idx)].astype(np.float32)


def sobel_energy(arr: np.ndarray) -> float:
    arr = arr.astype(np.float32)
    if cv2 is not None:
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.sqrt(gx * gx + gy * gy).mean())
    gy, gx = np.gradient(arr)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def laplacian_energy(arr: np.ndarray) -> float:
    arr = arr.astype(np.float32)
    if cv2 is not None:
        return float(np.abs(cv2.Laplacian(arr, cv2.CV_32F)).mean())
    center = arr[1:-1, 1:-1] * -4
    lap = center + arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:]
    return float(np.abs(lap).mean())


def connected_component_stats(mask: np.ndarray) -> dict[str, float]:
    mask_u8 = mask.astype(np.uint8)
    if cv2 is not None:
        n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) if n_labels > 1 else np.array([], dtype=np.float32)
    else:
        areas = flood_fill_areas(mask_u8)
    return {
        "component_count": float(len(areas)),
        "largest_component_area": float(areas.max()) if len(areas) else 0.0,
        "mean_component_area": float(areas.mean()) if len(areas) else 0.0,
    }


def flood_fill_areas(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros_like(mask, dtype=bool)
    areas: list[int] = []
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if visited[y, x] or not mask[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            areas.append(area)
    return np.asarray(areas, dtype=np.float32)


def corr_at_shift(a: np.ndarray, b: np.ndarray, dy: int, dx: int) -> float:
    if dy >= 0:
        ay = slice(dy, None)
        by = slice(0, b.shape[0] - dy)
    else:
        ay = slice(0, a.shape[0] + dy)
        by = slice(-dy, None)
    if dx >= 0:
        ax = slice(dx, None)
        bx = slice(0, b.shape[1] - dx)
    else:
        ax = slice(0, a.shape[1] + dx)
        bx = slice(-dx, None)
    aa = a[ay, ax].reshape(-1)
    bb = b[by, bx].reshape(-1)
    if aa.size < 8 or aa.std() == 0 or bb.std() == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def best_shift(a: np.ndarray, b: np.ndarray, radius: int = 6) -> tuple[int, int, float, float]:
    base = corr_at_shift(a, b, 0, 0)
    best = (0, 0, -np.inf)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            score = corr_at_shift(a, b, dy, dx)
            if np.isfinite(score) and score > best[2]:
                best = (dy, dx, score)
    return best[0], best[1], float(best[2]), float(base if np.isfinite(base) else np.nan)


def phase_displacement(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if cv2 is not None:
        shift, response = cv2.phaseCorrelate(a, b)
        return float(shift[1]), float(shift[0]), float(response)
    aa = a - a.mean()
    bb = b - b.mean()
    fa = np.fft.fft2(aa)
    fb = np.fft.fft2(bb)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-9)
    corr = np.abs(np.fft.ifft2(cross))
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    if y > a.shape[0] // 2:
        y -= a.shape[0]
    if x > a.shape[1] // 2:
        x -= a.shape[1]
    return float(y), float(x), float(corr.max())


def sample_rows(rows: list[dict[str, str]], cfg: dict[str, Any], key: str) -> list[dict[str, str]]:
    max_rows = int(cfg["sampling"].get(key, 0) or 0)
    max_per_location = int(cfg["sampling"].get("max_rows_per_location", 0) or 0)
    seed = int(cfg["sampling"].get("random_seed", 42))
    rng = random.Random(seed)
    by_location: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_location[row["name_location"]].append(row)
    sampled: list[dict[str, str]] = []
    for location_rows in by_location.values():
        rows_copy = list(location_rows)
        rng.shuffle(rows_copy)
        sampled.extend(rows_copy[:max_per_location] if max_per_location else rows_copy)
    rng.shuffle(sampled)
    return sampled[:max_rows] if max_rows else sampled


def latest_observation(row: dict[str, str]) -> str | None:
    obs = parse_obs(row["last_30_minutes_observation_filename"])
    return obs[-1] if obs else None


def load_sat_band(data_dir: Path, row: dict[str, str], band: int, target_size: tuple[int, int]) -> np.ndarray | None:
    obs_name = latest_observation(row)
    if obs_name is None:
        return None
    sat = row["satellite_target"]
    path = data_dir / sat / obs_name
    if not path.exists():
        return None
    arr = read_tiff_array(path)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if band >= arr.shape[2]:
        return None
    band_arr = arr[:, :, band].astype(np.float32) / 255.0
    return resize_to_target(band_arr, target_size)


def load_target(train_dir: Path, row: dict[str, str]) -> np.ndarray:
    arr = read_tiff_array(train_dir / "gpm_imerg" / row["gpm_imerg_filename"])
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def analyze_target_morphology(rows: list[dict[str, str]], train_dir: Path, cfg: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    thresholds = [float(x) for x in cfg["thresholds"]["rain_mm"]]
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        target = load_target(train_dir, row)
        base = {
            "unique_id": row["unique_id"],
            "name_location": row["name_location"],
            "satellite_target": row["satellite_target"],
            "datetime": row["datetime"],
            "target_mean": float(target.mean()),
            "target_std": float(target.std()),
            "target_max": float(target.max()),
            "target_positive_ratio": float((target > 0).mean()),
            "sobel_energy": sobel_energy(target),
            "laplacian_energy": laplacian_energy(target),
        }
        for threshold in thresholds:
            mask = target >= threshold
            stats = connected_component_stats(mask)
            suffix = str(threshold).replace(".", "p")
            base[f"rain_ratio_ge_{suffix}"] = float(mask.mean())
            base[f"component_count_ge_{suffix}"] = stats["component_count"]
            base[f"largest_component_area_ge_{suffix}"] = stats["largest_component_area"]
            base[f"mean_component_area_ge_{suffix}"] = stats["mean_component_area"]
        output_rows.append(base)
    write_csv(out_dir / "target_morphology.csv", output_rows)
    return output_rows


def analyze_parallax(rows: list[dict[str, str]], train_dir: Path, cfg: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    target_size = (int(cfg["data"]["target_height"]), int(cfg["data"]["target_width"]))
    ir_bands = {k: int(v) for k, v in cfg["bands"]["ir_window"].items()}
    min_target_mean = float(cfg["sampling"].get("min_target_mean_for_shift", 0.2))
    per_sample: list[dict[str, Any]] = []
    for row in rows:
        target = load_target(train_dir, row)
        if float(target.mean()) < min_target_mean:
            continue
        sat = row["satellite_target"]
        band = load_sat_band(train_dir, row, ir_bands[sat], target_size)
        if band is None or band.std() == 0 or target.std() == 0:
            continue
        cold = -(band - band.mean())
        target_z = target - target.mean()
        dy, dx, shifted_corr, base_corr = best_shift(cold, target_z, radius=6)
        per_sample.append(
            {
                "unique_id": row["unique_id"],
                "name_location": row["name_location"],
                "satellite_target": sat,
                "datetime": row["datetime"],
                "target_mean": float(target.mean()),
                "target_max": float(target.max()),
                "best_dy": dy,
                "best_dx": dx,
                "base_corr": base_corr,
                "best_corr": shifted_corr,
                "corr_gain": shifted_corr - base_corr if np.isfinite(base_corr) else np.nan,
            }
        )
    write_csv(out_dir / "parallax_shift_samples.csv", per_sample)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample:
        grouped[(row["name_location"], row["satellite_target"])].append(row)
    location_rows: list[dict[str, Any]] = []
    for (location, sat), values in sorted(grouped.items()):
        dys = np.asarray([float(v["best_dy"]) for v in values])
        dxs = np.asarray([float(v["best_dx"]) for v in values])
        gains = np.asarray([float(v["corr_gain"]) for v in values if np.isfinite(float(v["corr_gain"]))])
        location_rows.append(
            {
                "name_location": location,
                "satellite_target": sat,
                "samples": len(values),
                "median_dy": float(np.median(dys)),
                "median_dx": float(np.median(dxs)),
                "mean_dy": float(dys.mean()),
                "mean_dx": float(dxs.mean()),
                "mean_corr_gain": float(gains.mean()) if len(gains) else np.nan,
            }
        )
    write_csv(out_dir / "parallax_shift_by_location.csv", location_rows)
    return location_rows


def analyze_temporal_motion(rows: list[dict[str, str]], data_dir: Path, cfg: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    target_size = (int(cfg["data"]["target_height"]), int(cfg["data"]["target_width"]))
    ir_bands = {k: int(v) for k, v in cfg["bands"]["ir_window"].items()}
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        obs = parse_obs(row["last_30_minutes_observation_filename"])
        if len(obs) < 2:
            continue
        sat = row["satellite_target"]
        frames = []
        times = []
        for obs_name in obs[-3:]:
            path = data_dir / sat / obs_name
            if not path.exists():
                continue
            arr = read_tiff_array(path)
            if arr.ndim == 2 or ir_bands[sat] >= arr.shape[2]:
                continue
            frames.append(resize_to_target(arr[:, :, ir_bands[sat]].astype(np.float32) / 255.0, target_size))
            times.append(observation_time(obs_name))
        if len(frames) < 2:
            continue
        dy, dx, response = phase_displacement(frames[0], frames[-1])
        minutes = np.nan
        if times[0] is not None and times[-1] is not None:
            minutes = float((times[-1] - times[0]) / np.timedelta64(1, "m"))
        output_rows.append(
            {
                "unique_id": row["unique_id"],
                "name_location": row["name_location"],
                "satellite_target": sat,
                "datetime": row["datetime"],
                "frame_count": len(frames),
                "minutes_between": minutes,
                "phase_dy": dy,
                "phase_dx": dx,
                "phase_magnitude": float(math.sqrt(dy * dy + dx * dx)),
                "phase_response": response,
            }
        )
    write_csv(out_dir / "temporal_motion.csv", output_rows)
    return output_rows


def analyze_spectral_texture(rows: list[dict[str, str]], train_dir: Path, cfg: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    target_size = (int(cfg["data"]["target_height"]), int(cfg["data"]["target_width"]))
    feature_rows: list[dict[str, Any]] = []
    for row in rows:
        obs_name = latest_observation(row)
        if obs_name is None:
            continue
        sat = row["satellite_target"]
        path = train_dir / sat / obs_name
        if not path.exists():
            continue
        arr = read_tiff_array(path)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        target = load_target(train_dir, row)
        base: dict[str, Any] = {
            "unique_id": row["unique_id"],
            "name_location": row["name_location"],
            "satellite_target": sat,
            "datetime": row["datetime"],
            "target_mean": float(target.mean()),
            "target_max": float(target.max()),
            "target_positive_ratio": float((target > 0).mean()),
        }
        for band_idx in range(min(arr.shape[2], 16)):
            band = resize_to_target(arr[:, :, band_idx].astype(np.float32) / 255.0, target_size)
            base[f"band{band_idx:02d}_mean"] = float(band.mean())
            base[f"band{band_idx:02d}_std"] = float(band.std())
            base[f"band{band_idx:02d}_sobel"] = sobel_energy(band)
            base[f"band{band_idx:02d}_laplacian"] = laplacian_energy(band)
        feature_rows.append(base)
    write_csv(out_dir / "spectral_texture_features.csv", feature_rows)

    if not feature_rows:
        return []
    numeric_keys = [key for key in feature_rows[0] if key.startswith("band")]
    corr_rows: list[dict[str, Any]] = []
    target_values = {
        "target_mean": np.asarray([float(row["target_mean"]) for row in feature_rows]),
        "target_max": np.asarray([float(row["target_max"]) for row in feature_rows]),
        "target_positive_ratio": np.asarray([float(row["target_positive_ratio"]) for row in feature_rows]),
    }
    for key in numeric_keys:
        x = np.asarray([float(row.get(key, np.nan)) for row in feature_rows])
        if not np.isfinite(x).all() or x.std() == 0:
            continue
        for target_name, y in target_values.items():
            if y.std() == 0:
                continue
            corr_rows.append(
                {
                    "feature": key,
                    "target": target_name,
                    "pearson_corr": float(np.corrcoef(x, y)[0, 1]),
                    "abs_corr": float(abs(np.corrcoef(x, y)[0, 1])),
                }
            )
    corr_rows.sort(key=lambda row: row["abs_corr"], reverse=True)
    write_csv(out_dir / "spectral_texture_correlations.csv", corr_rows)
    return corr_rows


def row_feature_vector(row: dict[str, str], data_dir: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    target_size = (int(cfg["data"]["target_height"]), int(cfg["data"]["target_width"]))
    obs_name = latest_observation(row)
    if obs_name is None:
        return None
    sat = row["satellite_target"]
    path = data_dir / sat / obs_name
    if not path.exists():
        return None
    arr = read_tiff_array(path)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    out: dict[str, Any] = {"name_location": row["name_location"], "satellite_target": sat}
    for band_idx in range(min(arr.shape[2], 16)):
        band = resize_to_target(arr[:, :, band_idx].astype(np.float32) / 255.0, target_size)
        out[f"band{band_idx:02d}_mean"] = float(band.mean())
        out[f"band{band_idx:02d}_std"] = float(band.std())
    return out


def analyze_train_eval_shift(
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    train_dir: Path,
    eval_dir: Path,
    cfg: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    train_features = [x for row in train_rows if (x := row_feature_vector(row, train_dir, cfg)) is not None]
    eval_features = [x for row in eval_rows if (x := row_feature_vector(row, eval_dir, cfg)) is not None]
    if not train_features or not eval_features:
        return []
    keys = [key for key in train_features[0] if key.startswith("band")]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in train_features:
        grouped[("train", row["satellite_target"], row["name_location"])].append(row)
    for row in eval_features:
        grouped[("eval", row["satellite_target"], row["name_location"])].append(row)

    centroids: list[dict[str, Any]] = []
    for (split, sat, loc), values in grouped.items():
        centroid = {"split": split, "satellite_target": sat, "name_location": loc, "samples": len(values)}
        for key in keys:
            vals = [float(v.get(key, np.nan)) for v in values]
            centroid[key] = float(np.nanmean(vals))
        centroids.append(centroid)
    write_csv(out_dir / "train_eval_feature_centroids.csv", centroids)

    train_centroids = [row for row in centroids if row["split"] == "train"]
    eval_centroids = [row for row in centroids if row["split"] == "eval"]
    result_rows: list[dict[str, Any]] = []
    for ev in eval_centroids:
        candidates = [tr for tr in train_centroids if tr["satellite_target"] == ev["satellite_target"]]
        if not candidates:
            continue
        ev_vec = np.asarray([float(ev[key]) for key in keys], dtype=np.float32)
        distances = []
        for tr in candidates:
            tr_vec = np.asarray([float(tr[key]) for key in keys], dtype=np.float32)
            distances.append((float(np.sqrt(np.mean(np.square(ev_vec - tr_vec)))), tr))
        distances.sort(key=lambda x: x[0])
        nearest = distances[0][1]
        result_rows.append(
            {
                "eval_location": ev["name_location"],
                "satellite_target": ev["satellite_target"],
                "eval_samples": ev["samples"],
                "nearest_train_location": nearest["name_location"],
                "nearest_distance": distances[0][0],
                "median_train_distance": float(np.median([d for d, _ in distances])),
            }
        )
    write_csv(out_dir / "train_eval_feature_shift.csv", result_rows)
    return result_rows


def save_figures(out_dir: Path, morphology: list[dict[str, Any]], motion: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if morphology:
        means = np.asarray([float(row["target_mean"]) for row in morphology])
        maxes = np.asarray([float(row["target_max"]) for row in morphology])
        plt.figure(figsize=(8, 4))
        plt.scatter(means, maxes, s=4, alpha=0.35)
        plt.xlabel("target mean")
        plt.ylabel("target max")
        plt.title("Target mean vs peak precipitation")
        plt.tight_layout()
        plt.savefig(fig_dir / "target_mean_vs_max.png", dpi=140)
        plt.close()
    if motion:
        mags = np.asarray([float(row["phase_magnitude"]) for row in motion])
        plt.figure(figsize=(8, 4))
        plt.hist(mags[np.isfinite(mags)], bins=50)
        plt.xlabel("phase displacement magnitude (target-grid pixels)")
        plt.ylabel("samples")
        plt.title("Satellite-frame motion over available observation window")
        plt.tight_layout()
        plt.savefig(fig_dir / "temporal_motion_hist.png", dpi=140)
        plt.close()


def write_report(
    out_dir: Path,
    cfg: dict[str, Any],
    timings: dict[str, float],
    morphology: list[dict[str, Any]],
    parallax: list[dict[str, Any]],
    motion: list[dict[str, Any]],
    texture: list[dict[str, Any]],
    shift: list[dict[str, Any]],
) -> None:
    cv2_status = "available" if cv2 is not None else "not available, using NumPy fallbacks"
    top_texture = texture[:10]
    top_parallax = sorted(parallax, key=lambda row: abs(float(row.get("mean_corr_gain", 0.0))), reverse=True)[:10]
    report = [
        "# Image-Processing EDA Report",
        "",
        f"Generated by `g_eda/exp001/run_image_eda.py`.",
        "",
        f"- OpenCV: `{cv2_status}`",
        f"- Output dir: `{out_dir}`",
        f"- Timings: `{json.dumps(timings, sort_keys=True)}`",
        "",
        "## Produced Files",
        "",
        "- `target_morphology.csv`",
        "- `parallax_shift_samples.csv`",
        "- `parallax_shift_by_location.csv`",
        "- `temporal_motion.csv`",
        "- `spectral_texture_features.csv`",
        "- `spectral_texture_correlations.csv`",
        "- `train_eval_feature_centroids.csv`",
        "- `train_eval_feature_shift.csv`",
        "- `figures/*.png`",
        "",
        "## Quick Counts",
        "",
        f"- morphology rows: `{len(morphology)}`",
        f"- parallax location rows: `{len(parallax)}`",
        f"- temporal motion rows: `{len(motion)}`",
        f"- texture correlations: `{len(texture)}`",
        f"- train/eval shift rows: `{len(shift)}`",
        "",
        "## Top Parallax Corr Gains",
        "",
        "| location | satellite | samples | median_dy | median_dx | mean_corr_gain |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in top_parallax:
        report.append(
            f"| {row['name_location']} | {row['satellite_target']} | {row['samples']} | "
            f"{float(row['median_dy']):.2f} | {float(row['median_dx']):.2f} | "
            f"{float(row['mean_corr_gain']):.4f} |"
        )
    report.extend([
        "",
        "## Top Spectral/Texture Correlations",
        "",
        "| feature | target | pearson_corr |",
        "| --- | --- | ---: |",
    ])
    for row in top_texture:
        report.append(f"| {row['feature']} | {row['target']} | {float(row['pearson_corr']):.4f} |")
    report.extend([
        "",
        "## Notes",
        "",
        "- Large, stable parallax shifts suggest trying location/satellite-specific spatial correction.",
        "- Tiny phase displacement supports simple frame stacking, successor-frame features, or smoothing over heavy recurrent models.",
        "- Texture/rain correlations can suggest auxiliary image features or architecture attention points.",
        "- Train/eval feature distances can flag locations where calibration/thresholds may behave differently.",
    ])
    (out_dir / "EDA_IMAGE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    output_dir = resolve_path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = resolve_path(cfg["data"]["train_dir"])
    eval_dir = resolve_path(cfg["data"]["evaluation_dir"])
    train_rows_all = read_csv_rows(resolve_path(cfg["data"]["train_csv"]))
    eval_rows_all = read_csv_rows(resolve_path(cfg["data"]["evaluation_csv"]))
    train_rows = sample_rows(train_rows_all, cfg, "max_train_rows")
    eval_rows = sample_rows(eval_rows_all, cfg, "max_eval_rows")

    timings: dict[str, float] = {}
    morphology: list[dict[str, Any]] = []
    parallax: list[dict[str, Any]] = []
    motion: list[dict[str, Any]] = []
    texture: list[dict[str, Any]] = []
    shift: list[dict[str, Any]] = []

    analyses = cfg.get("analyses", {})
    if analyses.get("target_morphology", True):
        start = time.time()
        morphology = analyze_target_morphology(train_rows, train_dir, cfg, output_dir)
        timings["target_morphology"] = time.time() - start
    if analyses.get("parallax_shift", True):
        start = time.time()
        parallax = analyze_parallax(train_rows, train_dir, cfg, output_dir)
        timings["parallax_shift"] = time.time() - start
    if analyses.get("temporal_motion", True):
        start = time.time()
        motion = analyze_temporal_motion(train_rows, train_dir, cfg, output_dir)
        timings["temporal_motion_train"] = time.time() - start
    if analyses.get("spectral_texture", True):
        start = time.time()
        texture = analyze_spectral_texture(train_rows, train_dir, cfg, output_dir)
        timings["spectral_texture"] = time.time() - start
    if analyses.get("train_eval_shift", True):
        start = time.time()
        shift = analyze_train_eval_shift(train_rows, eval_rows, train_dir, eval_dir, cfg, output_dir)
        timings["train_eval_shift"] = time.time() - start

    save_figures(output_dir, morphology, motion)
    write_report(output_dir, cfg, timings, morphology, parallax, motion, texture, shift)
    summary = {
        "train_rows_sampled": len(train_rows),
        "eval_rows_sampled": len(eval_rows),
        "opencv_available": cv2 is not None,
        "timings": timings,
        "outputs": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
