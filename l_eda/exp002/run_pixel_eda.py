#!/usr/bin/env python3
"""Pixel-level EDA round 2 for Solafune precipitation nowcasting.

Follow-up to l_eda/exp001 and doc/data_characteristics_review.md. Each analysis is
config-gated and answers one modeling decision:

- train_eval_adjacency : do paired train/eval tiles spatially overlap? (FFT normalized
  cross-correlation at shared timestamps; consistent peak offset across timestamps = overlap.)
- ir_rain_lag          : predictive information of IR frames at row offsets -3..+3 vs rain at T
  (how wide should data.context_offsets go — MetNet/Weather4cast-style temporal context).
- target_autocorr      : GPM lag autocorrelation + persistence RMSE per location (ceiling for
  temporal smoothing; PySTEPS persistence-baseline convention).
- bt_rain_response     : E[rain | IR value] / P(rain>0 | IR value) curves per satellite
  (IR-QPE power-law literature, PERSIANN; informs input transform and loss shaping).
- band_health          : per-band exact-0 / 255-saturation fraction and variance by UTC hour
  (grounds the solar-zenith feature idea; audits exp010's IR-zero-as-nodata assumption).
- target_quantization  : is GPM quantized to a discrete value grid? (possible snapping
  post-processing; IMERG products are quantized upstream).
- spectrum             : radially averaged power spectrum of rainy targets vs IR inputs
  (which spatial scales carry signal — the DGMR critique of MSE blur, groundwork for A-3).
- position_bias        : per-pixel rain climatology across tiles (does tile geometry leak a
  positional prior worth encoding, or is the field homogeneous?).
- oof_join             : join exp009 OOF per-sample metrics with visible-band brightness and
  IR-zero fraction (A-2: is the night-time error bump sensor information loss or diurnal
  convection?).

Self-contained: stdlib + NumPy (+ optional OpenCV/matplotlib), same conventions as l_eda/exp001.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import struct
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/solafune_precipitation_nowcasting_mplconfig")

import numpy as np
import yaml

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

SCRIPT_DIR = Path(__file__).resolve().parent
SATELLITES = ("goes", "himawari", "meteosat")

TIFF_TYPE_INFO = {1: ("B", 1), 2: ("c", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8), 11: ("f", 4), 12: ("d", 8)}
TIFF_TAG_NAMES = {
    256: "width", 257: "height", 258: "bits_per_sample", 259: "compression", 273: "strip_offsets",
    277: "samples_per_pixel", 278: "rows_per_strip", 279: "strip_byte_counts", 339: "sample_format",
}


@dataclass(frozen=True)
class TiffMeta:
    width: int
    height: int
    samples_per_pixel: int
    compression: int
    strip_offsets: tuple[int, ...]
    strip_byte_counts: tuple[int, ...]
    dtype: str
    endian: str


def _raw_value(data: bytes, endian: str, typ: int, count: int, raw: bytes) -> Any:
    fmt, size = TIFF_TYPE_INFO[typ]
    total = count * size
    buf = raw[:total] if total <= 4 else data[struct.unpack(endian + "I", raw)[0] :][:total]
    if typ == 2:
        return buf.rstrip(b"\0").decode("utf-8", "replace")
    values = struct.unpack(endian + (fmt * count), buf)
    return values[0] if count == 1 else tuple(values)


def _as_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in value) if isinstance(value, tuple) else (int(value),)


def read_tiff_array(path: Path) -> np.ndarray:
    data = path.read_bytes()
    endian = "<" if data[:2] == b"II" else ">"
    ifd = struct.unpack(endian + "I", data[4:8])[0]
    n_tags = struct.unpack(endian + "H", data[ifd : ifd + 2])[0]
    tags: dict[str, Any] = {}
    for i in range(n_tags):
        off = ifd + 2 + i * 12
        tag, typ, count = struct.unpack(endian + "HHI", data[off : off + 8])
        if typ in TIFF_TYPE_INFO and tag in TIFF_TAG_NAMES:
            tags[TIFF_TAG_NAMES[tag]] = _raw_value(data, endian, typ, count, data[off + 8 : off + 12])
    bits = _as_tuple(tags["bits_per_sample"])[0]
    fmt = _as_tuple(tags.get("sample_format", 1))[0]
    if fmt == 1 and bits == 8:
        dtype = np.dtype("uint8")
    elif fmt == 3 and bits == 32:
        dtype = np.dtype("float32")
    else:
        raise ValueError(f"Unsupported TIFF dtype fmt={fmt} bits={bits}: {path}")
    chunks: list[bytes] = []
    comp = int(tags.get("compression", 1))
    for off, cnt in zip(_as_tuple(tags["strip_offsets"]), _as_tuple(tags["strip_byte_counts"])):
        chunk = data[off : off + cnt]
        chunks.append(chunk if comp == 1 else zlib.decompress(chunk))
    arr = np.frombuffer(b"".join(chunks), dtype=dtype.newbyteorder(endian))
    h, w, spp = int(tags["height"]), int(tags["width"]), int(tags.get("samples_per_pixel", 1))
    if arr.size != h * w * spp:
        raise ValueError(f"TIFF size mismatch: {path}")
    return arr.reshape(h, w) if spp == 1 else arr.reshape(h, w, spp)


def resize_to(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    dst_h, dst_w = size
    if arr.shape[:2] == (dst_h, dst_w):
        return arr.astype(np.float32, copy=False)
    if cv2 is not None:
        return cv2.resize(arr.astype(np.float32), (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    src_h, src_w = arr.shape[:2]
    y = np.minimum((np.arange(dst_h) * src_h / dst_h).astype(int), src_h - 1)
    x = np.minimum((np.arange(dst_w) * src_w / dst_w).astype(int), src_w - 1)
    return arr[np.ix_(y, x)].astype(np.float32)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()


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


def last_obs_path(base_dir: Path, row: dict[str, str]) -> Path | None:
    obs = parse_obs(row["last_30_minutes_observation_filename"])
    if not obs:
        return None
    return base_dir / row["satellite_target"] / obs[-1]


def load_ir_band(base_dir: Path, row: dict[str, str], ir_band: int, size: tuple[int, int] | None) -> np.ndarray | None:
    path = last_obs_path(base_dir, row)
    if path is None or not path.exists():
        return None
    arr = read_tiff_array(path)
    if arr.ndim == 2:
        band = arr
    else:
        band = arr[:, :, min(ir_band, arr.shape[2] - 1)]
    band = band.astype(np.float32)
    return resize_to(band, size) if size is not None else band


def load_target(base_dir: Path, row: dict[str, str]) -> np.ndarray | None:
    path = base_dir / "gpm_imerg" / row["gpm_imerg_filename"]
    if not path.exists():
        return None
    return read_tiff_array(path).astype(np.float32)


def build_row_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(r["name_location"], r["datetime"]): r for r in rows}


def offset_datetime(value: str, minutes: int) -> str:
    return (datetime.fromisoformat(value) + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def sample_rows(rows: list[dict[str, str]], cap: int, seed: int) -> list[dict[str, str]]:
    if len(rows) <= cap:
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, cap)


# ----------------------------------------------------------------------------- adjacency

def normalized_xcorr_peak(a: np.ndarray, b: np.ndarray, min_overlap_frac: float = 0.25) -> tuple[int, int, float, float]:
    """Full normalized cross-correlation via FFT. Returns (dy, dx, peak_corr, corr_at_zero).

    Positive (dy, dx) means b's content matches a displaced by (dy, dx).
    Overlap-normalized so partial tile overlap is scored fairly.
    """

    h, w = a.shape
    fs = (2 * h - 1, 2 * w - 1)
    ones = np.ones_like(a)
    fa, fb = np.fft.rfft2(a, fs), np.fft.rfft2(b[::-1, ::-1], fs)
    f1, f1b = np.fft.rfft2(ones, fs), np.fft.rfft2(ones[::-1, ::-1], fs)
    fa2, fb2 = np.fft.rfft2(a * a, fs), np.fft.rfft2((b * b)[::-1, ::-1], fs)

    n = np.fft.irfft2(f1 * f1b, fs)          # overlap pixel count
    s_ab = np.fft.irfft2(fa * fb, fs)        # sum a*b over overlap
    s_a = np.fft.irfft2(fa * f1b, fs)        # sum a over overlap
    s_b = np.fft.irfft2(f1 * fb, fs)         # sum b over overlap
    s_a2 = np.fft.irfft2(fa2 * f1b, fs)
    s_b2 = np.fft.irfft2(f1 * fb2, fs)

    n = np.maximum(n, 1e-6)
    cov = s_ab - s_a * s_b / n
    # Variance floor scales with overlap size: tiny overlaps of near-constant pixels
    # otherwise produce |corr| > 1 spikes (seen as 3.13 on sri_lanka/hat_yai).
    var_a = np.maximum(s_a2 - s_a * s_a / n, 1e-4 * n)
    var_b = np.maximum(s_b2 - s_b * s_b / n, 1e-4 * n)
    corr = cov / np.sqrt(var_a * var_b)
    corr[n < min_overlap_frac * h * w] = -2.0

    idx = int(np.argmax(corr))
    py, px = divmod(idx, corr.shape[1])
    dy, dx = py - (h - 1), px - (w - 1)
    zero = float(corr[h - 1, w - 1])
    return dy, dx, float(corr[py, px]), zero


def analyze_train_eval_adjacency(
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    train_dir: Path,
    eval_dir: Path,
    cfg: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    adj_cfg = cfg["adjacency"]
    ir_band = int(cfg["bands"]["ir_window_default"])
    max_ts = int(adj_cfg["max_timestamps_per_pair"])
    pairs_table = read_csv_rows(resolve_path(adj_cfg["pairs_table"]))
    pairs = [(r["eval_location"], r["nearest_train_location"], r["satellite_target"]) for r in pairs_table]

    train_by_loc: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in train_rows:
        train_by_loc[r["name_location"]][r["datetime"]] = r
    eval_by_loc: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in eval_rows:
        eval_by_loc[r["name_location"]][r["datetime"]] = r

    results: list[dict[str, Any]] = []
    per_ts_rows: list[dict[str, Any]] = []
    for eval_loc, train_loc, sat in pairs:
        shared = sorted(set(eval_by_loc[eval_loc]) & set(train_by_loc[train_loc]))
        if not shared:
            continue
        rng = random.Random(int(cfg["sampling"]["random_seed"]))
        rng.shuffle(shared)
        offsets: list[tuple[int, int]] = []
        peaks: list[float] = []
        zeros: list[float] = []
        used = 0
        for ts in shared:
            if used >= max_ts:
                break
            a = load_ir_band(eval_dir, eval_by_loc[eval_loc][ts], ir_band, None)
            b = load_ir_band(train_dir, train_by_loc[train_loc][ts], ir_band, None)
            if a is None or b is None or a.shape != b.shape or a.std() == 0 or b.std() == 0:
                continue
            dy, dx, peak, zero = normalized_xcorr_peak(a, b)
            offsets.append((dy, dx))
            peaks.append(peak)
            zeros.append(zero)
            per_ts_rows.append(
                {"eval_location": eval_loc, "train_location": train_loc, "satellite": sat,
                 "datetime": ts, "peak_dy": dy, "peak_dx": dx, "peak_corr": round(peak, 4),
                 "zero_offset_corr": round(zero, 4)}
            )
            used += 1
        if not offsets:
            continue
        counts = defaultdict(int)
        for o in offsets:
            counts[o] += 1
        modal_offset, modal_count = max(counts.items(), key=lambda kv: kv[1])
        results.append(
            {
                "eval_location": eval_loc,
                "train_location": train_loc,
                "satellite": sat,
                "timestamps_used": used,
                "modal_dy": modal_offset[0],
                "modal_dx": modal_offset[1],
                "modal_offset_share": round(modal_count / used, 3),
                "mean_peak_corr": round(float(np.mean(peaks)), 4),
                "mean_zero_offset_corr": round(float(np.mean(zeros)), 4),
                "overlap_hint": "LIKELY_OVERLAP" if modal_count / used >= 0.6 and float(np.mean(peaks)) >= 0.5 else "no_stable_overlap",
            }
        )
    write_csv(out_dir / "train_eval_adjacency_pairs.csv", results)
    write_csv(out_dir / "train_eval_adjacency_samples.csv", per_ts_rows)
    return results


# ----------------------------------------------------------------------------- ir_rain_lag

def analyze_ir_rain_lag(
    train_rows: list[dict[str, str]],
    train_dir: Path,
    cfg: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    lag_cfg = cfg["ir_rain_lag"]
    ir_band = int(cfg["bands"]["ir_window_default"])
    size = (int(cfg["data"]["target_height"]), int(cfg["data"]["target_width"]))
    offsets = [int(k) for k in lag_cfg["row_offsets"]]
    index = build_row_index(train_rows)
    rainy = [r for r in train_rows if True]
    rng = random.Random(int(cfg["sampling"]["random_seed"]))
    rng.shuffle(rainy)

    acc: dict[tuple[str, int], list[float]] = defaultdict(list)
    used = 0
    for row in rainy:
        if used >= int(lag_cfg["max_rows"]):
            break
        target = load_target(train_dir, row)
        if target is None or float(target.mean()) < float(lag_cfg["min_target_mean"]) or target.std() == 0:
            continue
        tz = target - target.mean()
        sat = row["satellite_target"]
        got_any = False
        for k in offsets:
            other = row if k == 0 else index.get((row["name_location"], offset_datetime(row["datetime"], 30 * k)))
            if other is None:
                continue
            band = load_ir_band(train_dir, other, ir_band, size)
            if band is None or band.std() == 0:
                continue
            cold = -(band - band.mean())
            corr = float(np.corrcoef(cold.ravel(), tz.ravel())[0, 1])
            if np.isfinite(corr):
                acc[(sat, k)].append(corr)
                got_any = True
        if got_any:
            used += 1

    rows_out: list[dict[str, Any]] = []
    for (sat, k), values in sorted(acc.items()):
        rows_out.append(
            {"satellite": sat, "row_offset": k, "minutes": 30 * k, "samples": len(values),
             "mean_corr": round(float(np.mean(values)), 4), "median_corr": round(float(np.median(values)), 4)}
        )
    write_csv(out_dir / "ir_rain_lag_correlation.csv", rows_out)

    if plt is not None and rows_out:
        fig, ax = plt.subplots(figsize=(7, 4))
        for sat in SATELLITES:
            pts = [(r["minutes"], r["mean_corr"]) for r in rows_out if r["satellite"] == sat]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, marker="o", label=sat)
        ax.set_xlabel("IR frame offset vs rain time (minutes)")
        ax.set_ylabel("mean pixel corr(cold IR, rain)")
        ax.axvline(0, color="gray", lw=0.5)
        ax.legend()
        fig.tight_layout()
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "figures" / "ir_rain_lag.png", dpi=120)
        plt.close(fig)
    return rows_out


# ----------------------------------------------------------------------------- target_autocorr

def analyze_target_autocorr(
    train_rows: list[dict[str, str]],
    train_dir: Path,
    cfg: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    ac_cfg = cfg["target_autocorr"]
    lags = [int(k) for k in ac_cfg["row_lags"]]
    index = build_row_index(train_rows)
    by_loc: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in train_rows:
        by_loc[r["name_location"]].append(r)

    rng = random.Random(int(cfg["sampling"]["random_seed"]))
    stats: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: {"n": 0.0, "corr_sum": 0.0, "se_sum": 0.0, "sq_sum": 0.0, "px": 0.0})
    for loc, rows in sorted(by_loc.items()):
        picked = rng.sample(rows, min(len(rows), int(ac_cfg["max_rows_per_location"])))
        for row in picked:
            base = load_target(train_dir, row)
            if base is None:
                continue
            for k in lags:
                other = index.get((loc, offset_datetime(row["datetime"], 30 * k)))
                if other is None:
                    continue
                fut = load_target(train_dir, other)
                if fut is None:
                    continue
                key = (loc, k)
                s = stats[key]
                s["n"] += 1
                if base.std() > 0 and fut.std() > 0:
                    c = float(np.corrcoef(base.ravel(), fut.ravel())[0, 1])
                    if np.isfinite(c):
                        s["corr_sum"] += c
                diff = fut - base
                s["se_sum"] += float((diff * diff).sum())
                s["sq_sum"] += float((fut * fut).sum())
                s["px"] += diff.size

    rows_out: list[dict[str, Any]] = []
    for (loc, k), s in sorted(stats.items()):
        if s["px"] == 0:
            continue
        rows_out.append(
            {
                "name_location": loc,
                "row_lag": k,
                "minutes": 30 * k,
                "pairs": int(s["n"]),
                "mean_field_corr": round(s["corr_sum"] / max(s["n"], 1), 4),
                "persistence_rmse": round(math.sqrt(s["se_sum"] / s["px"]), 4),
                "zero_rmse": round(math.sqrt(s["sq_sum"] / s["px"]), 4),
                "persistence_skill": round(1.0 - math.sqrt(s["se_sum"] / s["px"]) / max(math.sqrt(s["sq_sum"] / s["px"]), 1e-9), 4),
            }
        )
    write_csv(out_dir / "target_lag_autocorr.csv", rows_out)
    return rows_out


# ----------------------------------------------------------------------------- bt_rain_response

def analyze_bt_rain_response(
    train_rows: list[dict[str, str]],
    train_dir: Path,
    cfg: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    bt_cfg = cfg["bt_rain_response"]
    ir_band = int(cfg["bands"]["ir_window_default"])
    size = (int(cfg["data"]["target_height"]), int(cfg["data"]["target_width"]))
    n_bins = int(bt_cfg["value_bins"])
    edges = np.linspace(0.0, 255.0, n_bins + 1)

    hists: dict[str, dict[str, np.ndarray]] = {
        sat: {"count": np.zeros(n_bins), "rain_sum": np.zeros(n_bins), "rain_pos": np.zeros(n_bins)}
        for sat in SATELLITES
    }
    rng = random.Random(int(cfg["sampling"]["random_seed"]))
    rows = sample_rows(train_rows, int(bt_cfg["max_rows"]), rng.randint(0, 10**9))
    used = 0
    for row in rows:
        target = load_target(train_dir, row)
        band = load_ir_band(train_dir, row, ir_band, size)
        if target is None or band is None:
            continue
        sat = row["satellite_target"]
        idx = np.clip(np.digitize(band.ravel(), edges) - 1, 0, n_bins - 1)
        h = hists[sat]
        np.add.at(h["count"], idx, 1.0)
        np.add.at(h["rain_sum"], idx, target.ravel())
        np.add.at(h["rain_pos"], idx, (target.ravel() > 0).astype(np.float64))
        used += 1

    rows_out: list[dict[str, Any]] = []
    for sat in SATELLITES:
        h = hists[sat]
        for b in range(n_bins):
            if h["count"][b] == 0:
                continue
            rows_out.append(
                {
                    "satellite": sat,
                    "ir_value_low": round(float(edges[b]), 1),
                    "ir_value_high": round(float(edges[b + 1]), 1),
                    "pixels": int(h["count"][b]),
                    "mean_rain": round(float(h["rain_sum"][b] / h["count"][b]), 5),
                    "rain_prob": round(float(h["rain_pos"][b] / h["count"][b]), 5),
                }
            )
    write_csv(out_dir / "bt_rain_response.csv", rows_out)

    if plt is not None and rows_out:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for sat in SATELLITES:
            pts = [(r["ir_value_low"], r["mean_rain"], r["rain_prob"]) for r in rows_out if r["satellite"] == sat]
            if pts:
                xs, means, probs = zip(*sorted(pts))
                axes[0].plot(xs, means, label=sat)
                axes[1].plot(xs, probs, label=sat)
        axes[0].set_title("E[rain | IR value]")
        axes[1].set_title("P(rain>0 | IR value)")
        for ax in axes:
            ax.set_xlabel("IR window band value (0-255)")
            ax.legend()
        fig.tight_layout()
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "figures" / "bt_rain_response.png", dpi=120)
        plt.close(fig)
    return rows_out


# ----------------------------------------------------------------------------- band_health

def analyze_band_health(
    train_rows: list[dict[str, str]],
    train_dir: Path,
    cfg: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    bh_cfg = cfg["band_health"]
    rng = random.Random(int(cfg["sampling"]["random_seed"]))
    rows = sample_rows(train_rows, int(bh_cfg["max_rows"]), rng.randint(0, 10**9))

    # (satellite, band, hour_bucket) -> aggregates. hour buckets of 3h UTC.
    agg: dict[tuple[str, int, int], dict[str, float]] = defaultdict(lambda: {"n": 0.0, "zero": 0.0, "sat255": 0.0, "std_sum": 0.0, "files": 0.0})
    shape_counts: dict[tuple[str, tuple[int, ...]], int] = defaultdict(int)
    for row in rows:
        path = last_obs_path(train_dir, row)
        if path is None or not path.exists():
            continue
        arr = read_tiff_array(path)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        sat = row["satellite_target"]
        shape_counts[(sat, arr.shape)] += 1
        hour = int(row["datetime"][11:13]) // 3
        px = float(arr.shape[0] * arr.shape[1])
        for b in range(arr.shape[2]):
            band = arr[:, :, b]
            key = (sat, b, hour)
            a = agg[key]
            a["n"] += px
            a["zero"] += float((band == 0).sum())
            a["sat255"] += float((band == 255).sum())
            a["std_sum"] += float(band.astype(np.float32).std())
            a["files"] += 1

    rows_out: list[dict[str, Any]] = []
    for (sat, b, hour), a in sorted(agg.items()):
        rows_out.append(
            {
                "satellite": sat,
                "band": b,
                "utc_hour_bucket": f"{hour * 3:02d}-{hour * 3 + 3:02d}",
                "files": int(a["files"]),
                "zero_frac": round(a["zero"] / a["n"], 5),
                "sat255_frac": round(a["sat255"] / a["n"], 5),
                "mean_std": round(a["std_sum"] / a["files"], 3),
            }
        )
    write_csv(out_dir / "band_health_by_hour.csv", rows_out)
    shape_rows = [
        {"satellite": sat, "shape": "x".join(map(str, shp)), "files": n}
        for (sat, shp), n in sorted(shape_counts.items())
    ]
    write_csv(out_dir / "satellite_shape_counts.csv", shape_rows)
    return rows_out


# ----------------------------------------------------------------------------- target_quantization

def analyze_target_quantization(
    train_rows: list[dict[str, str]],
    train_dir: Path,
    cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    q_cfg = cfg["target_quantization"]
    rng = random.Random(int(cfg["sampling"]["random_seed"]))
    rows = sample_rows(train_rows, int(q_cfg["max_rows"]), rng.randint(0, 10**9))
    positives: list[np.ndarray] = []
    for row in rows:
        target = load_target(train_dir, row)
        if target is None:
            continue
        pos = target[target > 0]
        if pos.size:
            positives.append(pos)
    if not positives:
        return {}
    values = np.concatenate(positives)
    uniq = np.unique(np.round(values, 6))
    mult_001 = float(np.mean(np.abs(values * 100 - np.round(values * 100)) < 1e-4))
    mult_01 = float(np.mean(np.abs(values * 10 - np.round(values * 10)) < 1e-4))
    result = {
        "sampled_rows": len(rows),
        "positive_pixels": int(values.size),
        "distinct_positive_values": int(uniq.size),
        "min_positive": float(values.min()),
        "smallest_5_values": [float(v) for v in uniq[:5]],
        "frac_multiple_of_0p01": round(mult_001, 5),
        "frac_multiple_of_0p1": round(mult_01, 5),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "p999": float(np.percentile(values, 99.9)),
        "max": float(values.max()),
    }
    (out_dir / "target_quantization.json").write_text(json.dumps(result, indent=2))
    return result


# ----------------------------------------------------------------------------- spectrum

def radial_power_spectrum(field: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.abs(np.fft.fft2(field - field.mean())) ** 2)
    h, w = field.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    n_bins = min(cy, cx)
    out = np.zeros(n_bins)
    for k in range(n_bins):
        mask = r == k
        out[k] = f[mask].mean() if mask.any() else 0.0
    return out


def analyze_spectrum(
    train_rows: list[dict[str, str]],
    train_dir: Path,
    cfg: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    sp_cfg = cfg["spectrum"]
    ir_band = int(cfg["bands"]["ir_window_default"])
    size = (int(cfg["data"]["target_height"]), int(cfg["data"]["target_width"]))
    rng = random.Random(int(cfg["sampling"]["random_seed"]))
    rows = [r for r in train_rows]
    rng.shuffle(rows)
    target_spectra: list[np.ndarray] = []
    ir_spectra: list[np.ndarray] = []
    for row in rows:
        if len(target_spectra) >= int(sp_cfg["max_rows"]):
            break
        target = load_target(train_dir, row)
        if target is None or float(target.mean()) < float(sp_cfg["min_target_mean"]):
            continue
        band = load_ir_band(train_dir, row, ir_band, size)
        if band is None:
            continue
        target_spectra.append(radial_power_spectrum(target))
        ir_spectra.append(radial_power_spectrum(band))
    if not target_spectra:
        return []
    t_mean = np.mean(target_spectra, axis=0)
    i_mean = np.mean(ir_spectra, axis=0)
    rows_out = [
        {
            "wavenumber": k,
            "approx_wavelength_px": round(41.0 / max(k, 1), 1),
            "target_power": float(t_mean[k]),
            "ir_power": float(i_mean[k]),
            "target_power_norm": float(t_mean[k] / t_mean.sum()),
            "ir_power_norm": float(i_mean[k] / i_mean.sum()),
        }
        for k in range(len(t_mean))
    ]
    write_csv(out_dir / "radial_power_spectrum.csv", rows_out)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ks = [r["wavenumber"] for r in rows_out[1:]]
        ax.semilogy(ks, [r["target_power_norm"] for r in rows_out[1:]], label="GPM target")
        ax.semilogy(ks, [r["ir_power_norm"] for r in rows_out[1:]], label="IR input (resized)")
        ax.set_xlabel("radial wavenumber (41px tile)")
        ax.set_ylabel("normalized power")
        ax.legend()
        fig.tight_layout()
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "figures" / "radial_power_spectrum.png", dpi=120)
        plt.close(fig)
    return rows_out


# ----------------------------------------------------------------------------- position_bias

def analyze_position_bias(
    train_rows: list[dict[str, str]],
    train_dir: Path,
    cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    pb_cfg = cfg["position_bias"]
    rng = random.Random(int(cfg["sampling"]["random_seed"]))
    rows = sample_rows(train_rows, int(pb_cfg["max_rows"]), rng.randint(0, 10**9))
    mean_map = None
    freq_map = None
    n = 0
    for row in rows:
        target = load_target(train_dir, row)
        if target is None:
            continue
        if mean_map is None:
            mean_map = np.zeros_like(target, dtype=np.float64)
            freq_map = np.zeros_like(target, dtype=np.float64)
        mean_map += target
        freq_map += target > 0
        n += 1
    if mean_map is None or n == 0:
        return {}
    mean_map /= n
    freq_map /= n
    h, w = mean_map.shape
    cy0, cy1 = h // 4, 3 * h // 4
    center_mean = float(mean_map[cy0:cy1, cy0:cy1].mean())
    edge_mask = np.ones_like(mean_map, dtype=bool)
    edge_mask[cy0:cy1, cy0:cy1] = False
    edge_mean = float(mean_map[edge_mask].mean())
    result = {
        "rows_used": n,
        "center_mean_rain": round(center_mean, 5),
        "edge_mean_rain": round(edge_mean, 5),
        "center_over_edge_ratio": round(center_mean / max(edge_mean, 1e-9), 4),
        "row_marginal_max_over_min": round(float(mean_map.mean(axis=1).max() / max(mean_map.mean(axis=1).min(), 1e-9)), 4),
        "col_marginal_max_over_min": round(float(mean_map.mean(axis=0).max() / max(mean_map.mean(axis=0).min(), 1e-9)), 4),
    }
    (out_dir / "position_bias.json").write_text(json.dumps(result, indent=2))
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        im0 = axes[0].imshow(mean_map)
        axes[0].set_title("mean rain per pixel")
        fig.colorbar(im0, ax=axes[0], shrink=0.8)
        im1 = axes[1].imshow(freq_map)
        axes[1].set_title("rain frequency per pixel")
        fig.colorbar(im1, ax=axes[1], shrink=0.8)
        fig.tight_layout()
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "figures" / "position_bias.png", dpi=120)
        plt.close(fig)
    return result


# ----------------------------------------------------------------------------- oof_join

def analyze_oof_join(
    train_rows: list[dict[str, str]],
    train_dir: Path,
    cfg: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    oj_cfg = cfg["oof_join"]
    oof_path = resolve_path(oj_cfg["oof_sample_metrics"])
    if not oof_path.exists():
        print(f"oof_join skipped: {oof_path} not found")
        return []
    oof = {r["unique_id"]: r for r in read_csv_rows(oof_path)}
    visible_bands = [int(b) for b in cfg["bands"]["visible"]]
    ir_band = int(cfg["bands"]["ir_window_default"])

    rng = random.Random(int(cfg["sampling"]["random_seed"]))
    rows = sample_rows([r for r in train_rows if r["unique_id"] in oof], int(oj_cfg["max_rows"]), rng.randint(0, 10**9))
    joined: list[dict[str, Any]] = []
    for row in rows:
        path = last_obs_path(train_dir, row)
        if path is None or not path.exists():
            continue
        arr = read_tiff_array(path)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        vis = [b for b in visible_bands if b < arr.shape[2]]
        if not vis:
            continue
        vis_mean = float(arr[:, :, vis].astype(np.float32).mean())
        irb = arr[:, :, min(ir_band, arr.shape[2] - 1)]
        m = oof[row["unique_id"]]
        joined.append(
            {
                "unique_id": row["unique_id"],
                "satellite": row["satellite_target"],
                "utc_hour": int(row["datetime"][11:13]),
                "vis_mean": round(vis_mean, 3),
                "ir_zero_frac": round(float((irb == 0).mean()), 5),
                "tile_rmse": float(m["tile_rmse"]),
                "target_mean": float(m["target_mean"]),
            }
        )
    write_csv(out_dir / "oof_join_features.csv", joined)

    # Conditional table: same rain regime, day (bright visible) vs night (dark visible).
    bins_out: list[dict[str, Any]] = []
    for regime, lo, hi in (("light", 0.0, 0.1), ("mid", 0.1, 1.0), ("heavy", 1.0, 1e9)):
        sel = [r for r in joined if lo <= r["target_mean"] < hi]
        if len(sel) < 30:
            continue
        vis_values = sorted(r["vis_mean"] for r in sel)
        cut = vis_values[len(vis_values) // 2]
        dark = [r["tile_rmse"] for r in sel if r["vis_mean"] <= cut]
        bright = [r["tile_rmse"] for r in sel if r["vis_mean"] > cut]
        bins_out.append(
            {
                "rain_regime": regime,
                "samples": len(sel),
                "vis_median_cut": round(cut, 2),
                "dark_mean_tile_rmse": round(float(np.mean(dark)), 4),
                "bright_mean_tile_rmse": round(float(np.mean(bright)), 4),
                "dark_over_bright": round(float(np.mean(dark)) / max(float(np.mean(bright)), 1e-9), 4),
            }
        )
    write_csv(out_dir / "oof_daynight_conditional.csv", bins_out)
    return bins_out


# ----------------------------------------------------------------------------- report / main

def write_report(out_dir: Path, sections: dict[str, Any], timings: dict[str, float]) -> None:
    lines = [
        "# Pixel-Level EDA Report (l_eda/exp002)",
        "",
        "Generated by `l_eda/exp002/run_pixel_eda.py`. See the script docstring for what each",
        "analysis answers and `doc/data_characteristics_review.md` for the motivating findings.",
        "",
        f"- OpenCV: `{'available' if cv2 is not None else 'missing'}`",
        f"- Timings: `{json.dumps({k: round(v, 1) for k, v in timings.items()})}`",
        "",
        "## Key Tables",
        "",
    ]
    adjacency = sections.get("train_eval_adjacency") or []
    if adjacency:
        lines += ["### Train/Eval tile adjacency (overlap candidates)", "",
                  "| eval | train | ts | modal (dy,dx) | share | peak corr | hint |",
                  "| --- | --- | ---: | --- | ---: | ---: | --- |"]
        for r in sorted(adjacency, key=lambda x: -x["mean_peak_corr"]):
            lines.append(
                f"| {r['eval_location']} | {r['train_location']} | {r['timestamps_used']} | "
                f"({r['modal_dy']},{r['modal_dx']}) | {r['modal_offset_share']} | {r['mean_peak_corr']} | {r['overlap_hint']} |"
            )
        lines.append("")
    lag = sections.get("ir_rain_lag") or []
    if lag:
        lines += ["### IR->rain lag correlation (mean pixel corr)", "",
                  "| satellite | " + " | ".join(str(r["minutes"]) + "min" for r in lag if r["satellite"] == "himawari") + " |"]
        for sat in SATELLITES:
            vals = [r for r in lag if r["satellite"] == sat]
            if vals:
                lines.append(f"| {sat} | " + " | ".join(str(r["mean_corr"]) for r in sorted(vals, key=lambda x: x["minutes"])) + " |")
        lines.append("")
    quant = sections.get("target_quantization") or {}
    if quant:
        lines += ["### Target quantization", "",
                  f"- distinct positive values: `{quant['distinct_positive_values']}` over `{quant['positive_pixels']}` pixels",
                  f"- multiples of 0.01: `{quant['frac_multiple_of_0p01']}`, of 0.1: `{quant['frac_multiple_of_0p1']}`",
                  f"- min positive `{quant['min_positive']}`, p99 `{round(quant['p99'], 2)}`, p99.9 `{round(quant['p999'], 2)}`, max `{round(quant['max'], 2)}`",
                  ""]
    pos = sections.get("position_bias") or {}
    if pos:
        lines += ["### Position bias", "",
                  f"- center/edge mean-rain ratio: `{pos['center_over_edge_ratio']}`",
                  ""]
    daynight = sections.get("oof_join") or []
    if daynight:
        lines += ["### OOF error, dark vs bright visible bands (same rain regime)", "",
                  "| regime | n | dark rmse | bright rmse | dark/bright |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for r in daynight:
            lines.append(f"| {r['rain_regime']} | {r['samples']} | {r['dark_mean_tile_rmse']} | {r['bright_mean_tile_rmse']} | {r['dark_over_bright']} |")
        lines.append("")
    lines += ["## Produced Files", ""]
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "PIXEL_EDA_REPORT.md":
            lines.append(f"- `{p.relative_to(out_dir)}`")
    (out_dir / "PIXEL_EDA_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    train_dir = resolve_path(cfg["data"]["train_dir"])
    eval_dir = resolve_path(cfg["data"]["evaluation_dir"])
    train_rows = read_csv_rows(resolve_path(cfg["data"]["train_csv"]))
    eval_rows = read_csv_rows(resolve_path(cfg["data"]["evaluation_csv"]))
    out_dir = resolve_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    analyses = cfg["analyses"]
    sections: dict[str, Any] = {}
    timings: dict[str, float] = {}

    def run(name: str, fn, *fn_args):
        if not analyses.get(name, False):
            return
        start = time.time()
        sections[name] = fn(*fn_args)
        timings[name] = time.time() - start
        print(f"{name}: done in {timings[name]:.1f}s")

    run("train_eval_adjacency", analyze_train_eval_adjacency, train_rows, eval_rows, train_dir, eval_dir, cfg, out_dir)
    run("ir_rain_lag", analyze_ir_rain_lag, train_rows, train_dir, cfg, out_dir)
    run("target_autocorr", analyze_target_autocorr, train_rows, train_dir, cfg, out_dir)
    run("bt_rain_response", analyze_bt_rain_response, train_rows, train_dir, cfg, out_dir)
    run("band_health", analyze_band_health, train_rows, train_dir, cfg, out_dir)
    run("target_quantization", analyze_target_quantization, train_rows, train_dir, cfg, out_dir)
    run("spectrum", analyze_spectrum, train_rows, train_dir, cfg, out_dir)
    run("position_bias", analyze_position_bias, train_rows, train_dir, cfg, out_dir)
    run("oof_join", analyze_oof_join, train_rows, train_dir, cfg, out_dir)

    write_report(out_dir, sections, timings)
    (out_dir / "summary.json").write_text(
        json.dumps({"timings": {k: round(v, 1) for k, v in timings.items()}, "outputs": str(out_dir)}, indent=2)
    )
    print(f"report: {out_dir / 'PIXEL_EDA_REPORT.md'}")


if __name__ == "__main__":
    main()
