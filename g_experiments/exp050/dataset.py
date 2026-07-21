"""PyTorch datasets for exp050: exp017's full feature pipeline (canonical bands, engineered
physics channels, temporal_abs, target_time_first) plus exp018's zero-observation row filter.

New vs exp009, all opt-in via config `features:`:
- canonical_bands: append 6 physically-aligned channels per frame (VIS red, mid-IR 3.9um,
  WV 7.3um, IR 8.5um, IR window 10.5um, IR split 12.3um) selected per satellite with the
  discussion's verified index table. Band index N means different wavelengths on different
  satellites (Meteosat is wrong on 2 of 3 key bands with fixed indices; the WV fix alone was
  measured +25% correlation), so this gives shared conv weights a cross-satellite-consistent
  view without disturbing the raw 16-channel block.
- engineered: 3 channels per frame computed on RAW uint8 values before normalization (the
  classic split-window *difference* is destroyed by uint8 quantization; the *ratio* survives):
  split-window ratio SPL/(W+1), IR8.5-W, WV7.3-W. Plus 2 channels per context row: temporal
  diff W[newest]-W[oldest] (change magnitude was measured Spearman +0.69 with rain) and a
  day/night flag from mean raw VIS brightness (VIS bands are noise at night, 52-73% of data).
"""

from __future__ import annotations

import ast
from collections import Counter
import csv
from datetime import datetime, timedelta
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from tiff_utils import read_tiff_array

SATELLITES = ("goes", "himawari", "meteosat")

# 0-based band indices per satellite, from the official discussion's verified
# wavelength-mapping table (doc/discussion_insights.md section 3).
KEY_BANDS: dict[str, dict[str, int]] = {
    #             VIS red  MIR3.9  WV7.3  IR8.5  IRwin10.5  IRsplit12.3
    "himawari": {"vis": 2, "mir": 6, "wv": 9, "ir85": 10, "win": 12, "spl": 14},
    "goes":     {"vis": 1, "mir": 6, "wv": 9, "ir85": 10, "win": 12, "spl": 14},
    "meteosat": {"vis": 2, "mir": 8, "wv": 10, "ir85": 11, "win": 13, "spl": 14},
}
CANONICAL_ORDER = ("vis", "mir", "wv", "ir85", "win", "spl")

# Fixed analytic scalings for the engineered channels (documented, not learned):
# raw values are uint8 [0,255]; the ratio sits near 1.0, the differences within ~+-64.
RATIO_CENTER = 1.0
RATIO_SCALE = 4.0
DIFF_SCALE = 32.0
NIGHT_VIS_MEAN_THRESHOLD = 5.0  # discussion Finding 3: mean raw VIS < 5 => night


def features_from_config(config: dict) -> dict[str, bool]:
    features_cfg = config.get("features", {}) or {}
    return {
        "canonical_bands": bool(features_cfg.get("canonical_bands", False)),
        "engineered": bool(features_cfg.get("engineered", False)),
        "temporal_abs": bool(features_cfg.get("temporal_abs", False)),
        "target_time_first": bool(features_cfg.get("target_time_first", False)),
        "ir_rain_proxy": bool(features_cfg.get("ir_rain_proxy", False)),
        "split_window_btd": bool(features_cfg.get("split_window_btd", False)),
    }


def channels_per_frame(satellite_channels: int, features: dict[str, bool]) -> int:
    extra = 0
    if features.get("canonical_bands"):
        extra += len(CANONICAL_ORDER)
    if features.get("engineered"):
        extra += 3
    if features.get("ir_rain_proxy"):
        extra += 1
    if features.get("split_window_btd"):
        extra += 1
    return satellite_channels + extra


def channels_per_row(features: dict[str, bool]) -> int:
    if not features.get("engineered"):
        return 0
    return 2 + int(features.get("temporal_abs", False))


def expected_in_channels(
    satellite_channels: int, max_observations: int, context_rows: int, features: dict[str, bool]
) -> int:
    per_frame = channels_per_frame(satellite_channels, features)
    frames = max_observations * context_rows
    return frames * per_frame + frames + context_rows * channels_per_row(features) + len(SATELLITES)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_observation_filenames(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed)!r}")
    return [str(item) for item in parsed]


def drop_zero_observation_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop rows whose own observation list is empty (235 all-Meteosat rows in train).

    These rows have a valid GPM target but no causal satellite input of their own — pure label
    noise for training (doc/discussion_insights.md §3, Finding 9). Apply to TRAIN rows only:
    validation rows stay intact so OOF metrics remain comparable across experiments, and
    evaluation rows must never be dropped (every row needs a prediction).
    """
    kept: list[dict[str, str]] = []
    dropped = 0
    for row in rows:
        try:
            observations = parse_observation_filenames(row["last_30_minutes_observation_filename"])
        except (ValueError, SyntaxError):
            observations = []
        if observations:
            kept.append(row)
        else:
            dropped += 1
    if dropped:
        print(f"drop_zero_observation_rows: dropped {dropped} rows without observations", flush=True)
    return kept


def load_norm_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    stats: dict[str, dict[str, np.ndarray]] = {}
    for sat, values in raw.items():
        stats[sat] = {
            "mean": np.asarray(values["mean"], dtype=np.float32),
            "std": np.asarray(values["std"], dtype=np.float32),
        }
    return stats


def make_group_kfold_split(
    rows: list[dict[str, str]],
    n_splits: int,
    fold: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must be in [0, {n_splits}), got {fold}")

    group_counts = Counter(row["name_location"] for row in rows)
    if len(group_counts) < n_splits:
        raise ValueError(f"n_splits={n_splits} exceeds number of groups={len(group_counts)}")

    rng = random.Random(seed)
    locations = sorted(group_counts)
    rng.shuffle(locations)
    locations.sort(key=lambda loc: group_counts[loc], reverse=True)

    fold_counts = [0] * n_splits
    fold_locations: list[list[str]] = [[] for _ in range(n_splits)]
    for location in locations:
        target_fold = min(range(n_splits), key=lambda idx: (fold_counts[idx], idx))
        fold_locations[target_fold].append(location)
        fold_counts[target_fold] += group_counts[location]

    valid_location_set = set(fold_locations[fold])
    train_rows = [row for row in rows if row["name_location"] not in valid_location_set]
    valid_rows = [row for row in rows if row["name_location"] in valid_location_set]
    valid_locations = sorted(valid_location_set)
    return train_rows, valid_rows, valid_locations


def sample_rows(rows: list[dict[str, str]], max_samples: int | None, seed: int) -> list[dict[str, str]]:
    if not max_samples or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_samples)


def _resize_chw(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Anti-aliased resize for the always-downsizing case here; falls back to bilinear
    if the source happens to be smaller than the target in any dimension."""

    src_h, src_w = tensor.shape[-2:]
    dst_h, dst_w = size
    if src_h >= dst_h and src_w >= dst_w:
        return F.adaptive_avg_pool2d(tensor.unsqueeze(0), size).squeeze(0)
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


def read_satellite_raw(
    path: Path,
    channels: int,
    target_size: tuple[int, int],
) -> tuple[torch.Tensor, int]:
    """Read a frame at RAW uint8 scale (0..255), resized, zero-padded to `channels`.

    Returns (tensor, n_chan_present). The raw scale is what the engineered physics channels
    must be computed on (uint8 quantization kills differences taken after normalization).
    """
    arr, _ = read_tiff_array(path)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    n_chan = min(arr.shape[2], channels)
    arr = np.moveaxis(arr[:, :, :n_chan].astype(np.float32), -1, 0)
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    if tensor.shape[0] < channels:
        pad = torch.zeros((channels - tensor.shape[0], tensor.shape[1], tensor.shape[2]), dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    return _resize_chw(tensor, target_size), n_chan


def normalize_raw(
    raw: torch.Tensor,
    n_chan: int,
    satellite: str,
    norm_stats: dict[str, dict[str, np.ndarray]] | None,
) -> torch.Tensor:
    """Same normalization semantics as exp009: /255 then per-(sat,band) mean/std on the
    channels actually present; padded channels stay exactly 0."""
    base = raw / 255.0
    if norm_stats is not None and satellite in norm_stats:
        mean = torch.from_numpy(norm_stats[satellite]["mean"]).to(base.dtype).view(-1, 1, 1)
        std = torch.from_numpy(norm_stats[satellite]["std"]).to(base.dtype).view(-1, 1, 1)
        base = (base - mean[: base.shape[0]]) / std[: base.shape[0]]
    if n_chan < base.shape[0]:
        base = base.clone()
        base[n_chan:] = 0.0
    return base


def frame_engineered_channels(raw: torch.Tensor, satellite: str) -> torch.Tensor:
    """3 physics channels computed on the raw uint8 scale (doc/discussion_insights.md §3):
    split-window ratio (the difference is quantized to death; the ratio survives, measured
    partial corr -0.31/-0.20/-0.13), IR8.5-window ("hidden gem"), WV-window (deep convection)."""
    bands = KEY_BANDS[satellite]
    win = raw[bands["win"]]
    ratio = (raw[bands["spl"]] / (win + 1.0) - RATIO_CENTER) * RATIO_SCALE
    ir85_minus_w = (raw[bands["ir85"]] - win) / DIFF_SCALE
    wv_minus_w = (raw[bands["wv"]] - win) / DIFF_SCALE
    return torch.stack([ratio, ir85_minus_w, wv_minus_w], dim=0)


def split_window_btd_channel(raw: torch.Tensor, satellite: str) -> torch.Tensor:
    """Split-window brightness-temperature DIFFERENCE (spl 12.3um - win 10.5um), the
    literature-standard form (doc/domain_knowledge_review_2026-07-20.md section 2.3) --
    an ablation against frame_engineered_channels' ratio version (SPL/(win+1)), which was
    chosen there specifically to survive uint8 quantization at the cost of matching the
    textbook definition less closely. This channel tests the plain-difference form directly."""
    bands = KEY_BANDS[satellite]
    return ((raw[bands["spl"]] - raw[bands["win"]]) / DIFF_SCALE).unsqueeze(0)


def ir_rain_proxy_channel(raw: torch.Tensor, satellite: str) -> torch.Tensor:
    """Bounded satellite-aware cold-cloud proxy derived from the IR-window band.

    Raw brightness values increase with brightness temperature in this dataset.  The
    per-satellite anchors avoid pretending that the three sensors share one radiometric
    response; the output is a dimensionless [0, 1] feature for an isolated ablation.
    """
    cold_anchor = {"goes": 105.0, "himawari": 105.0, "meteosat": 95.0}[satellite]
    warm_anchor = {"goes": 190.0, "himawari": 190.0, "meteosat": 180.0}[satellite]
    win = raw[KEY_BANDS[satellite]["win"]]
    proxy = ((warm_anchor - win) / (warm_anchor - cold_anchor)).clamp(0.0, 1.0)
    return proxy.unsqueeze(0)


def build_frame_tensor(
    raw: torch.Tensor,
    n_chan: int,
    satellite: str,
    norm_stats: dict[str, dict[str, np.ndarray]] | None,
    features: dict[str, bool],
) -> torch.Tensor:
    parts = [normalize_raw(raw, n_chan, satellite, norm_stats)]
    if features.get("canonical_bands"):
        indices = [KEY_BANDS[satellite][name] for name in CANONICAL_ORDER]
        parts.append(parts[0][indices])
    if features.get("engineered"):
        parts.append(frame_engineered_channels(raw, satellite))
    if features.get("ir_rain_proxy"):
        parts.append(ir_rain_proxy_channel(raw, satellite))
    if features.get("split_window_btd"):
        parts.append(split_window_btd_channel(raw, satellite))
    return torch.cat(parts, dim=0)


def read_target_tensor(path: Path) -> torch.Tensor:
    arr, _ = read_tiff_array(path)
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


def _augment(x: torch.Tensor, y: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Relies on the caller (train.py's worker_init_fn) having seeded the `random` module
    per-worker; uses the module-level RNG directly so augmentation varies across epochs
    instead of being fixed per sample index."""

    if random.random() < 0.5:
        x = torch.flip(x, dims=(-1,))
        y = torch.flip(y, dims=(-1,)) if y is not None else None
    if random.random() < 0.5:
        x = torch.flip(x, dims=(-2,))
        y = torch.flip(y, dims=(-2,)) if y is not None else None
    k = random.randint(0, 3)
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
        y = torch.rot90(y, k, dims=(-2, -1)) if y is not None else None
    return x, y


class PrecipDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        data_dir: Path,
        max_observations: int,
        satellite_channels: int,
        target_size: tuple[int, int],
        has_target: bool,
        context_rows: int = 1,
        norm_stats: dict[str, dict[str, np.ndarray]] | None = None,
        augment: bool = False,
        features: dict[str, bool] | None = None,
    ) -> None:
        self.rows = rows
        self.data_dir = data_dir
        self.max_observations = max_observations
        self.satellite_channels = satellite_channels
        self.target_size = target_size
        self.context_rows = context_rows
        self.has_target = has_target
        self.norm_stats = norm_stats
        self.augment = augment
        self.features = features or {"canonical_bands": False, "engineered": False}
        self._frame_channels = channels_per_frame(satellite_channels, self.features)
        self._row_by_location_time: dict[tuple[str, datetime], dict[str, str]] = {}
        for row in rows:
            try:
                key = (row["name_location"], datetime.fromisoformat(row["datetime"]))
            except (KeyError, ValueError):
                continue
            self._row_by_location_time[key] = row

    def __len__(self) -> int:
        return len(self.rows)

    def _context_rows(self, row: dict[str, str]) -> list[dict[str, str] | None]:
        rows: list[dict[str, str] | None] = [row]
        if self.context_rows <= 1:
            return rows
        try:
            location = row["name_location"]
            start_time = datetime.fromisoformat(row["datetime"])
        except (KeyError, ValueError):
            rows.extend([None] * (self.context_rows - 1))
            return rows
        for offset in range(1, self.context_rows):
            rows.append(self._row_by_location_time.get((location, start_time + timedelta(minutes=30 * offset))))
        return rows

    def _row_feature_channels(self, raw_frames: list[torch.Tensor | None], satellite: str) -> list[torch.Tensor]:
        """2 channels per context row (features.engineered): temporal diff of the IR-window
        band between the newest and oldest present frames (change magnitude, Spearman +0.69
        with rain), and a day/night flag from the newest frame's mean raw VIS brightness."""
        if not channels_per_row(self.features):
            return []
        present = [raw for raw in raw_frames if raw is not None]
        bands = KEY_BANDS[satellite]
        if len(present) >= 2:
            diff = (present[-1][bands["win"]] - present[0][bands["win"]]) / DIFF_SCALE
            temporal = diff.unsqueeze(0)
        else:
            temporal = torch.zeros((1, *self.target_size), dtype=torch.float32)
        if present:
            is_day = 1.0 if float(present[-1][bands["vis"]].mean()) >= NIGHT_VIS_MEAN_THRESHOLD else 0.0
        else:
            is_day = 0.0
        day_flag = torch.full((1, *self.target_size), is_day, dtype=torch.float32)
        channels = [temporal, day_flag]
        if self.features.get("temporal_abs"):
            channels.append(temporal.abs())
        return channels

    def _observation_tensors(
        self, row: dict[str, str] | None, expected_satellite: str
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        maps: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        raw_frames: list[torch.Tensor | None] = []
        if row is None or row.get("satellite_target") != expected_satellite:
            for _ in range(self.max_observations):
                maps.append(torch.zeros((self._frame_channels, *self.target_size), dtype=torch.float32))
                masks.append(torch.zeros((1, *self.target_size), dtype=torch.float32))
                raw_frames.append(None)
            return maps, masks, self._row_feature_channels(raw_frames, expected_satellite)

        obs_names = parse_observation_filenames(row["last_30_minutes_observation_filename"])
        obs_names = obs_names[-self.max_observations :]
        start = self.max_observations - len(obs_names)
        for slot in range(self.max_observations):
            if slot < start:
                maps.append(torch.zeros((self._frame_channels, *self.target_size), dtype=torch.float32))
                masks.append(torch.zeros((1, *self.target_size), dtype=torch.float32))
                raw_frames.append(None)
                continue
            obs_name = obs_names[slot - start]
            raw, n_chan = read_satellite_raw(
                self.data_dir / expected_satellite / obs_name,
                channels=self.satellite_channels,
                target_size=self.target_size,
            )
            maps.append(
                build_frame_tensor(
                    raw,
                    n_chan=n_chan,
                    satellite=expected_satellite,
                    norm_stats=self.norm_stats,
                    features=self.features,
                )
            )
            masks.append(torch.ones((1, *self.target_size), dtype=torch.float32))
            raw_frames.append(raw)
        return maps, masks, self._row_feature_channels(raw_frames, expected_satellite)

    def _input_tensor(self, row: dict[str, str]) -> torch.Tensor:
        satellite = row["satellite_target"]
        maps: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        row_features: list[torch.Tensor] = []

        for context_row in self._context_rows(row):
            context_maps, context_masks, context_row_features = self._observation_tensors(context_row, satellite)
            maps.extend(context_maps)
            masks.extend(context_masks)
            row_features.extend(context_row_features)

        if self.features.get("target_time_first") and self.context_rows > 1:
            # The newest frame of the successor row is the closest available observation
            # to target time.  Put it first so it is not buried among six symmetric slots.
            per_row = self.max_observations
            primary = per_row * (self.context_rows - 1) + (per_row - 1)
            order = [primary] + [i for i in range(len(maps)) if i != primary]
            maps = [maps[i] for i in order]
            masks = [masks[i] for i in order]

        satellite_maps = []
        for sat in SATELLITES:
            value = 1.0 if satellite == sat else 0.0
            satellite_maps.append(torch.full((1, *self.target_size), value, dtype=torch.float32))
        return torch.cat(maps + masks + row_features + satellite_maps, dim=0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x = self._input_tensor(row)
        y = None
        if self.has_target:
            y = read_target_tensor(self.data_dir / "gpm_imerg" / row["gpm_imerg_filename"])

        if self.augment:
            x, y = _augment(x, y)

        item: dict[str, Any] = {
            "x": x,
            "unique_id": row["unique_id"],
            "name_location": row.get("name_location", ""),
            "satellite_target": row.get("satellite_target", ""),
            "datetime": row.get("datetime", ""),
            "gpm_imerg_filename": row["gpm_imerg_filename"],
        }
        if y is not None:
            item["y"] = y
        return item
