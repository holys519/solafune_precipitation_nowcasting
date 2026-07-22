"""PyTorch datasets for exp053: exp038's strict current-row-only pipeline (canonical bands,
engineered physics channels, temporal_abs, target_time_first — all off in exp053's own
config.yaml) plus a new autoregressive own-past-prediction input channel.

New vs exp038, opt-in via config `features.autoregressive_prev_pred`:
- 2 extra channels per row (not per frame): the precip value at the SAME name_location 30
  minutes earlier (T-30min), plus a validity mask (1 if that earlier row/prediction exists,
  0 otherwise) — the same zero-pad + mask-channel convention as `_observation_tensors`.
- This is legal per the organizers' 2026-07-20 ruling (doc/submission_registry.md): "自分自身の
  過去(<T)の予測値の再利用(自己回帰/recursive手法)" is explicitly permitted, because T-30min is
  always causal (<= T) relative to the row being predicted.
- Two distinct sourcing modes (see `_autoregressive_prev_pred_channels` on PrecipDataset):
  teacher forcing with the TRUE ground-truth GPM value during train/valid (reusing the
  `_row_by_location_time` lookup already used for context_rows successor lookups), vs. the
  model's OWN self-generated prediction (read from an externally-injected `ar_cache`) during
  evaluation and the self-prediction-substituted OOF pass, since the true T-30min GPM value
  does not exist for those rows (it's exactly what's being predicted for that row too).

Carried over from exp038, all opt-in via config `features:` (all off in exp053's config.yaml):
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
        "autoregressive_prev_pred": bool(features_cfg.get("autoregressive_prev_pred", False)),
    }


def channels_per_frame(satellite_channels: int, features: dict[str, bool]) -> int:
    extra = 0
    if features.get("canonical_bands"):
        extra += len(CANONICAL_ORDER)
    if features.get("engineered"):
        extra += 3
    if features.get("ir_rain_proxy"):
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
    ar_channels = 2 if features.get("autoregressive_prev_pred") else 0
    return (
        frames * per_frame
        + frames
        + context_rows * channels_per_row(features)
        + len(SATELLITES)
        + ar_channels
    )


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
        ar_cache: dict[tuple[str, datetime], np.ndarray | torch.Tensor] | None = None,
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
        # features.autoregressive_prev_pred sourcing mode: None (default) => teacher forcing,
        # look up the true ground-truth GPM value for the same (location, T-30min) row via
        # `_row_by_location_time` below. A dict (even empty) => eval / self-prediction-
        # substituted OOF mode: source the AR channel ONLY from this externally-managed cache
        # (populated by the caller's chronological per-location sequential inference loop),
        # never from ground truth, even if `has_target` is also True. See inference.py and
        # self_pred_oof.py for the two callers that set this.
        self.ar_cache = ar_cache
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

    def _autoregressive_prev_pred_channels(self, row: dict[str, str]) -> list[torch.Tensor]:
        """Own-past-prediction autoregressive feature (2 channels: value + validity mask).

        Explicitly permitted by the organizers' 2026-07-20 ruling ("自分自身の過去(<T)の予測値の
        再利用(自己回帰/recursive手法)"): the T-30min row referenced here is always causal
        relative to the row being predicted (< T). Two sourcing modes:

        - self.ar_cache is None (teacher forcing, used during normal train/valid): look up the
          SAME split's row at (name_location, T-30min) via `_row_by_location_time` (same
          pattern used for context_rows successor lookups). If present, read its TRUE
          ground-truth GPM target -- legal because it is causal ground truth already known at
          training time, not future data.
        - self.ar_cache is a dict (evaluation, and the self-prediction-substituted OOF pass):
          the true T-30min GPM value does not exist yet for these rows (it's exactly what's
          being predicted for that row too), so ground truth is NEVER read in this mode, even
          if `has_target` happens to be True (needed for the OOF pass, which still needs GT
          for the *current* row to score against, just not for the AR *input* channel). The
          model's own just-computed prediction is looked up from the cache instead; the cache
          is populated by the caller's chronological, per-name_location sequential inference
          loop (inference.py's `run_autoregressive_inference`, self_pred_oof.py's
          `run_self_prediction_substituted_pass`). Missing key => zero + mask=0, same as any
          other missing-observation slot.
        """
        zero_value = torch.zeros((1, *self.target_size), dtype=torch.float32)
        zero_mask = torch.zeros((1, *self.target_size), dtype=torch.float32)
        if not self.features.get("autoregressive_prev_pred"):
            return []
        try:
            location = row["name_location"]
            prev_time = datetime.fromisoformat(row["datetime"]) - timedelta(minutes=30)
        except (KeyError, ValueError):
            return [zero_value, zero_mask]
        key = (location, prev_time)

        if self.ar_cache is not None:
            cached = self.ar_cache.get(key)
            if cached is None:
                return [zero_value, zero_mask]
            value = cached if torch.is_tensor(cached) else torch.from_numpy(np.asarray(cached))
            value = value.reshape(1, *self.target_size).to(dtype=torch.float32)
            return [value, torch.ones((1, *self.target_size), dtype=torch.float32)]

        prev_row = self._row_by_location_time.get(key)
        if prev_row is None or not self.has_target:
            return [zero_value, zero_mask]
        value = read_target_tensor(self.data_dir / "gpm_imerg" / prev_row["gpm_imerg_filename"])
        if value.shape[-2:] != self.target_size:
            value = _resize_chw(value, self.target_size)
        return [value, torch.ones((1, *self.target_size), dtype=torch.float32)]

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

        ar_maps = self._autoregressive_prev_pred_channels(row)
        return torch.cat(maps + masks + row_features + satellite_maps + ar_maps, dim=0)

    def input_tensor(self, row: dict[str, str]) -> torch.Tensor:
        """Public entry point used by the sequential autoregressive inference loops (they
        bypass __getitem__/index-based access since row processing order is dependency-driven,
        not index-driven, and no augmentation must ever apply at inference time)."""
        return self._input_tensor(row)

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
