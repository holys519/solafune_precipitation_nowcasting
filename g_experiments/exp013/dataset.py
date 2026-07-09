"""PyTorch datasets for exp013 (tickets G-017 + G-018).

On top of exp009/exp012's dataset (normalization, anti-aliased resize, augmentation,
GroupKFold, successor-row context), exp013 adds two config-gated changes:

- G-018 `data.context_offsets`: generalizes `context_rows` to arbitrary row offsets in
  30-minute steps, e.g. `[-1, 0, 1]` = predecessor + current + successor. `[0, 1]`
  reproduces exp009/exp012 exactly. Missing rows at either end become zero maps + zero masks.
- G-017 `registration`: per-(location, satellite) parallax correction. Shifts each satellite
  frame (and its mask) on the 41x41 target grid by the median (dy, dx) measured in
  `outputs/l_eda/exp001` (tables copied into this experiment dir so it stays self-contained).
  Shift semantics match l_eda's `corr_at_shift`: satellite pixel (y, x) corresponds to target
  pixel (y-dy, x-dx), so registration is `out[y, x] = sat[y+dy, x+dx]` with zero fill.
  Unseen evaluation locations inherit the shift of their nearest train location (feature-space
  nearest neighbour from `train_eval_feature_shift.csv`), falling back to the per-satellite
  median, then (0, 0).
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_observation_filenames(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed)!r}")
    return [str(item) for item in parsed]


def context_offsets_from_config(config: dict) -> list[int]:
    """Row offsets in 30-minute steps. `data.context_offsets` wins; else fall back to the
    exp009-style `data.context_rows` (N rows forward: 0..N-1); else just the current row."""

    data_cfg = config.get("data", {})
    offsets = data_cfg.get("context_offsets")
    if offsets is not None:
        offsets = [int(o) for o in offsets]
        if 0 not in offsets:
            raise ValueError("data.context_offsets must include 0 (the current row)")
        return offsets
    return list(range(int(data_cfg.get("context_rows", 1))))


def load_shift_lookup(config: dict) -> dict[str, tuple[int, int]] | None:
    """Location -> integer (dy, dx) registration shift on the target grid, or None if disabled.

    Table paths are resolved relative to this experiment directory so runs work the same from
    Slurm, run.sh, and ad-hoc invocations regardless of CWD.
    """

    reg_cfg = config.get("registration", {})
    if not reg_cfg.get("enabled", False):
        return None
    base_dir = Path(__file__).resolve().parent
    min_corr_gain = float(reg_cfg.get("min_corr_gain", 0.0))

    by_location: dict[str, tuple[int, int]] = {}
    sat_shifts: dict[str, list[tuple[float, float]]] = {}
    table_path = base_dir / str(reg_cfg.get("shift_table", "parallax_shift_by_location.csv"))
    with table_path.open(newline="") as f:
        for row in csv.DictReader(f):
            gain = float(row.get("mean_corr_gain") or "nan")
            if not np.isfinite(gain) or gain < min_corr_gain:
                continue
            dy, dx = float(row["median_dy"]), float(row["median_dx"])
            by_location[row["name_location"]] = (round(dy), round(dx))
            sat_shifts.setdefault(row["satellite_target"], []).append((dy, dx))

    by_satellite = {
        sat: (round(float(np.median([s[0] for s in shifts]))), round(float(np.median([s[1] for s in shifts]))))
        for sat, shifts in sat_shifts.items()
    }

    eval_table = reg_cfg.get("eval_location_table", "train_eval_feature_shift.csv")
    if eval_table:
        with (base_dir / str(eval_table)).open(newline="") as f:
            for row in csv.DictReader(f):
                eval_loc = row["eval_location"]
                if eval_loc in by_location:
                    continue
                nearest = by_location.get(row["nearest_train_location"])
                if nearest is not None:
                    by_location[eval_loc] = nearest
                else:
                    fallback = by_satellite.get(row["satellite_target"])
                    if fallback is not None:
                        by_location[eval_loc] = fallback
    return by_location


def _shift_hw(tensor: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Registration shift with zero fill: out[y, x] = tensor[y+dy, x+dx] (last two dims)."""

    if dy == 0 and dx == 0:
        return tensor
    h, w = tensor.shape[-2:]
    if abs(dy) >= h or abs(dx) >= w:
        return torch.zeros_like(tensor)
    out = torch.zeros_like(tensor)
    y0, y1 = max(0, -dy), h - max(0, dy)
    x0, x1 = max(0, -dx), w - max(0, dx)
    out[..., y0:y1, x0:x1] = tensor[..., y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    return out


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


def read_satellite_tensor(
    path: Path,
    satellite: str,
    channels: int,
    target_size: tuple[int, int],
    norm_stats: dict[str, dict[str, np.ndarray]] | None,
) -> torch.Tensor:
    arr, _ = read_tiff_array(path)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    n_chan = min(arr.shape[2], channels)
    arr = arr[:, :, :n_chan].astype(np.float32) / 255.0

    if norm_stats is not None and satellite in norm_stats:
        mean = norm_stats[satellite]["mean"][:n_chan]
        std = norm_stats[satellite]["std"][:n_chan]
        arr = (arr - mean) / std

    arr = np.moveaxis(arr, -1, 0)
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    if tensor.shape[0] < channels:
        pad = torch.zeros((channels - tensor.shape[0], tensor.shape[1], tensor.shape[2]), dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    return _resize_chw(tensor, target_size)


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
        context_offsets: list[int] | None = None,
        shift_lookup: dict[str, tuple[int, int]] | None = None,
        norm_stats: dict[str, dict[str, np.ndarray]] | None = None,
        augment: bool = False,
    ) -> None:
        self.rows = rows
        self.data_dir = data_dir
        self.max_observations = max_observations
        self.satellite_channels = satellite_channels
        self.target_size = target_size
        self.context_offsets = context_offsets if context_offsets is not None else [0]
        self.shift_lookup = shift_lookup
        self.has_target = has_target
        self.norm_stats = norm_stats
        self.augment = augment
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
        if self.context_offsets == [0]:
            return [row]
        try:
            location = row["name_location"]
            start_time = datetime.fromisoformat(row["datetime"])
        except (KeyError, ValueError):
            return [row if offset == 0 else None for offset in self.context_offsets]
        return [
            row
            if offset == 0
            else self._row_by_location_time.get((location, start_time + timedelta(minutes=30 * offset)))
            for offset in self.context_offsets
        ]

    def _registration_shift(self, row: dict[str, str]) -> tuple[int, int]:
        if self.shift_lookup is None:
            return (0, 0)
        return self.shift_lookup.get(row.get("name_location", ""), (0, 0))

    def _observation_tensors(
        self,
        row: dict[str, str] | None,
        expected_satellite: str,
        shift: tuple[int, int] = (0, 0),
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        maps: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        if row is None or row.get("satellite_target") != expected_satellite:
            for _ in range(self.max_observations):
                maps.append(torch.zeros((self.satellite_channels, *self.target_size), dtype=torch.float32))
                masks.append(torch.zeros((1, *self.target_size), dtype=torch.float32))
            return maps, masks

        obs_names = parse_observation_filenames(row["last_30_minutes_observation_filename"])
        obs_names = obs_names[-self.max_observations :]
        start = self.max_observations - len(obs_names)
        for slot in range(self.max_observations):
            if slot < start:
                maps.append(torch.zeros((self.satellite_channels, *self.target_size), dtype=torch.float32))
                masks.append(torch.zeros((1, *self.target_size), dtype=torch.float32))
                continue
            obs_name = obs_names[slot - start]
            obs_map = read_satellite_tensor(
                self.data_dir / expected_satellite / obs_name,
                satellite=expected_satellite,
                channels=self.satellite_channels,
                target_size=self.target_size,
                norm_stats=self.norm_stats,
            )
            obs_mask = torch.ones((1, *self.target_size), dtype=torch.float32)
            # Shift map and mask together: the zero border introduced by registration is
            # marked invalid in the mask so the model can tell padding from real zeros.
            maps.append(_shift_hw(obs_map, *shift))
            masks.append(_shift_hw(obs_mask, *shift))
        return maps, masks

    def _input_tensor(self, row: dict[str, str]) -> torch.Tensor:
        satellite = row["satellite_target"]
        shift = self._registration_shift(row)
        maps: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for context_row in self._context_rows(row):
            context_maps, context_masks = self._observation_tensors(context_row, satellite, shift=shift)
            maps.extend(context_maps)
            masks.extend(context_masks)

        satellite_maps = []
        for sat in SATELLITES:
            value = 1.0 if satellite == sat else 0.0
            satellite_maps.append(torch.full((1, *self.target_size), value, dtype=torch.float32))
        return torch.cat(maps + masks + satellite_maps, dim=0)

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
