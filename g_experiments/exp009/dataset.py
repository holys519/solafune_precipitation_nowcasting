"""PyTorch datasets for exp009.

Differences from exp001's dataset.py (see doc/exp001_retrospective.md for why):
- per-satellite, per-band normalization using stats from normalize_stats.py, instead of a flat
  /255 that treats every band of every sensor identically.
- anti-aliased downsampling (adaptive average pooling) instead of bilinear interpolation, since
  every satellite is always being downsized onto the coarser 41x41 IMERG grid.
- synchronized flip/rot90 augmentation for the train split (input and target are both simple
  square rasters with no fixed "up", so this is safe).
- GroupKFold-like split over name_location instead of a single fixed random holdout.
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
        context_rows: int = 1,
        norm_stats: dict[str, dict[str, np.ndarray]] | None = None,
        augment: bool = False,
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

    def _observation_tensors(self, row: dict[str, str] | None, expected_satellite: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
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
            maps.append(
                read_satellite_tensor(
                    self.data_dir / expected_satellite / obs_name,
                    satellite=expected_satellite,
                    channels=self.satellite_channels,
                    target_size=self.target_size,
                    norm_stats=self.norm_stats,
                )
            )
            masks.append(torch.ones((1, *self.target_size), dtype=torch.float32))
        return maps, masks

    def _input_tensor(self, row: dict[str, str]) -> torch.Tensor:
        satellite = row["satellite_target"]
        maps: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for context_row in self._context_rows(row):
            context_maps, context_masks = self._observation_tensors(context_row, satellite)
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
