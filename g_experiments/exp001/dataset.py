"""PyTorch datasets for exp001."""

from __future__ import annotations

import ast
import csv
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


def make_location_split(
    rows: list[dict[str, str]],
    valid_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    locations = sorted({row["name_location"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(locations)
    n_valid = max(1, round(len(locations) * valid_fraction))
    valid_locations = set(locations[:n_valid])
    train_rows = [row for row in rows if row["name_location"] not in valid_locations]
    valid_rows = [row for row in rows if row["name_location"] in valid_locations]
    return train_rows, valid_rows, sorted(valid_locations)


def sample_rows(rows: list[dict[str, str]], max_samples: int | None, seed: int) -> list[dict[str, str]]:
    if not max_samples or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_samples)


def _resize_chw(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


def read_satellite_tensor(
    path: Path,
    channels: int,
    target_size: tuple[int, int],
) -> torch.Tensor:
    arr, _ = read_tiff_array(path)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.astype(np.float32) / 255.0
    arr = np.moveaxis(arr, -1, 0)
    tensor = torch.from_numpy(arr)
    if tensor.shape[0] < channels:
        pad = torch.zeros((channels - tensor.shape[0], tensor.shape[1], tensor.shape[2]), dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    elif tensor.shape[0] > channels:
        tensor = tensor[:channels]
    return _resize_chw(tensor, target_size)


def read_target_tensor(path: Path) -> torch.Tensor:
    arr, _ = read_tiff_array(path)
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


class PrecipDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        data_dir: Path,
        max_observations: int,
        satellite_channels: int,
        target_size: tuple[int, int],
        has_target: bool,
    ) -> None:
        self.rows = rows
        self.data_dir = data_dir
        self.max_observations = max_observations
        self.satellite_channels = satellite_channels
        self.target_size = target_size
        self.has_target = has_target

    def __len__(self) -> int:
        return len(self.rows)

    def _input_tensor(self, row: dict[str, str]) -> torch.Tensor:
        satellite = row["satellite_target"]
        obs_names = parse_observation_filenames(row["last_30_minutes_observation_filename"])
        obs_names = obs_names[-self.max_observations :]
        start = self.max_observations - len(obs_names)
        maps: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for slot in range(self.max_observations):
            if slot < start:
                maps.append(torch.zeros((self.satellite_channels, *self.target_size), dtype=torch.float32))
                masks.append(torch.zeros((1, *self.target_size), dtype=torch.float32))
                continue
            obs_name = obs_names[slot - start]
            maps.append(
                read_satellite_tensor(
                    self.data_dir / satellite / obs_name,
                    channels=self.satellite_channels,
                    target_size=self.target_size,
                )
            )
            masks.append(torch.ones((1, *self.target_size), dtype=torch.float32))

        satellite_maps = []
        for sat in SATELLITES:
            value = 1.0 if satellite == sat else 0.0
            satellite_maps.append(torch.full((1, *self.target_size), value, dtype=torch.float32))
        return torch.cat(maps + masks + satellite_maps, dim=0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x = self._input_tensor(row)
        item: dict[str, Any] = {
            "x": x,
            "unique_id": row["unique_id"],
            "gpm_imerg_filename": row["gpm_imerg_filename"],
        }
        if self.has_target:
            item["y"] = read_target_tensor(self.data_dir / "gpm_imerg" / row["gpm_imerg_filename"])
        return item

