"""Dataset and split utilities that preserve satellite spatial information."""

from __future__ import annotations

import ast
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SATELLITES = ("goes", "himawari", "meteosat")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def observation_names(row: dict[str, str]) -> list[str]:
    value = ast.literal_eval(row["last_30_minutes_observation_filename"])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"invalid observation list for {row.get('unique_id')}")
    return value


def has_observation(row: dict[str, str]) -> bool:
    try:
        return bool(observation_names(row))
    except (KeyError, ValueError, SyntaxError):
        return False


@dataclass(frozen=True)
class NormalizationStats:
    mean: dict[str, torch.Tensor]
    std: dict[str, torch.Tensor]

    @classmethod
    def load(cls, path: Path) -> "NormalizationStats":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            mean={sat: torch.tensor(values["mean"], dtype=torch.float32) for sat, values in raw.items()},
            std={sat: torch.tensor(values["std"], dtype=torch.float32) for sat, values in raw.items()},
        )


def _read_image(path: Path) -> np.ndarray:
    array = tifffile.imread(path)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"expected HWC TIFF, got {array.shape}: {path}")
    return np.asarray(array)


def _resize_chw(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tensor.shape[-2:] == size:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False, antialias=True
    ).squeeze(0)


class PrecipitationDataset(Dataset[dict[str, Any]]):
    """Load each source frame at model input resolution, never target resolution."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        root: Path,
        stats: NormalizationStats,
        input_size: tuple[int, int],
        output_size: tuple[int, int],
        max_observations: int = 3,
        satellite_channels: int = 16,
        has_target: bool = True,
        augment: bool = False,
    ) -> None:
        self.rows = rows
        self.root = root
        self.stats = stats
        self.input_size = input_size
        self.output_size = output_size
        self.max_observations = max_observations
        self.satellite_channels = satellite_channels
        self.has_target = has_target
        self.augment = augment

    @property
    def in_channels(self) -> int:
        return self.max_observations * (self.satellite_channels + 1) + len(SATELLITES)

    def __len__(self) -> int:
        return len(self.rows)

    def _frame(self, path: Path, satellite: str) -> torch.Tensor:
        array = _read_image(path)
        present = min(array.shape[-1], self.satellite_channels)
        tensor = torch.from_numpy(
            np.ascontiguousarray(np.moveaxis(array[..., :present].astype(np.float32), -1, 0))
        )
        tensor = _resize_chw(tensor, self.input_size) / 255.0
        mean = self.stats.mean[satellite][:present, None, None]
        std = self.stats.std[satellite][:present, None, None].clamp_min(1e-6)
        tensor = (tensor - mean) / std
        if present < self.satellite_channels:
            padding = tensor.new_zeros(
                self.satellite_channels - present, *self.input_size
            )
            tensor = torch.cat((tensor, padding))
        return tensor

    def _input(self, row: dict[str, str]) -> torch.Tensor:
        satellite = row["satellite_target"]
        names = observation_names(row)[-self.max_observations :]
        frames: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        missing = self.max_observations - len(names)
        for _ in range(missing):
            frames.append(torch.zeros(self.satellite_channels, *self.input_size))
            masks.append(torch.zeros(1, *self.input_size))
        for name in names:
            frames.append(self._frame(self.root / satellite / name, satellite))
            masks.append(torch.ones(1, *self.input_size))
        satellite_maps = [
            torch.full((1, *self.input_size), float(satellite == candidate))
            for candidate in SATELLITES
        ]
        return torch.cat((*frames, *masks, *satellite_maps))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        x = self._input(row)
        y: torch.Tensor | None = None
        if self.has_target:
            array = tifffile.imread(self.root / "gpm_imerg" / row["gpm_imerg_filename"])
            y = torch.from_numpy(np.asarray(array, dtype=np.float32)).view(1, *self.output_size)
        if self.augment:
            if random.random() < 0.5:
                x = x.flip(-1)
                y = None if y is None else y.flip(-1)
            if random.random() < 0.5:
                x = x.flip(-2)
                y = None if y is None else y.flip(-2)
        return {
            "x": x,
            "y": y,
            "unique_id": row["unique_id"],
            "location": row.get("name_location", ""),
            "satellite": row["satellite_target"],
        }


def make_balanced_group_folds(
    rows: Iterable[dict[str, str]], n_splits: int, seed: int
) -> dict[str, int]:
    """Balance sample count and satellite composition without leaking locations.

    Primary key is strictly "fewest rows assigned so far" (the same greedy
    bin-packing every existing `g_experiments`/`l_experiments` split uses, proven
    across dozens of experiments) so every fold is guaranteed non-empty and total
    counts stay close to `n_rows / n_splits`. Satellite composition only breaks
    ties among folds that currently have exactly the same row count -- this is
    where it actually matters (mostly the first few assignments, while several
    folds are still empty) and it cannot override the count balance the way a
    weighted-sum cost did previously.

    An earlier version combined count deviation and satellite-mix deviation into
    a single weighted cost (`count_cost + 0.25 * satellite_cost`) and picked the
    single cheapest fold for every location. That let satellite-mix cheapness
    dominate: once two folds held the same single satellite, the cost function
    could keep preferring to grow already-large, satellite-diverse folds over
    ever touching a still-empty one, silently collapsing 5-way splits to 3 used
    folds on this project's real 20-location/3-satellite distribution (every
    seed tried) -- caught by `tests/test_research_pipeline.py::test_group_folds_never_split_a_location`.

    Target-derived stratification must be handled by a separately persisted fold
    manifest. This function deliberately uses only inference-available metadata.
    """

    profiles: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        profiles[row["name_location"]][row["satellite_target"]] += 1
    if len(profiles) < n_splits:
        raise ValueError("fewer location groups than folds")

    rng = random.Random(seed)
    locations = list(profiles)
    rng.shuffle(locations)
    locations.sort(key=lambda loc: sum(profiles[loc].values()), reverse=True)
    fold_profiles = [Counter() for _ in range(n_splits)]
    fold_totals = [0] * n_splits
    assignment: dict[str, int] = {}

    def satellite_cost(fold: int, location: str) -> float:
        candidate = fold_profiles[fold] + profiles[location]
        total = sum(candidate.values())
        return sum((candidate[sat] / total - 1 / len(SATELLITES)) ** 2 for sat in SATELLITES)

    for location in locations:
        min_total = min(fold_totals)
        tied = [idx for idx in range(n_splits) if fold_totals[idx] == min_total]
        fold = tied[0] if len(tied) == 1 else min(tied, key=lambda idx: (satellite_cost(idx, location), idx))
        assignment[location] = fold
        fold_totals[fold] += sum(profiles[location].values())
        fold_profiles[fold].update(profiles[location])
    return assignment
