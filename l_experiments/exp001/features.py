"""Feature extraction for exp001 baseline."""

from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np

from tiff_utils import read_tiff_array


SATELLITES = ("goes", "himawari", "meteosat")


def parse_observation_filenames(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed)!r}")
    return [str(item) for item in parsed]


def feature_names(max_observations: int = 3) -> list[str]:
    names: list[str] = []
    for idx in range(max_observations):
        names.extend([f"obs{idx}_mean", f"obs{idx}_std", f"obs{idx}_mask"])
    names.append("latest_minus_earliest_mean")
    names.extend(["x_coord", "y_coord"])
    names.extend([f"sat_{sat}" for sat in SATELLITES])
    return names


def _resize_map(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = target_shape
    return cv2.resize(image.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)


def _satellite_maps(path: Path, target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    arr, _ = read_tiff_array(path)
    arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 2:
        mean_map = arr
        std_map = np.zeros_like(arr, dtype=np.float32)
    else:
        mean_map = arr.mean(axis=2)
        std_map = arr.std(axis=2)
    return _resize_map(mean_map, target_shape), _resize_map(std_map, target_shape)


def sample_feature_maps(
    row: dict[str, str],
    split_dir: Path,
    target_shape: tuple[int, int] = (41, 41),
    max_observations: int = 3,
) -> np.ndarray:
    """Return H x W x F feature maps for one sample."""

    height, width = target_shape
    obs_names = parse_observation_filenames(row["last_30_minutes_observation_filename"])
    obs_names = obs_names[-max_observations:]

    feature_maps: list[np.ndarray] = []
    mean_slots: list[np.ndarray | None] = [None] * max_observations
    start = max_observations - len(obs_names)
    satellite = row["satellite_target"]

    for slot in range(max_observations):
        if slot < start:
            feature_maps.append(np.zeros((height, width), dtype=np.float32))
            feature_maps.append(np.zeros((height, width), dtype=np.float32))
            feature_maps.append(np.zeros((height, width), dtype=np.float32))
            continue

        obs_name = obs_names[slot - start]
        obs_path = split_dir / satellite / obs_name
        mean_map, std_map = _satellite_maps(obs_path, target_shape)
        mean_slots[slot] = mean_map
        feature_maps.append(mean_map)
        feature_maps.append(std_map)
        feature_maps.append(np.ones((height, width), dtype=np.float32))

    available_means = [item for item in mean_slots if item is not None]
    if len(available_means) >= 2:
        diff_map = available_means[-1] - available_means[0]
    else:
        diff_map = np.zeros((height, width), dtype=np.float32)
    feature_maps.append(diff_map)

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    feature_maps.extend([xx, yy])

    for sat in SATELLITES:
        value = 1.0 if satellite == sat else 0.0
        feature_maps.append(np.full((height, width), value, dtype=np.float32))

    return np.stack(feature_maps, axis=2).astype(np.float32)

