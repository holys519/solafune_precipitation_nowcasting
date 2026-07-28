"""Typed, fail-fast experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    train_csv: Path
    evaluation_csv: Path
    train_dir: Path
    evaluation_dir: Path
    norm_stats: Path
    input_size: tuple[int, int] = (164, 164)
    output_size: tuple[int, int] = (41, 41)
    max_observations: int = 3
    satellite_channels: int = 16
    drop_zero_observation_rows: bool = True


@dataclass(frozen=True)
class SplitConfig:
    n_splits: int = 5
    fold: int = 0
    seed: int = 42
    stratify_bins: int = 5


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int = 32
    total_hidden: int = 128
    dropout: float = 0.1
    output_size: tuple[int, int] = (41, 41)


@dataclass(frozen=True)
class LossConfig:
    field_huber: float = 1.0
    total_huber: float = 0.35
    distribution: float = 0.15
    occurrence_bce: float = 0.1
    multiscale_mse: float = 0.1
    huber_delta: float = 1.0
    wet_threshold: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 80
    batch_size: int = 32
    num_workers: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    amp: bool = True
    patience: int = 15


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path
    data: DataConfig
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    pair = (int(value[0]), int(value[1]))
    if min(pair) <= 0:
        raise ValueError(f"{name} values must be positive, got {pair}")
    return pair


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML and reject silent, scientifically dangerous configuration drift."""

    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("configuration root must be a mapping")
    base = config_path.parent
    data = raw["data"]
    split = raw.get("split", {})
    model = raw.get("model", {})
    loss = raw.get("loss", {})
    train = raw.get("train", {})

    input_size = _pair(data.get("input_size", [164, 164]), "data.input_size")
    output_size = _pair(data.get("output_size", [41, 41]), "data.output_size")
    if input_size[0] < output_size[0] or input_size[1] < output_size[1]:
        raise ValueError("input_size must not be smaller than output_size")

    cfg = ExperimentConfig(
        name=str(raw["experiment"]["name"]),
        output_dir=_resolve(base, raw["paths"]["output_dir"]),
        data=DataConfig(
            train_csv=_resolve(base, data["train_csv"]),
            evaluation_csv=_resolve(base, data["evaluation_csv"]),
            train_dir=_resolve(base, data["train_dir"]),
            evaluation_dir=_resolve(base, data["evaluation_dir"]),
            norm_stats=_resolve(base, data["norm_stats"]),
            input_size=input_size,
            output_size=output_size,
            max_observations=int(data.get("max_observations", 3)),
            satellite_channels=int(data.get("satellite_channels", 16)),
            drop_zero_observation_rows=bool(data.get("drop_zero_observation_rows", True)),
        ),
        split=SplitConfig(
            n_splits=int(split.get("n_splits", 5)),
            fold=int(split.get("fold", 0)),
            seed=int(split.get("seed", 42)),
            stratify_bins=int(split.get("stratify_bins", 5)),
        ),
        model=ModelConfig(
            base_channels=int(model.get("base_channels", 32)),
            total_hidden=int(model.get("total_hidden", 128)),
            dropout=float(model.get("dropout", 0.1)),
            output_size=output_size,
        ),
        loss=LossConfig(**{key: value for key, value in loss.items() if key in LossConfig.__dataclass_fields__}),
        train=TrainConfig(**{key: value for key, value in train.items() if key in TrainConfig.__dataclass_fields__}),
    )
    if not 0 <= cfg.split.fold < cfg.split.n_splits:
        raise ValueError("split.fold is outside [0, n_splits)")
    if cfg.data.max_observations < 1:
        raise ValueError("max_observations must be positive")
    return cfg
