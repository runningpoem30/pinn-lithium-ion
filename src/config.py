"""Validated YAML configuration for DeepHPM experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset_path: str = "SeversonBattery.mat"
    batch_size: int = 1024
    num_workers: int = 0

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("data.batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")


@dataclass(frozen=True)
class ModelConfig:
    surrogate_hidden_dims: tuple[int, ...] = (128, 128)
    dynamics_hidden_dims: tuple[int, ...] = (64, 64)

    def validate(self) -> None:
        for name, widths in (
            ("surrogate_hidden_dims", self.surrogate_hidden_dims),
            ("dynamics_hidden_dims", self.dynamics_hidden_dims),
        ):
            if not widths or any(width <= 0 for width in widths):
                raise ValueError(f"model.{name} must contain positive widths")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1234
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    physics_weight: float = 0.1
    gradient_clip_norm: float = 5.0
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-5
    device: str = "auto"
    max_train_batches: int | None = None
    max_validation_batches: int | None = None

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("training.epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay must be non-negative")
        if self.physics_weight < 0:
            raise ValueError("training.physics_weight must be non-negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("training.gradient_clip_norm must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("training.early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0:
            raise ValueError("training.early_stopping_min_delta must be non-negative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("training.device must be auto, cpu, cuda, or mps")
        for name, limit in (
            ("max_train_batches", self.max_train_batches),
            ("max_validation_batches", self.max_validation_batches),
        ):
            if limit is not None and limit <= 0:
                raise ValueError(f"training.{name} must be positive or null")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output_dir: str

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.training.validate()
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_config(path: Path) -> ExperimentConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    model_values = _section(raw, "model")
    for name in ("surrogate_hidden_dims", "dynamics_hidden_dims"):
        if name in model_values:
            model_values[name] = tuple(model_values[name])

    known_root_keys = {"data", "model", "training", "output_dir"}
    unknown = sorted(set(raw) - known_root_keys)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")

    config = ExperimentConfig(
        data=DataConfig(**_section(raw, "data")),
        model=ModelConfig(**model_values),
        training=TrainingConfig(**_section(raw, "training")),
        output_dir=str(raw.get("output_dir", "artifacts/deephpm/run")),
    )
    config.validate()
    return config
