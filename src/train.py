"""Training entry point for the clean DeepHPM implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config import ExperimentConfig, load_config
from src.data import (
    BatteryDataLoaders,
    DatasetScalers,
    create_data_loaders,
    load_battery_data,
)
from src.losses import DeepHPMLoss
from src.models import DeepHPMNN, VerhulstPINN


@dataclass(frozen=True)
class LossMetrics:
    total: float
    data: float
    physics: float
    samples: int
    gradient_norm: float | None = None


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_total: float
    train_data: float
    train_physics: float
    validation_total: float
    validation_data: float
    validation_physics: float
    gradient_norm: float
    improved: bool


@dataclass(frozen=True)
class TrainingResult:
    history: list[EpochRecord]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    checkpoint_path: Path


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float) -> None:
        if patience <= 0:
            raise ValueError("patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.bad_epochs = 0

    def update(self, value: float) -> tuple[bool, bool]:
        """Return ``(improved, should_stop)`` for a new monitored value."""

        if not math.isfinite(value):
            raise FloatingPointError("Early-stopping metric is not finite")
        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _finite_loss_or_raise(losses: dict[str, torch.Tensor]) -> None:
    invalid = [name for name, value in losses.items() if not torch.isfinite(value)]
    if invalid:
        raise FloatingPointError(f"Non-finite losses detected: {invalid}")


def _iter_limited(
    loader: DataLoader, maximum_batches: int | None
) -> Iterable[dict[str, torch.Tensor]]:
    for batch_index, batch in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        yield batch


def run_epoch(
    *,
    model: DeepHPMNN | VerhulstPINN,
    loader: DataLoader,
    criterion: DeepHPMLoss,
    device: torch.device,
    optimizer: Adam | None,
    gradient_clip_norm: float,
    maximum_batches: int | None,
) -> LossMetrics:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "data": 0.0, "physics": 0.0}
    samples = 0
    gradient_norm_total = 0.0
    gradient_steps = 0

    for batch in _iter_limited(loader, maximum_batches):
        features = batch["features_normalized"].to(device)
        cycle_raw = batch["cycle_raw"].to(device)
        targets = batch["target_pcl_normalized"].to(device)
        batch_size = targets.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)
        output = model(features, cycle_raw, create_graph=training)
        loss_values = criterion(output, targets)
        loss_tensors = {
            "total": loss_values.total,
            "data": loss_values.data,
            "physics": loss_values.physics,
        }
        _finite_loss_or_raise(loss_tensors)

        if training:
            loss_values.total.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            gradient_norm_total += float(gradient_norm.detach().cpu())
            gradient_steps += 1

        for name, value in loss_tensors.items():
            totals[name] += float(value.detach().cpu()) * batch_size
        samples += batch_size

    if samples == 0:
        raise ValueError("Data loader produced no batches")
    return LossMetrics(
        total=totals["total"] / samples,
        data=totals["data"] / samples,
        physics=totals["physics"] / samples,
        samples=samples,
        gradient_norm=(gradient_norm_total / gradient_steps if training else None),
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    best_validation_loss: float,
) -> None:
    _atomic_torch_save(
        {
            "format_version": 1,
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Adam | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def save_history(history: list[EpochRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(history[0])) if history else list(EpochRecord.__annotations__)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in history)


def fit(
    *,
    model: DeepHPMNN | VerhulstPINN,
    loaders: BatteryDataLoaders,
    scalers: DatasetScalers,
    experiment_config: ExperimentConfig,
    device: torch.device,
) -> TrainingResult:
    training_config = experiment_config.training
    output_dir = Path(experiment_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    scalers.save(output_dir / "scalers.npz")
    (output_dir / "resolved_config.json").write_text(
        json.dumps(experiment_config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    model.to(device)
    criterion = DeepHPMLoss(training_config.physics_weight)
    optimizer = Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    early_stopping = EarlyStopping(
        training_config.early_stopping_patience,
        training_config.early_stopping_min_delta,
    )
    history: list[EpochRecord] = []
    best_epoch = 0
    stopped_early = False

    for epoch in range(1, training_config.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=loaders.train,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=training_config.gradient_clip_norm,
            maximum_batches=training_config.max_train_batches,
        )
        validation_metrics = run_epoch(
            model=model,
            loader=loaders.validation,
            criterion=criterion,
            device=device,
            optimizer=None,
            gradient_clip_norm=training_config.gradient_clip_norm,
            maximum_batches=training_config.max_validation_batches,
        )
        monitored_validation_loss = getattr(
            validation_metrics, training_config.early_stopping_monitor
        )
        improved, should_stop = early_stopping.update(monitored_validation_loss)
        if improved:
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_loss=monitored_validation_loss,
            )

        history.append(
            EpochRecord(
                epoch=epoch,
                train_total=train_metrics.total,
                train_data=train_metrics.data,
                train_physics=train_metrics.physics,
                validation_total=validation_metrics.total,
                validation_data=validation_metrics.data,
                validation_physics=validation_metrics.physics,
                gradient_norm=float(train_metrics.gradient_norm),
                improved=improved,
            )
        )
        print(
            f"Epoch {epoch:04d} | train={train_metrics.total:.6f} "
            f"(data={train_metrics.data:.6f}, physics={train_metrics.physics:.6f}) "
            f"| val={validation_metrics.total:.6f} "
            f"(data={validation_metrics.data:.6f}, "
            f"physics={validation_metrics.physics:.6f})"
        )
        if should_stop:
            stopped_early = True
            break

    save_history(history, output_dir / "history.csv")
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    result = TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=early_stopping.best,
        stopped_early=stopped_early,
        checkpoint_path=checkpoint_path,
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "epochs_completed": len(result.history),
                "stopped_early": result.stopped_early,
                "device": str(device),
                "early_stopping_monitor": training_config.early_stopping_monitor,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def build_model(config: ExperimentConfig, scalers: DatasetScalers) -> DeepHPMNN:
    return DeepHPMNN.from_scalers(
        scalers,
        feature_dim=8,
        surrogate_hidden_dims=config.model.surrogate_hidden_dims,
        dynamics_hidden_dims=config.model.dynamics_hidden_dims,
    )


def run_experiment(config: ExperimentConfig) -> TrainingResult:
    set_random_seed(config.training.seed)
    device = resolve_device(config.training.device)
    bundle = load_battery_data(Path(config.data.dataset_path))
    loaders = create_data_loaders(
        bundle,
        batch_size=config.data.batch_size,
        seed=config.training.seed,
        num_workers=config.data.num_workers,
    )
    model = build_model(config, bundle.scalers)
    return fit(
        model=model,
        loaders=loaders,
        scalers=bundle.scalers,
        experiment_config=config,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_experiment(config)
    print(
        f"Best epoch: {result.best_epoch}; "
        f"validation loss: {result.best_validation_loss:.6f}; "
        f"checkpoint: {result.checkpoint_path.resolve()}"
    )


if __name__ == "__main__":
    main()
