"""Train and evaluate the fixed three-seed DataDrivenNN SoH baseline."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.data import (
    BatteryDataLoaders,
    DatasetScalers,
    create_data_loaders,
    load_battery_data,
)
from src.evaluation import calculate_soh_metrics, prediction_table_from_columns
from src.models import DataDrivenNN
from src.protocol import (
    ArtifactLayout,
    COMMON_METRICS,
    CORE_PREDICTION_COLUMNS,
    FIXED_SEEDS,
    ModelType,
    SHARED_SOH_SETTINGS,
    validate_final_seed_artifacts,
)
from src.train import (
    EarlyStopping,
    EpochRecord,
    TrainingResult,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_history,
    set_random_seed,
)


@dataclass(frozen=True)
class BaselineEpochMetrics:
    data: float
    samples: int
    gradient_norm: float | None = None


@dataclass(frozen=True)
class BaselineRunResult:
    seed: int
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class BaselineFinalConfig:
    seeds: tuple[int, ...]
    base_experiment: ExperimentConfig
    output_dir: str

    def validate(self) -> None:
        if self.seeds != FIXED_SEEDS:
            raise ValueError(f"Baseline seeds must be the frozen seeds {FIXED_SEEDS}")
        base = self.base_experiment
        base.validate()
        expected = SHARED_SOH_SETTINGS
        actual = {
            "batch_size": base.data.batch_size,
            "surrogate_hidden_dims": base.model.surrogate_hidden_dims,
            "optimizer": "Adam",
            "epochs": base.training.epochs,
            "learning_rate": base.training.learning_rate,
            "weight_decay": base.training.weight_decay,
            "gradient_clip_norm": base.training.gradient_clip_norm,
            "early_stopping_patience": base.training.early_stopping_patience,
            "early_stopping_min_delta": base.training.early_stopping_min_delta,
            "early_stopping_monitor": base.training.early_stopping_monitor,
        }
        if actual != expected:
            differences = {
                name: {"expected": expected[name], "actual": actual[name]}
                for name in expected
                if actual[name] != expected[name]
            }
            raise ValueError(f"Baseline differs from shared SoH settings: {differences}")
        if base.training.physics_weight != 0.0:
            raise ValueError("DataDrivenNN physics_weight must be zero")
        if base.training.max_train_batches is not None:
            raise ValueError("Final baseline must use every training batch")
        if base.training.max_validation_batches is not None:
            raise ValueError("Final baseline must use every validation batch")


def _section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def load_baseline_config(path: Path) -> BaselineFinalConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Baseline configuration root must be a mapping")
    base = _section(raw, "base_experiment")
    model_values = _section(base, "model")
    for name in ("surrogate_hidden_dims", "dynamics_hidden_dims"):
        if name in model_values:
            model_values[name] = tuple(model_values[name])
    experiment = ExperimentConfig(
        data=DataConfig(**_section(base, "data")),
        model=ModelConfig(**model_values),
        training=TrainingConfig(**_section(base, "training")),
        output_dir="unused_per_seed_output",
    )
    config = BaselineFinalConfig(
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        base_experiment=experiment,
        output_dir=str(raw.get("output_dir", "artifacts/data_driven/final")),
    )
    config.validate()
    expected_output = ArtifactLayout().final_dir(ModelType.DATA_DRIVEN)
    if Path(config.output_dir) != expected_output:
        raise ValueError(f"Baseline output_dir must be {expected_output}")
    return config


def _iter_limited(
    loader: DataLoader, maximum_batches: int | None
) -> Iterable[dict[str, torch.Tensor]]:
    for batch_index, batch in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        yield batch


def run_baseline_epoch(
    *,
    model: DataDrivenNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: Adam | None,
    gradient_clip_norm: float,
    maximum_batches: int | None,
) -> BaselineEpochMetrics:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    samples = 0
    gradient_total = 0.0
    gradient_steps = 0
    for batch in _iter_limited(loader, maximum_batches):
        features = batch["features_normalized"].to(device)
        cycle = batch["cycle_raw"].to(device)
        target = batch["target_pcl_normalized"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(features, cycle)
            loss = torch.mean(torch.square(output.pcl_normalized - target))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite baseline data loss")
        if training:
            loss.backward()
            norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            gradient_total += float(norm.detach().cpu())
            gradient_steps += 1
        batch_size = target.shape[0]
        total += float(loss.detach().cpu()) * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("Data loader produced no baseline batches")
    return BaselineEpochMetrics(
        data=total / samples,
        samples=samples,
        gradient_norm=(gradient_total / gradient_steps if training else None),
    )


def fit_baseline(
    *,
    model: DataDrivenNN,
    loaders: BatteryDataLoaders,
    scalers: DatasetScalers,
    experiment_config: ExperimentConfig,
    device: torch.device,
) -> TrainingResult:
    training = experiment_config.training
    output_dir = Path(experiment_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    scalers.save(output_dir / "scalers.npz")
    resolved = experiment_config.to_dict()
    resolved["model_type"] = ModelType.DATA_DRIVEN.value
    resolved["loss_definition"] = "normalized PCL mean squared error only"
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )

    model.to(device)
    optimizer = Adam(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    stopping = EarlyStopping(
        training.early_stopping_patience, training.early_stopping_min_delta
    )
    history: list[EpochRecord] = []
    best_epoch = 0
    stopped_early = False
    for epoch in range(1, training.epochs + 1):
        train_metrics = run_baseline_epoch(
            model=model,
            loader=loaders.train,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=training.gradient_clip_norm,
            maximum_batches=training.max_train_batches,
        )
        validation_metrics = run_baseline_epoch(
            model=model,
            loader=loaders.validation,
            device=device,
            optimizer=None,
            gradient_clip_norm=training.gradient_clip_norm,
            maximum_batches=training.max_validation_batches,
        )
        improved, should_stop = stopping.update(validation_metrics.data)
        if improved:
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_loss=validation_metrics.data,
            )
        history.append(
            EpochRecord(
                epoch=epoch,
                train_total=train_metrics.data,
                train_data=train_metrics.data,
                train_physics=0.0,
                validation_total=validation_metrics.data,
                validation_data=validation_metrics.data,
                validation_physics=0.0,
                gradient_norm=float(train_metrics.gradient_norm),
                improved=improved,
            )
        )
        print(
            f"Epoch {epoch:04d} | train_data={train_metrics.data:.6f} "
            f"| val_data={validation_metrics.data:.6f}"
        )
        if should_stop:
            stopped_early = True
            break

    save_history(history, output_dir / "history.csv")
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    result = TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=stopping.best,
        stopped_early=stopped_early,
        checkpoint_path=checkpoint_path,
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_validation_loss": stopping.best,
                "epochs_completed": len(history),
                "stopped_early": stopped_early,
                "device": str(device),
                "early_stopping_monitor": "data",
                "physics_loss_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def evaluate_baseline(
    *, model: DataDrivenNN, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {
        "battery_id": [],
        "cycle": [],
        "target_pcl": [],
        "predicted_pcl": [],
    }
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["features_normalized"].to(device),
                batch["cycle_raw"].to(device),
            )
            collected["battery_id"].append(batch["battery_id"].cpu())
            collected["cycle"].append(batch["cycle_raw"].cpu())
            collected["target_pcl"].append(batch["target_pcl"].cpu())
            collected["predicted_pcl"].append(output.pcl.cpu())
    if not collected["target_pcl"]:
        raise ValueError("Baseline evaluation loader produced no batches")
    columns = {
        name: torch.cat(parts).reshape(-1).numpy() for name, parts in collected.items()
    }
    columns["target_soh"] = 1.0 - columns["target_pcl"]
    columns["predicted_soh"] = 1.0 - columns["predicted_pcl"]
    columns["soh_error"] = columns["predicted_soh"] - columns["target_soh"]
    table = prediction_table_from_columns(columns)
    return calculate_soh_metrics(table), table.columns


def save_baseline_evaluation(
    metrics: dict[str, float | int],
    columns: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    matrix = np.column_stack([columns[name] for name in CORE_PREDICTION_COLUMNS])
    np.savetxt(
        output_dir / "test_predictions.csv",
        matrix,
        delimiter=",",
        header=",".join(CORE_PREDICTION_COLUMNS),
        comments="",
        fmt=("%d",) + ("%.10g",) * (len(CORE_PREDICTION_COLUMNS) - 1),
    )


def aggregate_baseline_results(results: list[BaselineRunResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot aggregate empty baseline results")
    aggregate: dict[str, Any] = {
        "num_seeds": len(results),
        "seeds": [result.seed for result in results],
        "standard_deviation": "sample (ddof=1)",
    }
    for name in COMMON_METRICS:
        values = np.asarray([float(result.metrics[name]) for result in results])
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return aggregate


def save_baseline_report(
    results: list[BaselineRunResult], aggregate: dict[str, Any], output_dir: Path
) -> None:
    rows = [
        {
            "seed": result.seed,
            "best_epoch": result.best_epoch,
            "epochs_completed": result.epochs_completed,
            "stopped_early": result.stopped_early,
            **result.metrics,
        }
        for result in results
    ]
    payload = {
        "experiment": "DataDrivenNN SoH final multi-seed evaluation",
        "model_type": ModelType.DATA_DRIVEN.value,
        "loss": "normalized PCL mean squared error only",
        "model_selection_data": "validation split only",
        "test_evaluation_timing": "once per seed after best checkpoint restoration",
        "per_seed": rows,
        "aggregate": aggregate,
    }
    (output_dir / "final_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "final_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Final DataDrivenNN SoH results",
        "",
        "The final test split was evaluated once per seed after validation-based checkpoint selection.",
        "No physics loss or test-based seed selection was used.",
        "",
        "| Seed | Best epoch | MAE | RMSE | R2 | Bias |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metric = result.metrics
        lines.append(
            f"| {result.seed} | {result.best_epoch} | {metric['mae']:.8f} | "
            f"{metric['rmse']:.8f} | {metric['r2']:.6f} | {metric['bias']:.8f} |"
        )
    lines.extend(["", "## Aggregate (mean +/- sample standard deviation)", ""])
    for name in COMMON_METRICS:
        values = aggregate[name]
        lines.append(f"- {name}: {values['mean']:.10f} +/- {values['std']:.10f}")
    (output_dir / "final_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _load_reference(path: Path) -> dict[str, np.ndarray]:
    values = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    values = np.atleast_1d(values)
    return {name: np.asarray(values[name]) for name in values.dtype.names or ()}


def run_baseline_experiment(
    config: BaselineFinalConfig,
) -> tuple[list[BaselineRunResult], dict[str, Any]]:
    base = config.base_experiment
    device = resolve_device(base.training.device)
    bundle = load_battery_data(Path(base.data.dataset_path))
    output_root = Path(config.output_dir)
    results = []
    for seed in config.seeds:
        print(f"\n=== Final DataDrivenNN seed {seed} ===")
        set_random_seed(seed)
        loaders = create_data_loaders(
            bundle,
            batch_size=base.data.batch_size,
            seed=seed,
            num_workers=base.data.num_workers,
        )
        run_config = replace(
            base,
            training=replace(base.training, seed=seed),
            output_dir=str(output_root / f"seed_{seed}"),
        )
        model = DataDrivenNN.from_scalers(
            bundle.scalers,
            feature_dim=8,
            surrogate_hidden_dims=base.model.surrogate_hidden_dims,
        )
        training_result = fit_baseline(
            model=model,
            loaders=loaders,
            scalers=bundle.scalers,
            experiment_config=run_config,
            device=device,
        )

        # Deliberately first test-loader access: training and checkpoint selection
        # have completed using training and validation batteries only.
        metrics, columns = evaluate_baseline(
            model=model, loader=loaders.test, device=device
        )
        run_dir = Path(run_config.output_dir)
        save_baseline_evaluation(metrics, columns, run_dir)
        validate_final_seed_artifacts(
            model_type=ModelType.DATA_DRIVEN,
            seed=seed,
            run_dir=run_dir,
            reference_scalers=Path("artifacts/data_scalers.npz"),
            reference_columns=_load_reference(
                Path("artifacts/protocol/soh_test_reference.csv")
            ),
        )
        result = BaselineRunResult(
            seed=seed,
            best_epoch=training_result.best_epoch,
            epochs_completed=len(training_result.history),
            stopped_early=training_result.stopped_early,
            metrics=metrics,
        )
        results.append(result)
        print(
            f"Test SoH RMSE={metrics['rmse']:.8f}, "
            f"R2={metrics['r2']:.6f}, bias={metrics['bias']:.8f}"
        )
    aggregate = aggregate_baseline_results(results)
    save_baseline_report(results, aggregate, output_root)
    return results, aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_baseline_config(args.config)
    _, aggregate = run_baseline_experiment(config)
    print("\nFinal multi-seed DataDrivenNN experiment complete")
    print(
        f"SoH RMSE: {aggregate['rmse']['mean']:.8f} +/- "
        f"{aggregate['rmse']['std']:.8f}"
    )
    print(f"Report: {(Path(config.output_dir) / 'final_results.md').resolve()}")


if __name__ == "__main__":
    main()
