"""Train and evaluate the frozen three-seed DataDrivenNN RUL baseline."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
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
from src.evaluation import regression_metrics
from src.models import DataDrivenRULNN
from src.protocol import FIXED_SEEDS
from src.rul_audit import RUL_PREDICTION_COLUMNS
from src.train import EarlyStopping, TrainingResult, load_checkpoint, resolve_device
from src.train import save_checkpoint, set_random_seed


AGGREGATE_METRICS = (
    "mae",
    "rmse",
    "r2",
    "bias",
    "signed_error_p05",
    "signed_error_p50",
    "signed_error_p95",
    "absolute_error_p50",
    "absolute_error_p90",
    "absolute_error_p95",
    "absolute_error_p99",
    "battery_macro_mae",
    "battery_macro_rmse",
    "battery_macro_bias",
    "negative_prediction_fraction",
)


@dataclass(frozen=True)
class RULEpochRecord:
    epoch: int
    train_data_mse_normalized: float
    validation_data_mse_normalized: float
    validation_battery_macro_rmse_cycles: float
    validation_cycle_weighted_rmse_cycles: float
    gradient_norm: float
    improved: bool


@dataclass(frozen=True)
class RULRunResult:
    seed: int
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class RULBaselineConfig:
    seeds: tuple[int, ...]
    base_experiment: ExperimentConfig
    output_dir: str

    def validate(self) -> None:
        if self.seeds != FIXED_SEEDS:
            raise ValueError(f"RUL baseline seeds must be {FIXED_SEEDS}")
        self.base_experiment.validate()
        training = self.base_experiment.training
        if training.physics_weight != 0.0:
            raise ValueError("DataDrivenNN RUL physics_weight must be zero")
        if training.max_train_batches is not None:
            raise ValueError("Final RUL baseline must use every training batch")
        if training.max_validation_batches is not None:
            raise ValueError("Final RUL baseline must use every validation batch")
        if Path(self.output_dir) != Path("artifacts/data_driven/rul/final"):
            raise ValueError(
                "RUL baseline output_dir must be artifacts/data_driven/rul/final"
            )


def _section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def load_rul_baseline_config(path: Path) -> RULBaselineConfig:
    if not path.is_file():
        raise FileNotFoundError(f"RUL baseline configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("RUL baseline configuration root must be a mapping")
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
    config = RULBaselineConfig(
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        base_experiment=experiment,
        output_dir=str(raw.get("output_dir", "artifacts/data_driven/rul/final")),
    )
    config.validate()
    return config


def _iter_limited(
    loader: DataLoader, maximum_batches: int | None
) -> Iterable[dict[str, torch.Tensor]]:
    for batch_index, batch in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        yield batch


def run_rul_training_epoch(
    *,
    model: DataDrivenRULNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: Adam,
    gradient_clip_norm: float,
    maximum_batches: int | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    samples = 0
    gradient_total = 0.0
    steps = 0
    for batch in _iter_limited(loader, maximum_batches):
        features = batch["features_normalized"].to(device)
        cycle = batch["cycle_raw"].to(device)
        target = batch["target_rul_normalized"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(features, cycle)
        loss = torch.mean(torch.square(output.rul_normalized - target))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite RUL training loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        count = target.shape[0]
        total_loss += float(loss.detach().cpu()) * count
        samples += count
        gradient_total += float(norm.detach().cpu())
        steps += 1
    if samples == 0:
        raise ValueError("RUL training loader produced no batches")
    return total_loss / samples, gradient_total / steps


def calculate_rul_metrics(
    columns: Mapping[str, np.ndarray],
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    target = np.asarray(columns["target_rul_cycles"], dtype=np.float64)
    prediction = np.asarray(columns["predicted_rul_cycles"], dtype=np.float64)
    battery = np.asarray(columns["battery_id"], dtype=np.int64)
    common = regression_metrics(target, prediction)
    per_battery: list[dict[str, float | int]] = []
    for battery_id in np.unique(battery):
        mask = battery == battery_id
        row = regression_metrics(target[mask], prediction[mask])
        per_battery.append(
            {
                "battery_id": int(battery_id),
                "records": int(mask.sum()),
                **row,
            }
        )
    metrics: dict[str, float | int] = {
        "records": int(target.size),
        "batteries": len(per_battery),
        **common,
        "battery_macro_mae": float(np.mean([row["mae"] for row in per_battery])),
        "battery_macro_rmse": float(np.mean([row["rmse"] for row in per_battery])),
        "battery_macro_bias": float(np.mean([row["bias"] for row in per_battery])),
        "negative_prediction_count": int(np.sum(prediction < 0)),
        "negative_prediction_fraction": float(np.mean(prediction < 0)),
    }
    if not all(
        np.isfinite(value)
        for value in metrics.values()
        if isinstance(value, float)
    ):
        raise FloatingPointError("Non-finite RUL evaluation metric")
    return metrics, per_battery


def evaluate_rul_baseline(
    *, model: DataDrivenRULNN, loader: DataLoader, device: torch.device
) -> tuple[
    dict[str, float | int], dict[str, np.ndarray], list[dict[str, float | int]], float
]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {
        "battery_id": [],
        "cycle": [],
        "eol_cycle": [],
        "target_rul_cycles": [],
        "predicted_rul_cycles": [],
        "target_rul_normalized": [],
        "predicted_rul_normalized": [],
    }
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["features_normalized"].to(device),
                batch["cycle_raw"].to(device),
            )
            collected["battery_id"].append(batch["battery_id"].cpu())
            collected["cycle"].append(batch["cycle_raw"].cpu())
            collected["eol_cycle"].append(batch["eol_cycle"].cpu())
            collected["target_rul_cycles"].append(batch["target_rul"].cpu())
            collected["predicted_rul_cycles"].append(output.rul.cpu())
            collected["target_rul_normalized"].append(
                batch["target_rul_normalized"].cpu()
            )
            collected["predicted_rul_normalized"].append(
                output.rul_normalized.cpu()
            )
    if not collected["target_rul_cycles"]:
        raise ValueError("RUL evaluation loader produced no batches")
    raw = {
        name: torch.cat(parts).reshape(-1).numpy()
        for name, parts in collected.items()
    }
    columns = {
        name: raw[name]
        for name in RUL_PREDICTION_COLUMNS
        if name != "rul_error_cycles"
    }
    columns["rul_error_cycles"] = (
        columns["predicted_rul_cycles"] - columns["target_rul_cycles"]
    )
    metrics, per_battery = calculate_rul_metrics(columns)
    normalized_mse = float(
        np.mean(
            np.square(
                raw["predicted_rul_normalized"] - raw["target_rul_normalized"]
            )
        )
    )
    return metrics, columns, per_battery, normalized_mse


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_history(history: list[RULEpochRecord], path: Path) -> None:
    _write_rows(path, [asdict(row) for row in history])


def fit_rul_baseline(
    *,
    model: DataDrivenRULNN,
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
    resolved.update(
        model_type="data_driven",
        task="rul",
        target="RUL_Flt in physical cycles",
        loss_definition="training-only-normalized RUL mean squared error",
        checkpoint_selection="battery-macro validation RMSE in physical cycles",
    )
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
    history: list[RULEpochRecord] = []
    best_epoch = 0
    stopped_early = False
    for epoch in range(1, training.epochs + 1):
        train_mse, gradient_norm = run_rul_training_epoch(
            model=model,
            loader=loaders.train,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=training.gradient_clip_norm,
            maximum_batches=training.max_train_batches,
        )
        validation, _, _, validation_mse = evaluate_rul_baseline(
            model=model, loader=loaders.validation, device=device
        )
        monitored = float(validation["battery_macro_rmse"])
        improved, should_stop = stopping.update(monitored)
        if improved:
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_loss=monitored,
            )
        history.append(
            RULEpochRecord(
                epoch=epoch,
                train_data_mse_normalized=train_mse,
                validation_data_mse_normalized=validation_mse,
                validation_battery_macro_rmse_cycles=monitored,
                validation_cycle_weighted_rmse_cycles=float(validation["rmse"]),
                gradient_norm=gradient_norm,
                improved=improved,
            )
        )
        print(
            f"Epoch {epoch:04d} | train normalized MSE={train_mse:.6f} "
            f"| val macro RMSE={monitored:.3f} cycles "
            f"| val cycle RMSE={validation['rmse']:.3f} cycles"
        )
        if should_stop:
            stopped_early = True
            break

    _save_history(history, output_dir / "history.csv")
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    result = TrainingResult(
        history=history,  # type: ignore[arg-type]
        best_epoch=best_epoch,
        best_validation_loss=stopping.best,
        stopped_early=stopped_early,
        checkpoint_path=checkpoint_path,
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_validation_battery_macro_rmse_cycles": stopping.best,
                "epochs_completed": len(history),
                "stopped_early": stopped_early,
                "device": str(device),
                "early_stopping_monitor": "battery-macro validation RMSE in cycles",
                "physics_loss_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def validate_rul_predictions(
    columns: Mapping[str, np.ndarray], reference_path: Path
) -> None:
    reference = np.genfromtxt(reference_path, delimiter=",", names=True)
    reference = np.atleast_1d(reference)
    order = np.lexsort((columns["cycle"], columns["battery_id"]))
    for name in ("battery_id", "cycle", "eol_cycle", "target_rul_cycles"):
        actual = np.asarray(columns[name])[order]
        expected = np.asarray(reference[name])
        if not np.array_equal(actual, expected):
            raise ValueError(f"RUL prediction column {name} differs from frozen reference")
    if not np.allclose(
        columns["rul_error_cycles"],
        columns["predicted_rul_cycles"] - columns["target_rul_cycles"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("rul_error_cycles violates its definition")


def save_rul_evaluation(
    *,
    metrics: dict[str, float | int],
    columns: dict[str, np.ndarray],
    per_battery: list[dict[str, float | int]],
    output_dir: Path,
) -> None:
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    _write_rows(output_dir / "per_battery_test_metrics.csv", per_battery)
    matrix = np.column_stack([columns[name] for name in RUL_PREDICTION_COLUMNS])
    np.savetxt(
        output_dir / "test_predictions.csv",
        matrix,
        delimiter=",",
        header=",".join(RUL_PREDICTION_COLUMNS),
        comments="",
        fmt=("%d", "%.10g", "%.10g", "%.10g", "%.10g", "%.10g"),
    )


def aggregate_rul_results(results: list[RULRunResult]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "num_seeds": len(results),
        "seeds": [result.seed for result in results],
        "standard_deviation": "sample (ddof=1)",
    }
    for name in AGGREGATE_METRICS:
        values = np.asarray([float(result.metrics[name]) for result in results])
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return aggregate


def save_final_report(
    results: list[RULRunResult], aggregate: dict[str, Any], output_dir: Path
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
        "experiment": "DataDrivenNN RUL final multi-seed evaluation",
        "model_type": "data_driven",
        "task": "rul",
        "units": "cycles",
        "loss": "training-normalized RUL mean squared error only",
        "checkpoint_selection": "battery-macro validation RMSE only",
        "test_evaluation_timing": "once per seed after best checkpoint restoration",
        "per_seed": rows,
        "aggregate": aggregate,
    }
    (output_dir / "final_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_rows(output_dir / "final_results.csv", rows)
    lines = [
        "# Final DataDrivenNN RUL results",
        "",
        "Checkpoints were selected by battery-macro validation RMSE. The final",
        "test batteries were evaluated once per seed without prediction clipping.",
        "",
        "| Seed | Best epoch | Macro MAE | Macro RMSE | Cycle RMSE | R2 | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metric = result.metrics
        lines.append(
            f"| {result.seed} | {result.best_epoch} | "
            f"{metric['battery_macro_mae']:.3f} | "
            f"{metric['battery_macro_rmse']:.3f} | {metric['rmse']:.3f} | "
            f"{metric['r2']:.5f} | {metric['bias']:.3f} |"
        )
    lines.extend(["", "## Aggregate (mean +/- sample standard deviation)", ""])
    for name in AGGREGATE_METRICS:
        value = aggregate[name]
        lines.append(f"- {name}: {value['mean']:.6f} +/- {value['std']:.6f}")
    (output_dir / "final_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_rul_baseline_experiment(
    config: RULBaselineConfig,
) -> tuple[list[RULRunResult], dict[str, Any]]:
    base = config.base_experiment
    device = resolve_device(base.training.device)
    bundle = load_battery_data(Path(base.data.dataset_path))
    if bundle.scalers.target_rul is None:
        raise RuntimeError("RUL scaler was not created")
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    reference = Path("artifacts/rul/task_definition/rul_test_reference.csv")
    results: list[RULRunResult] = []
    for seed in config.seeds:
        print(f"\n=== Final DataDrivenNN RUL seed {seed} ===")
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
        model = DataDrivenRULNN.from_scalers(
            bundle.scalers,
            feature_dim=len(bundle.audit_report["feature_names"]),
            surrogate_hidden_dims=base.model.surrogate_hidden_dims,
        )
        training_result = fit_rul_baseline(
            model=model,
            loaders=loaders,
            scalers=bundle.scalers,
            experiment_config=run_config,
            device=device,
        )
        metrics, columns, per_battery, _ = evaluate_rul_baseline(
            model=model, loader=loaders.test, device=device
        )
        validate_rul_predictions(columns, reference)
        run_dir = Path(run_config.output_dir)
        save_rul_evaluation(
            metrics=metrics,
            columns=columns,
            per_battery=per_battery,
            output_dir=run_dir,
        )
        result = RULRunResult(
            seed=seed,
            best_epoch=training_result.best_epoch,
            epochs_completed=len(training_result.history),
            stopped_early=training_result.stopped_early,
            metrics=metrics,
        )
        results.append(result)
        print(
            f"Test macro RMSE={metrics['battery_macro_rmse']:.3f} cycles, "
            f"cycle RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.5f}"
        )
    aggregate = aggregate_rul_results(results)
    save_final_report(results, aggregate, output_root)
    return results, aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/data_driven_rul_final.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_rul_baseline_config(args.config)
    _, aggregate = run_rul_baseline_experiment(config)
    metric = aggregate["battery_macro_rmse"]
    print("\nFinal multi-seed DataDrivenNN RUL experiment complete")
    print(f"Macro RMSE: {metric['mean']:.3f} +/- {metric['std']:.3f} cycles")
    print(f"Report: {(Path(config.output_dir) / 'final_results.md').resolve()}")


if __name__ == "__main__":
    main()
