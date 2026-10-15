"""Tune and run the direct-RUL DeepHPM experiment from the Step 13 audit."""

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
from src.data import BatteryDataLoaders, DatasetScalers, create_data_loaders, load_battery_data
from src.models import DeepHPMRULNN
from src.protocol import FIXED_SEEDS
from src.run_rul_baseline import AGGREGATE_METRICS, calculate_rul_metrics, validate_rul_predictions
from src.train import EarlyStopping, load_checkpoint, resolve_device, save_checkpoint, set_random_seed


PHYSICS_WEIGHTS = (0.001, 0.01, 0.1, 1.0)
PREDICTION_COLUMNS = (
    "battery_id", "cycle", "eol_cycle", "target_rul_cycles",
    "predicted_rul_cycles", "rul_error_cycles", "time_derivative_rul_per_cycle",
    "learned_dynamics_rul_per_cycle", "physics_residual_rul_per_cycle",
)
PHYSICS_METRICS = (
    "physical_residual_mae", "physical_residual_rmse",
    "derivative_minus_one_mae", "dynamics_minus_one_mae",
)


@dataclass(frozen=True)
class DeepHPMRULConfig:
    base_experiment: ExperimentConfig
    output_dir: str
    physics_weights: tuple[float, ...] | None = None
    seeds: tuple[int, ...] | None = None
    tuning_report: str | None = None
    test_reference: str = "artifacts/rul/task_definition/rul_test_reference.csv"

    def validate_tuning(self) -> None:
        self.base_experiment.validate()
        if self.physics_weights != PHYSICS_WEIGHTS:
            raise ValueError(f"physics_weights must be {PHYSICS_WEIGHTS}")
        if Path(self.output_dir) != Path("artifacts/deephpm/rul/physics_weight_tuning"):
            raise ValueError("Unexpected DeepHPM RUL tuning output directory")
        self._validate_complete_splits()

    def validate_final(self) -> None:
        self.base_experiment.validate()
        if self.seeds != FIXED_SEEDS:
            raise ValueError(f"seeds must be {FIXED_SEEDS}")
        if Path(self.output_dir) != Path("artifacts/deephpm/rul/final"):
            raise ValueError("Unexpected DeepHPM RUL final output directory")
        self._validate_complete_splits()
        if not self.tuning_report:
            raise ValueError("tuning_report is required")
        report = json.loads(Path(self.tuning_report).read_text(encoding="utf-8"))
        if report.get("final_test_set_used") is not False:
            raise ValueError("Tuning report does not certify an untouched test set")
        if float(report["selected_physics_weight"]) != self.base_experiment.training.physics_weight:
            raise ValueError("Final physics weight differs from validation selection")

    def _validate_complete_splits(self) -> None:
        training = self.base_experiment.training
        if training.max_train_batches is not None or training.max_validation_batches is not None:
            raise ValueError("DeepHPM RUL must use complete training and validation splits")


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_total_loss: float
    train_data_loss: float
    train_physics_loss: float
    validation_total_loss: float
    validation_data_loss: float
    validation_physics_loss: float
    validation_battery_macro_rmse_cycles: float
    validation_cycle_rmse_cycles: float
    gradient_norm: float
    improved: bool


def _section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _experiment(raw: Mapping[str, Any]) -> ExperimentConfig:
    base = _section(raw, "base_experiment")
    model = _section(base, "model")
    for name in ("surrogate_hidden_dims", "dynamics_hidden_dims"):
        if name in model:
            model[name] = tuple(model[name])
    return ExperimentConfig(
        data=DataConfig(**_section(base, "data")),
        model=ModelConfig(**model),
        training=TrainingConfig(**_section(base, "training")),
        output_dir="unused",
    )


def load_tuning_config(path: Path) -> DeepHPMRULConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = DeepHPMRULConfig(
        base_experiment=_experiment(raw), output_dir=str(raw["output_dir"]),
        physics_weights=tuple(float(x) for x in raw["physics_weights"]),
    )
    config.validate_tuning()
    return config


def load_final_config(path: Path) -> DeepHPMRULConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = DeepHPMRULConfig(
        base_experiment=_experiment(raw), output_dir=str(raw["output_dir"]),
        seeds=tuple(int(x) for x in raw["seeds"]),
        tuning_report=str(raw["tuning_report"]),
        test_reference=str(raw.get("test_reference", "artifacts/rul/task_definition/rul_test_reference.csv")),
    )
    config.validate_final()
    return config


def build_model(config: ExperimentConfig, scalers: DatasetScalers) -> DeepHPMRULNN:
    return DeepHPMRULNN.from_scalers(
        scalers, feature_dim=8,
        surrogate_hidden_dims=config.model.surrogate_hidden_dims,
        dynamics_hidden_dims=config.model.dynamics_hidden_dims,
    )


def _iter_limited(loader: DataLoader, maximum: int | None) -> Iterable[dict[str, torch.Tensor]]:
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield batch


def _losses(output: Any, target: torch.Tensor, physics_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.mean(torch.square(output.rul_normalized - target))
    physics = torch.mean(torch.square(output.residual_normalized))
    return data + physics_weight * physics, data, physics


def train_epoch(model: DeepHPMRULNN, loader: DataLoader, optimizer: Adam,
                device: torch.device, physics_weight: float, clip: float,
                maximum: int | None) -> tuple[dict[str, float], float]:
    model.train(); totals = {"total": 0.0, "data": 0.0, "physics": 0.0}
    records = steps = 0; gradient_sum = 0.0
    for batch in _iter_limited(loader, maximum):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["features_normalized"].to(device), batch["cycle_raw"].to(device), create_graph=True)
        values = _losses(output, batch["target_rul_normalized"].to(device), physics_weight)
        if not all(torch.isfinite(value) for value in values):
            raise FloatingPointError("Non-finite DeepHPM RUL loss")
        values[0].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step(); count = batch["cycle_raw"].shape[0]
        for name, value in zip(("total", "data", "physics"), values):
            totals[name] += float(value.detach().cpu()) * count
        records += count; steps += 1; gradient_sum += float(norm.detach().cpu())
    if records == 0:
        raise ValueError("Empty training loader")
    return {name: value / records for name, value in totals.items()}, gradient_sum / steps


def evaluate(model: DeepHPMRULNN, loader: DataLoader, device: torch.device,
             physics_weight: float) -> tuple[dict[str, float | int], dict[str, np.ndarray], list[dict[str, float | int]], dict[str, float]]:
    model.train()  # no dropout; raw-cycle autograd remains enabled
    collected: dict[str, list[torch.Tensor]] = {
        name: [] for name in PREDICTION_COLUMNS if name != "rul_error_cycles"
    }
    normalized_errors: list[torch.Tensor] = []; normalized_residuals: list[torch.Tensor] = []
    for batch in loader:
        with torch.enable_grad():
            output = model(batch["features_normalized"].to(device), batch["cycle_raw"].to(device), create_graph=False)
        values = {
            "battery_id": batch["battery_id"], "cycle": batch["cycle_raw"],
            "eol_cycle": batch["eol_cycle"], "target_rul_cycles": batch["target_rul"],
            "predicted_rul_cycles": output.rul,
            "time_derivative_rul_per_cycle": output.time_derivative,
            "learned_dynamics_rul_per_cycle": output.dynamics,
            "physics_residual_rul_per_cycle": output.residual,
        }
        for name, value in values.items():
            collected[name].append(value.detach().cpu())
        normalized_errors.append((output.rul_normalized.detach().cpu() - batch["target_rul_normalized"]))
        normalized_residuals.append(output.residual_normalized.detach().cpu())
    columns = {name: torch.cat(parts).reshape(-1).numpy() for name, parts in collected.items()}
    for name in PREDICTION_COLUMNS:
        if name not in {"battery_id", "rul_error_cycles"}:
            columns[name] = columns[name].astype(np.float64)
    columns["rul_error_cycles"] = columns["predicted_rul_cycles"] - columns["target_rul_cycles"]
    metrics, per_battery = calculate_rul_metrics(columns)
    residual = columns["physics_residual_rul_per_cycle"]
    derivative = columns["time_derivative_rul_per_cycle"]
    dynamics = columns["learned_dynamics_rul_per_cycle"]
    metrics.update(
        physical_residual_mae=float(np.mean(np.abs(residual))),
        physical_residual_rmse=float(np.sqrt(np.mean(np.square(residual)))),
        derivative_mean=float(np.mean(derivative)),
        learned_dynamics_mean=float(np.mean(dynamics)),
        derivative_minus_one_mae=float(np.mean(np.abs(derivative + 1.0))),
        dynamics_minus_one_mae=float(np.mean(np.abs(dynamics + 1.0))),
    )
    data_loss = float(torch.mean(torch.square(torch.cat(normalized_errors))).item())
    physics_loss = float(torch.mean(torch.square(torch.cat(normalized_residuals))).item())
    loss_values = {"data": data_loss, "physics": physics_loss,
                   "weighted_physics": physics_weight * physics_loss,
                   "total": data_loss + physics_weight * physics_loss}
    return metrics, columns, per_battery, loss_values


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def fit(model: DeepHPMRULNN, loaders: BatteryDataLoaders, scalers: DatasetScalers,
        experiment: ExperimentConfig, device: torch.device) -> tuple[list[EpochRecord], int, bool]:
    output = Path(experiment.output_dir); output.mkdir(parents=True, exist_ok=True)
    scalers.save(output / "scalers.npz")
    resolved = experiment.to_dict()
    resolved.update(
        model_type="deephpm", task="rul", target="RUL_Flt = cycle_life - cycle",
        surrogate="R(features, raw cycle) in physical cycles",
        learned_dynamics="G(normalized cycle, normalized RUL) converted to cycles per cycle",
        derivative="partial dR/d(raw cycle), holding row features fixed",
        residual="dR/dcycle - G, in cycles per cycle",
        audited_ideal_trajectory_derivative=-1.0,
        checkpoint_selection="validation battery-macro RUL RMSE in cycles",
    )
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    model.to(device); training = experiment.training
    optimizer = Adam(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    stopping = EarlyStopping(training.early_stopping_patience, training.early_stopping_min_delta)
    history: list[EpochRecord] = []; best_epoch = 0; stopped = False
    for epoch in range(1, training.epochs + 1):
        train, grad = train_epoch(model, loaders.train, optimizer, device, training.physics_weight,
                                  training.gradient_clip_norm, training.max_train_batches)
        validation, _, _, val_loss = evaluate(model, loaders.validation, device, training.physics_weight)
        monitored = float(validation["battery_macro_rmse"])
        improved, should_stop = stopping.update(monitored)
        if improved:
            best_epoch = epoch
            save_checkpoint(output / "best_checkpoint.pt", model=model, optimizer=optimizer,
                            epoch=epoch, best_validation_loss=monitored)
        history.append(EpochRecord(
            epoch, train["total"], train["data"], train["physics"], val_loss["total"],
            val_loss["data"], val_loss["physics"], monitored, float(validation["rmse"]), grad, improved,
        ))
        print(f"Epoch {epoch:03d} total={train['total']:.6f} data={train['data']:.6f} physics={train['physics']:.6f} val macro RMSE={monitored:.3f}")
        if should_stop:
            stopped = True; break
    _write_rows(output / "history.csv", [asdict(row) for row in history])
    load_checkpoint(output / "best_checkpoint.pt", model=model, map_location=device)
    (output / "training_summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "epochs_completed": len(history), "stopped_early": stopped,
        "best_validation_battery_macro_rmse_cycles": stopping.best,
        "checkpoint_selection": "validation battery-macro RUL RMSE only",
        "test_set_used_during_training": False,
    }, indent=2) + "\n")
    return history, best_epoch, stopped


def save_evaluation(output: Path, metrics: dict[str, Any], columns: dict[str, np.ndarray],
                    per_battery: list[dict[str, Any]]) -> None:
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_rows(output / "per_battery_test_metrics.csv", per_battery)
    matrix = np.column_stack([columns[name] for name in PREDICTION_COLUMNS])
    np.savetxt(output / "test_predictions.csv", matrix, delimiter=",",
               header=",".join(PREDICTION_COLUMNS), comments="", fmt=("%d",) + ("%.17g",) * 8)


def run_tuning(config: DeepHPMRULConfig) -> dict[str, Any]:
    base = config.base_experiment; device = resolve_device(base.training.device)
    bundle = load_battery_data(Path(base.data.dataset_path)); rows: list[dict[str, Any]] = []
    assert config.physics_weights is not None
    for weight in config.physics_weights:
        print(f"\n=== DeepHPM RUL validation physics weight {weight:g} ===")
        set_random_seed(base.training.seed)
        loaders = create_data_loaders(bundle, batch_size=base.data.batch_size, seed=base.training.seed, num_workers=base.data.num_workers)
        run_dir = Path(config.output_dir) / f"physics_{str(weight).replace('.', 'p')}"
        experiment = replace(base, training=replace(base.training, physics_weight=weight), output_dir=str(run_dir))
        model = build_model(experiment, bundle.scalers)
        history, best_epoch, stopped = fit(model, loaders, bundle.scalers, experiment, device)
        metrics, _, _, losses = evaluate(model, loaders.validation, device, weight)
        rows.append({"physics_weight": weight, "best_epoch": best_epoch, "epochs_completed": len(history),
                     "stopped_early": stopped, **metrics, **{f"normalized_{k}_loss": v for k, v in losses.items()}})
    selected = min(rows, key=lambda row: (row["battery_macro_rmse"], row["physical_residual_rmse"]))
    output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "DeepHPM direct-RUL physics-weight tuning",
        "selection_metric": "validation battery-macro RMSE in cycles",
        "tie_breaker": "validation physical residual RMSE",
        "selected_physics_weight": selected["physics_weight"],
        "final_test_set_used": False, "shared_seed": base.training.seed, "results": rows,
    }
    (output / "tuning_report.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_rows(output / "tuning_report.csv", rows)
    lines = ["# DeepHPM RUL physics-weight tuning", "", "Test batteries were not evaluated.", "",
             "| Weight | Epoch | Validation macro RMSE | Validation RMSE | Residual RMSE | dR/dcycle vs -1 MAE |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        mark = " **selected**" if row is selected else ""
        lines.append(f"| {row['physics_weight']:g}{mark} | {row['best_epoch']} | {row['battery_macro_rmse']:.3f} | {row['rmse']:.3f} | {row['physical_residual_rmse']:.6g} | {row['derivative_minus_one_mae']:.4f} |")
    (output / "tuning_report.md").write_text("\n".join(lines) + "\n")
    return report


def run_final(config: DeepHPMRULConfig) -> dict[str, Any]:
    base = config.base_experiment; device = resolve_device(base.training.device)
    bundle = load_battery_data(Path(base.data.dataset_path)); output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True); rows: list[dict[str, Any]] = []
    assert config.seeds is not None
    for seed in config.seeds:
        print(f"\n=== Final DeepHPM RUL seed {seed} ===")
        set_random_seed(seed)
        loaders = create_data_loaders(bundle, batch_size=base.data.batch_size, seed=seed, num_workers=base.data.num_workers)
        experiment = replace(base, training=replace(base.training, seed=seed), output_dir=str(output / f"seed_{seed}"))
        model = build_model(experiment, bundle.scalers)
        history, best_epoch, stopped = fit(model, loaders, bundle.scalers, experiment, device)
        metrics, columns, per_battery, _ = evaluate(model, loaders.test, device, base.training.physics_weight)
        validate_rul_predictions(columns, Path(config.test_reference))
        save_evaluation(Path(experiment.output_dir), metrics, columns, per_battery)
        rows.append({"seed": seed, "best_epoch": best_epoch, "epochs_completed": len(history), "stopped_early": stopped, **metrics})
    names = AGGREGATE_METRICS + PHYSICS_METRICS
    aggregate = {name: {"mean": float(np.mean([float(row[name]) for row in rows])),
                        "std": float(np.std([float(row[name]) for row in rows], ddof=1))} for name in names}
    payload = {
        "experiment": "Final DeepHPM direct-RUL multi-seed evaluation",
        "target": "RUL_Flt = cycle_life - cycle, in cycles",
        "learned_dynamics_units": "cycles per cycle",
        "residual": "dRUL/d(raw cycle) - G(cycle,RUL), cycles per cycle",
        "physics_weight": base.training.physics_weight,
        "model_selection_data": "validation only",
        "test_evaluation_timing": "once per seed after checkpoint restoration",
        "per_seed": rows, "aggregate": aggregate,
    }
    (output / "final_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_rows(output / "final_results.csv", rows)
    lines = ["# Final DeepHPM RUL results", "", "RUL is predicted directly in cycles; G and the residual have units cycles/cycle.", "",
             "| Seed | Epoch | Macro RMSE | RMSE | R2 | Bias | Residual RMSE | dR/dcycle vs -1 MAE |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['best_epoch']} | {row['battery_macro_rmse']:.3f} | {row['rmse']:.3f} | {row['r2']:.5f} | {row['bias']:.3f} | {row['physical_residual_rmse']:.6f} | {row['derivative_minus_one_mae']:.4f} |")
    lines += ["", f"Aggregate macro RMSE: {aggregate['battery_macro_rmse']['mean']:.3f} +/- {aggregate['battery_macro_rmse']['std']:.3f} cycles."]
    (output / "final_results.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tune", "final")); parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "tune":
        run_tuning(load_tuning_config(args.config or Path("configs/deephpm_rul_tuning.yaml")))
    else:
        run_final(load_final_config(args.config or Path("configs/deephpm_rul_final.yaml")))


if __name__ == "__main__":
    main()
