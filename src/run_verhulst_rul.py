"""Tune and run trajectory-derived VerhulstPINN RUL experiments.

RUL is not a second unconstrained network output.  The surrogate estimates the
degradation state ``U = PCL = 1 - SoH`` and the shifted Verhulst law is
integrated analytically from the current state to ``U_eol = 0.20``.  This is
consistent with the Step 13 audit's 80% SoH intended threshold while retaining
the audited ``RUL_Flt = cycle_life - cycle`` labels as ground truth.
"""

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
from src.models import VerhulstPINN, verhulst_time_to_threshold
from src.protocol import FIXED_SEEDS
from src.run_rul_baseline import calculate_rul_metrics, validate_rul_predictions
from src.run_verhulst import VerhulstParameterConfig, build_verhulst_model
from src.train import EarlyStopping, load_checkpoint, resolve_device, save_checkpoint, set_random_seed


PHYSICS_WEIGHTS = (0.001, 0.01, 0.1, 1.0)
PREDICTION_COLUMNS = (
    "battery_id", "cycle", "eol_cycle", "target_rul_cycles",
    "predicted_rul_cycles", "rul_error_cycles", "target_pcl",
    "predicted_pcl", "time_derivative", "verhulst_rhs", "physics_residual",
    "threshold_crossing_possible", "threshold_reached", "non_crossing",
)


@dataclass(frozen=True)
class CrossingConfig:
    eol_pcl_threshold: float = 0.2
    non_crossing_horizon_cycles: float = 5000.0
    epsilon: float = 1e-6
    pcl_data_weight: float = 1.0
    non_crossing_penalty_weight: float = 1.0
    non_crossing_margin: float = 1e-4

    def validate(self) -> None:
        if not 0 < self.eol_pcl_threshold < 1:
            raise ValueError("eol_pcl_threshold must be inside (0, 1)")
        for name in (
            "non_crossing_horizon_cycles", "epsilon", "pcl_data_weight",
            "non_crossing_penalty_weight", "non_crossing_margin",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class RULTuningConfig:
    physics_weights: tuple[float, ...]
    base_experiment: ExperimentConfig
    parameters: VerhulstParameterConfig
    crossing: CrossingConfig
    output_dir: str

    def validate(self) -> None:
        if self.physics_weights != PHYSICS_WEIGHTS:
            raise ValueError(f"physics_weights must be {PHYSICS_WEIGHTS}")
        self.base_experiment.validate()
        self.parameters.validate()
        self.crossing.validate()
        if Path(self.output_dir) != Path("artifacts/verhulst/rul/physics_weight_tuning"):
            raise ValueError("Unexpected Verhulst RUL tuning output directory")
        if self.base_experiment.training.max_train_batches is not None:
            raise ValueError("Tuning must use the complete training split")
        if self.base_experiment.training.max_validation_batches is not None:
            raise ValueError("Tuning must use the complete validation split")


@dataclass(frozen=True)
class RULFinalConfig:
    seeds: tuple[int, ...]
    base_experiment: ExperimentConfig
    parameters: VerhulstParameterConfig
    crossing: CrossingConfig
    tuning_report: str
    output_dir: str
    test_reference: str

    def validate(self) -> None:
        if self.seeds != FIXED_SEEDS:
            raise ValueError(f"seeds must be {FIXED_SEEDS}")
        self.base_experiment.validate()
        self.parameters.validate()
        self.crossing.validate()
        if Path(self.output_dir) != Path("artifacts/verhulst/rul/final"):
            raise ValueError("Unexpected Verhulst RUL final output directory")
        report = json.loads(Path(self.tuning_report).read_text(encoding="utf-8"))
        if report.get("final_test_set_used") is not False:
            raise ValueError("Tuning report does not certify an untouched test set")
        if float(report["selected_physics_weight"]) != self.base_experiment.training.physics_weight:
            raise ValueError("Final physics weight differs from validation selection")


@dataclass(frozen=True)
class LossValues:
    total: torch.Tensor
    rul: torch.Tensor
    pcl: torch.Tensor
    physics: torch.Tensor
    non_crossing_penalty: torch.Tensor


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_total_loss: float
    train_rul_loss: float
    train_pcl_loss: float
    train_physics_loss: float
    train_non_crossing_penalty: float
    validation_battery_macro_rmse_cycles: float
    validation_cycle_rmse_cycles: float
    validation_physics_residual_rmse: float
    validation_non_crossing_fraction: float
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


def _parameters(raw: Mapping[str, Any]) -> VerhulstParameterConfig:
    values = _section(raw, "verhulst_parameters")
    for name in ("capacity_bounds", "offset_bounds"):
        if name in values:
            values[name] = tuple(values[name])
    return VerhulstParameterConfig(**values)


def load_tuning_config(path: Path) -> RULTuningConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = RULTuningConfig(
        physics_weights=tuple(float(x) for x in raw["physics_weights"]),
        base_experiment=_experiment(raw), parameters=_parameters(raw),
        crossing=CrossingConfig(**_section(raw, "crossing")),
        output_dir=str(raw["output_dir"]),
    )
    config.validate()
    return config


def load_final_config(path: Path) -> RULFinalConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = RULFinalConfig(
        seeds=tuple(int(x) for x in raw["seeds"]), base_experiment=_experiment(raw),
        parameters=_parameters(raw), crossing=CrossingConfig(**_section(raw, "crossing")),
        tuning_report=str(raw["tuning_report"]), output_dir=str(raw["output_dir"]),
        test_reference=str(raw["test_reference"]),
    )
    config.validate()
    return config


def _iter_limited(loader: DataLoader, maximum: int | None) -> Iterable[dict[str, torch.Tensor]]:
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield batch


def calculate_losses(
    model: VerhulstPINN, batch: Mapping[str, torch.Tensor], crossing: CrossingConfig,
    physics_weight: float, device: torch.device, *, create_graph: bool,
) -> tuple[LossValues, Any, Any]:
    output = model(batch["features_normalized"].to(device), batch["cycle_raw"].to(device), create_graph=create_graph)
    crossing_output = verhulst_time_to_threshold(
        output.pcl, model.rate, model.carrying_capacity, model.offset,
        threshold=crossing.eol_pcl_threshold,
        non_crossing_horizon=crossing.non_crossing_horizon_cycles,
        epsilon=crossing.epsilon,
    )
    target_rul_norm = batch["target_rul_normalized"].to(device)
    predicted_rul_norm = (crossing_output.rul - model.rul_mean) / model.rul_scale
    rul_loss = torch.mean(torch.square(predicted_rul_norm - target_rul_norm))
    pcl_loss = torch.mean(torch.square(output.pcl_normalized - batch["target_pcl_normalized"].to(device)))
    physics_loss = torch.mean(torch.square(output.residual_normalized))
    deficit = torch.relu(model.offset + crossing.non_crossing_margin - output.pcl) / model.target_scale
    non_crossing_penalty = torch.mean(torch.square(deficit))
    total = rul_loss + crossing.pcl_data_weight * pcl_loss + physics_weight * physics_loss + crossing.non_crossing_penalty_weight * non_crossing_penalty
    return LossValues(total, rul_loss, pcl_loss, physics_loss, non_crossing_penalty), output, crossing_output


def attach_rul_scaler(model: VerhulstPINN, scalers: DatasetScalers) -> None:
    if scalers.target_rul is None:
        raise ValueError("Training-fitted RUL scaler is required")
    model.register_buffer("rul_mean", torch.tensor([float(scalers.target_rul.mean[0])], dtype=torch.float32))
    model.register_buffer("rul_scale", torch.tensor([float(scalers.target_rul.scale[0])], dtype=torch.float32))


def train_epoch(model: VerhulstPINN, loader: DataLoader, optimizer: Adam, device: torch.device,
                crossing: CrossingConfig, physics_weight: float, clip: float,
                maximum: int | None) -> tuple[dict[str, float], float]:
    model.train()
    totals = {name: 0.0 for name in ("total", "rul", "pcl", "physics", "non_crossing_penalty")}
    records = steps = 0
    grad_total = 0.0
    for batch in _iter_limited(loader, maximum):
        optimizer.zero_grad(set_to_none=True)
        losses, _, _ = calculate_losses(model, batch, crossing, physics_weight, device, create_graph=True)
        if not torch.isfinite(losses.total):
            raise FloatingPointError("Non-finite Verhulst RUL loss")
        losses.total.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
        count = batch["cycle_raw"].shape[0]
        for name in totals:
            totals[name] += float(getattr(losses, name).detach().cpu()) * count
        records += count
        steps += 1
        grad_total += float(norm.detach().cpu())
    if records == 0:
        raise ValueError("Empty training loader")
    return {name: value / records for name, value in totals.items()}, grad_total / steps


def evaluate(model: VerhulstPINN, loader: DataLoader, device: torch.device,
             crossing: CrossingConfig) -> tuple[dict[str, float | int], dict[str, np.ndarray], list[dict[str, float | int]]]:
    model.train()  # no dropout; retain autograd required for dU/dcycle
    collected = {name: [] for name in PREDICTION_COLUMNS if name != "rul_error_cycles"}
    for batch in loader:
        with torch.enable_grad():
            output = model(batch["features_normalized"].to(device), batch["cycle_raw"].to(device), create_graph=False)
            cross = verhulst_time_to_threshold(
                output.pcl, model.rate, model.carrying_capacity, model.offset,
                threshold=crossing.eol_pcl_threshold,
                non_crossing_horizon=crossing.non_crossing_horizon_cycles,
                epsilon=crossing.epsilon,
            )
        values = {
            "battery_id": batch["battery_id"], "cycle": batch["cycle_raw"],
            "eol_cycle": batch["eol_cycle"], "target_rul_cycles": batch["target_rul"],
            "predicted_rul_cycles": cross.rul, "target_pcl": batch["target_pcl"],
            "predicted_pcl": output.pcl, "time_derivative": output.time_derivative,
            "verhulst_rhs": output.dynamics, "physics_residual": output.residual,
            "threshold_crossing_possible": cross.crossing_possible,
            "threshold_reached": cross.threshold_reached, "non_crossing": cross.non_crossing,
        }
        for name, value in values.items():
            collected[name].append(value.detach().cpu())
    columns = {name: torch.cat(parts).reshape(-1).numpy() for name, parts in collected.items()}
    for name in (
        "cycle", "eol_cycle", "target_rul_cycles", "predicted_rul_cycles",
        "target_pcl", "predicted_pcl", "time_derivative", "verhulst_rhs",
        "physics_residual",
    ):
        columns[name] = columns[name].astype(np.float64)
    # Compute the exported identity in float64 so it remains exact to the
    # validator's tolerance after decimal CSV values are parsed as float64.
    columns["rul_error_cycles"] = (
        columns["predicted_rul_cycles"].astype(np.float64)
        - columns["target_rul_cycles"].astype(np.float64)
    )
    metrics, per_battery = calculate_rul_metrics(columns)
    residual = columns["physics_residual"].astype(float)
    metrics.update(
        physical_residual_mae=float(np.mean(np.abs(residual))),
        physical_residual_rmse=float(np.sqrt(np.mean(np.square(residual)))),
        threshold_crossing_possible_count=int(np.sum(columns["threshold_crossing_possible"])),
        threshold_reached_count=int(np.sum(columns["threshold_reached"])),
        non_crossing_count=int(np.sum(columns["non_crossing"])),
        non_crossing_fraction=float(np.mean(columns["non_crossing"])),
    )
    return metrics, columns, per_battery


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def fit(model: VerhulstPINN, loaders: BatteryDataLoaders, scalers: DatasetScalers,
        experiment: ExperimentConfig, crossing: CrossingConfig, device: torch.device) -> tuple[list[EpochRecord], int, bool]:
    output = Path(experiment.output_dir); output.mkdir(parents=True, exist_ok=True)
    scalers.save(output / "scalers.npz")
    attach_rul_scaler(model, scalers)
    resolved = experiment.to_dict()
    resolved.update(model_type="verhulst", task="rul", rul_method="analytic threshold crossing",
                    state="U=PCL=1-SoH", eol_definition="U=0.20 (SoH=0.80)", crossing=asdict(crossing))
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=experiment.training.learning_rate, weight_decay=experiment.training.weight_decay)
    stopping = EarlyStopping(experiment.training.early_stopping_patience, experiment.training.early_stopping_min_delta)
    history: list[EpochRecord] = []; best_epoch = 0; stopped = False
    for epoch in range(1, experiment.training.epochs + 1):
        losses, grad = train_epoch(model, loaders.train, optimizer, device, crossing,
                                   experiment.training.physics_weight, experiment.training.gradient_clip_norm,
                                   experiment.training.max_train_batches)
        validation, _, _ = evaluate(model, loaders.validation, device, crossing)
        monitored = float(validation["battery_macro_rmse"])
        improved, should_stop = stopping.update(monitored)
        if improved:
            best_epoch = epoch
            save_checkpoint(output / "best_checkpoint.pt", model=model, optimizer=optimizer,
                            epoch=epoch, best_validation_loss=monitored)
        history.append(EpochRecord(
            epoch, losses["total"], losses["rul"], losses["pcl"], losses["physics"],
            losses["non_crossing_penalty"], monitored, float(validation["rmse"]),
            float(validation["physical_residual_rmse"]), float(validation["non_crossing_fraction"]), grad, improved,
        ))
        print(f"Epoch {epoch:03d} total={losses['total']:.5f} val macro RMSE={monitored:.3f} non-cross={validation['non_crossing_fraction']:.3%}")
        if should_stop:
            stopped = True; break
    _write_rows(output / "history.csv", [asdict(row) for row in history])
    load_checkpoint(output / "best_checkpoint.pt", model=model, map_location=device)
    learned = {
        "rate_r_per_cycle": float(model.rate.detach().cpu()),
        "capacity_K_pcl": float(model.carrying_capacity.detach().cpu()),
        "offset_C_pcl": float(model.offset.detach().cpu()),
    }
    (output / "verhulst_parameters.json").write_text(json.dumps(learned, indent=2) + "\n")
    (output / "training_summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "epochs_completed": len(history), "stopped_early": stopped,
        "best_validation_battery_macro_rmse_cycles": stopping.best,
        "checkpoint_selection": "validation battery-macro RUL RMSE only",
        "test_set_used_during_training": False, "learned_parameters": learned,
    }, indent=2) + "\n")
    return history, best_epoch, stopped


def _save_predictions(output: Path, metrics: dict[str, Any], columns: dict[str, np.ndarray],
                      per_battery: list[dict[str, Any]]) -> None:
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_rows(output / "per_battery_test_metrics.csv", per_battery)
    matrix = np.column_stack([columns[name] for name in PREDICTION_COLUMNS])
    # Preserve the algebraic error-column identity after CSV round-tripping.
    fmts = ("%d",) + ("%.17g",) * 10 + ("%d", "%d", "%d")
    np.savetxt(output / "test_predictions.csv", matrix, delimiter=",", header=",".join(PREDICTION_COLUMNS), comments="", fmt=fmts)


def _new_model(base: ExperimentConfig, scalers: DatasetScalers, parameters: VerhulstParameterConfig) -> VerhulstPINN:
    model = build_verhulst_model(base, scalers, parameters)
    return model


def run_tuning(config: RULTuningConfig) -> dict[str, Any]:
    base = config.base_experiment; device = resolve_device(base.training.device)
    bundle = load_battery_data(Path(base.data.dataset_path))
    rows: list[dict[str, Any]] = []
    for weight in config.physics_weights:
        print(f"\n=== Verhulst RUL validation physics weight {weight:g} ===")
        set_random_seed(base.training.seed)
        loaders = create_data_loaders(bundle, batch_size=base.data.batch_size, seed=base.training.seed, num_workers=base.data.num_workers)
        run_dir = Path(config.output_dir) / f"physics_{str(weight).replace('.', 'p')}"
        experiment = replace(base, training=replace(base.training, physics_weight=weight), output_dir=str(run_dir))
        model = _new_model(experiment, bundle.scalers, config.parameters)
        history, best_epoch, stopped = fit(model, loaders, bundle.scalers, experiment, config.crossing, device)
        metrics, _, _ = evaluate(model, loaders.validation, device, config.crossing)
        rows.append({"physics_weight": weight, "best_epoch": best_epoch, "epochs_completed": len(history),
                     "stopped_early": stopped, **metrics})
    selected = min(rows, key=lambda row: (row["battery_macro_rmse"], row["physical_residual_rmse"]))
    output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "Verhulst trajectory-derived RUL physics-weight tuning",
        "selection_metric": "validation battery-macro RMSE in cycles",
        "tie_breaker": "validation physics residual RMSE",
        "selected_physics_weight": selected["physics_weight"],
        "fixed_pcl_data_weight": config.crossing.pcl_data_weight,
        "final_test_set_used": False, "results": rows,
    }
    (output / "tuning_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_rows(output / "tuning_report.csv", rows)
    lines = ["# Verhulst RUL tuning", "", "Test batteries were not evaluated.", "",
             "| Physics weight | Best epoch | Validation macro RMSE | Validation RMSE | Residual RMSE | Non-crossing |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        mark = " **selected**" if row is selected else ""
        lines.append(f"| {row['physics_weight']:g}{mark} | {row['best_epoch']} | {row['battery_macro_rmse']:.3f} | {row['rmse']:.3f} | {row['physical_residual_rmse']:.8g} | {row['non_crossing_fraction']:.3%} |")
    (output / "tuning_report.md").write_text("\n".join(lines) + "\n")
    return payload


def run_final(config: RULFinalConfig) -> dict[str, Any]:
    base = config.base_experiment; device = resolve_device(base.training.device)
    bundle = load_battery_data(Path(base.data.dataset_path)); output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True); rows = []
    for seed in config.seeds:
        print(f"\n=== Final Verhulst RUL seed {seed} ===")
        set_random_seed(seed)
        loaders = create_data_loaders(bundle, batch_size=base.data.batch_size, seed=seed, num_workers=base.data.num_workers)
        experiment = replace(base, training=replace(base.training, seed=seed), output_dir=str(output_root / f"seed_{seed}"))
        model = _new_model(experiment, bundle.scalers, config.parameters)
        history, best_epoch, stopped = fit(model, loaders, bundle.scalers, experiment, config.crossing, device)
        metrics, columns, per_battery = evaluate(model, loaders.test, device, config.crossing)
        validate_rul_predictions(columns, Path(config.test_reference))
        _save_predictions(Path(experiment.output_dir), metrics, columns, per_battery)
        parameters = json.loads((Path(experiment.output_dir) / "verhulst_parameters.json").read_text())
        rows.append({"seed": seed, "best_epoch": best_epoch, "epochs_completed": len(history), "stopped_early": stopped, **metrics, **parameters})
    metric_names = ("mae", "rmse", "r2", "bias", "battery_macro_mae", "battery_macro_rmse", "battery_macro_bias",
                    "physical_residual_mae", "physical_residual_rmse", "non_crossing_fraction")
    aggregate = {name: {"mean": float(np.mean([float(row[name]) for row in rows])),
                        "std": float(np.std([float(row[name]) for row in rows], ddof=1))} for name in metric_names}
    payload = {"experiment": "Final Verhulst trajectory-derived RUL", "rul_method": "analytic PCL threshold crossing",
               "eol_pcl_threshold": config.crossing.eol_pcl_threshold, "physics_weight": base.training.physics_weight,
               "model_selection_data": "validation only", "test_evaluation_timing": "once per seed after checkpoint restoration",
               "per_seed": rows, "aggregate": aggregate}
    (output_root / "final_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_rows(output_root / "final_results.csv", rows)
    lines = ["# Final Verhulst RUL results", "", "RUL is the analytic time for the learned PCL trajectory to reach U=0.20.", "",
             "| Seed | Epoch | Macro RMSE | RMSE | R2 | Bias | Non-crossing |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['best_epoch']} | {row['battery_macro_rmse']:.3f} | {row['rmse']:.3f} | {row['r2']:.5f} | {row['bias']:.3f} | {row['non_crossing_fraction']:.3%} |")
    lines += ["", f"Aggregate macro RMSE: {aggregate['battery_macro_rmse']['mean']:.3f} +/- {aggregate['battery_macro_rmse']['std']:.3f} cycles."]
    (output_root / "final_results.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tune", "final"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "tune":
        run_tuning(load_tuning_config(args.config or Path("configs/verhulst_rul_tuning.yaml")))
    else:
        run_final(load_final_config(args.config or Path("configs/verhulst_rul_final.yaml")))


if __name__ == "__main__":
    main()
