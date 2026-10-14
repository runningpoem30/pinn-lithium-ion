"""Tune and run the protocol-aligned VerhulstPINN SoH experiment."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.data import DatasetScalers, create_data_loaders, load_battery_data
from src.evaluate import evaluate_model, save_evaluation
from src.evaluation import regression_metrics, residual_metrics
from src.models import VerhulstPINN
from src.protocol import (
    ArtifactLayout,
    COMMON_METRICS,
    FIXED_SEEDS,
    ModelType,
    SHARED_SOH_SETTINGS,
    validate_bundle_protocol,
    validate_final_seed_artifacts,
)
from src.train import fit, resolve_device, set_random_seed
from src.tune_physics import (
    ValidationMetrics,
    _weight_directory_name,
    evaluate_validation,
    select_best_result,
)


REQUIRED_PHYSICS_WEIGHTS = (0.001, 0.01, 0.1, 1.0)
PHYSICS_METRICS = ("physical_residual_mae", "physical_residual_rmse")


@dataclass(frozen=True)
class VerhulstParameterConfig:
    initial_rate: float = 0.005
    initial_capacity: float = 0.5
    initial_offset: float = 0.01
    capacity_bounds: tuple[float, float] = (0.2, 1.0)
    offset_bounds: tuple[float, float] = (0.0, 0.1)
    rate_epsilon: float = 1e-8

    def validate(self) -> None:
        capacity_lower, capacity_upper = self.capacity_bounds
        offset_lower, offset_upper = self.offset_bounds
        if self.rate_epsilon <= 0 or self.initial_rate <= self.rate_epsilon:
            raise ValueError("initial_rate must be greater than rate_epsilon > 0")
        if capacity_lower >= capacity_upper or offset_lower >= offset_upper:
            raise ValueError("Verhulst bounds must be strictly increasing")
        if not capacity_lower < self.initial_capacity < capacity_upper:
            raise ValueError("initial_capacity must be inside capacity_bounds")
        if not offset_lower < self.initial_offset < offset_upper:
            raise ValueError("initial_offset must be inside offset_bounds")
        if offset_lower < 0 or offset_upper >= capacity_lower:
            raise ValueError("Verhulst bounds must guarantee 0 <= C < K")

    def model_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerhulstTuningConfig:
    physics_weights: tuple[float, ...]
    base_experiment: ExperimentConfig
    parameters: VerhulstParameterConfig
    output_dir: str
    reference_scalers: str = "artifacts/data_scalers.npz"

    def validate(self) -> None:
        if self.physics_weights != REQUIRED_PHYSICS_WEIGHTS:
            raise ValueError(
                f"Verhulst tuning weights must be {REQUIRED_PHYSICS_WEIGHTS}"
            )
        self.base_experiment.validate()
        self.parameters.validate()
        if self.base_experiment.training.early_stopping_monitor != "data":
            raise ValueError("Tuning checkpoints must use validation data loss")
        if self.base_experiment.training.max_train_batches is not None:
            raise ValueError("Tuning must use the complete training split")
        if self.base_experiment.training.max_validation_batches is not None:
            raise ValueError("Tuning must use the complete validation split")
        expected = Path("artifacts/verhulst/physics_weight_tuning")
        if Path(self.output_dir) != expected:
            raise ValueError(f"Verhulst tuning output_dir must be {expected}")


@dataclass(frozen=True)
class VerhulstFinalConfig:
    seeds: tuple[int, ...]
    base_experiment: ExperimentConfig
    parameters: VerhulstParameterConfig
    tuning_report: str
    output_dir: str
    reference_scalers: str = "artifacts/data_scalers.npz"
    test_reference: str = "artifacts/protocol/soh_test_reference.csv"

    def validate(self, *, require_tuning_report: bool = True) -> None:
        if self.seeds != FIXED_SEEDS:
            raise ValueError(f"Verhulst seeds must be the frozen seeds {FIXED_SEEDS}")
        self.base_experiment.validate()
        self.parameters.validate()
        _validate_shared_final_settings(self.base_experiment)
        expected = ArtifactLayout().final_dir(ModelType.VERHULST)
        if Path(self.output_dir) != expected:
            raise ValueError(f"Verhulst output_dir must be {expected}")
        if require_tuning_report:
            report_path = Path(self.tuning_report)
            if not report_path.is_file():
                raise FileNotFoundError(f"Tuning report not found: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            selected = float(report["selected_physics_weight"])
            if self.base_experiment.training.physics_weight != selected:
                raise ValueError(
                    "Final physics weight does not match validation tuning report"
                )
            if report.get("final_test_set_used") is not False:
                raise ValueError("Tuning report does not certify an untouched test set")


@dataclass(frozen=True)
class VerhulstRunResult:
    seed: int
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    learned_parameters: dict[str, float | str]
    metrics: dict[str, float | int]


def _section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _load_base_experiment(raw: Mapping[str, Any]) -> ExperimentConfig:
    base = _section(raw, "base_experiment")
    model_values = _section(base, "model")
    for name in ("surrogate_hidden_dims", "dynamics_hidden_dims"):
        if name in model_values:
            model_values[name] = tuple(model_values[name])
    return ExperimentConfig(
        data=DataConfig(**_section(base, "data")),
        model=ModelConfig(**model_values),
        training=TrainingConfig(**_section(base, "training")),
        output_dir="unused_per_run_output",
    )


def _load_parameters(raw: Mapping[str, Any]) -> VerhulstParameterConfig:
    values = _section(raw, "verhulst_parameters")
    for name in ("capacity_bounds", "offset_bounds"):
        if name in values:
            values[name] = tuple(float(value) for value in values[name])
    return VerhulstParameterConfig(**values)


def load_verhulst_tuning_config(path: Path) -> VerhulstTuningConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Verhulst tuning configuration must be a mapping")
    config = VerhulstTuningConfig(
        physics_weights=tuple(float(value) for value in raw["physics_weights"]),
        base_experiment=_load_base_experiment(raw),
        parameters=_load_parameters(raw),
        output_dir=str(raw.get("output_dir", "artifacts/verhulst/physics_weight_tuning")),
        reference_scalers=str(raw.get("reference_scalers", "artifacts/data_scalers.npz")),
    )
    config.validate()
    return config


def load_verhulst_final_config(path: Path) -> VerhulstFinalConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Verhulst final configuration must be a mapping")
    config = VerhulstFinalConfig(
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        base_experiment=_load_base_experiment(raw),
        parameters=_load_parameters(raw),
        tuning_report=str(raw["tuning_report"]),
        output_dir=str(raw.get("output_dir", "artifacts/verhulst/final")),
        reference_scalers=str(raw.get("reference_scalers", "artifacts/data_scalers.npz")),
        test_reference=str(raw.get("test_reference", "artifacts/protocol/soh_test_reference.csv")),
    )
    config.validate()
    return config


def _validate_shared_final_settings(config: ExperimentConfig) -> None:
    actual = {
        "batch_size": config.data.batch_size,
        "surrogate_hidden_dims": config.model.surrogate_hidden_dims,
        "optimizer": "Adam",
        "epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "gradient_clip_norm": config.training.gradient_clip_norm,
        "early_stopping_patience": config.training.early_stopping_patience,
        "early_stopping_min_delta": config.training.early_stopping_min_delta,
        "early_stopping_monitor": config.training.early_stopping_monitor,
    }
    if actual != SHARED_SOH_SETTINGS:
        differences = {
            name: {"expected": SHARED_SOH_SETTINGS[name], "actual": actual[name]}
            for name in SHARED_SOH_SETTINGS
            if actual[name] != SHARED_SOH_SETTINGS[name]
        }
        raise ValueError(f"Verhulst differs from shared SoH settings: {differences}")
    if config.training.max_train_batches is not None:
        raise ValueError("Final training must use every training batch")
    if config.training.max_validation_batches is not None:
        raise ValueError("Final training must use every validation batch")


def build_verhulst_model(
    config: ExperimentConfig,
    scalers: DatasetScalers,
    parameters: VerhulstParameterConfig,
) -> VerhulstPINN:
    return VerhulstPINN.from_scalers(
        scalers,
        feature_dim=8,
        surrogate_hidden_dims=config.model.surrogate_hidden_dims,
        **parameters.model_kwargs(),
    )


def _learned_parameters(model: VerhulstPINN) -> dict[str, float | str]:
    return {
        "state": "U = PCL = 1 - SoH",
        "rate_r_per_cycle": float(model.rate.detach().cpu()),
        "capacity_K_pcl": float(model.carrying_capacity.detach().cpu()),
        "offset_C_pcl": float(model.offset.detach().cpu()),
        "residual_units": "PCL per physical cycle",
    }


def _annotate_run(
    output_dir: Path,
    model: VerhulstPINN,
    parameters: VerhulstParameterConfig,
) -> dict[str, float | str]:
    learned = _learned_parameters(model)
    resolved_path = output_dir / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved.update(
        {
            "model_type": ModelType.VERHULST.value,
            "state_definition": "U = PCL = 1 - SoH",
            "physics_residual": "dU/dcycle - r(U-C)(1-(U-C)/(K-C))",
            "verhulst_parameter_config": asdict(parameters),
        }
    )
    resolved_path.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    summary_path = output_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["learned_verhulst_parameters"] = learned
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "verhulst_parameters.json").write_text(
        json.dumps(learned, indent=2) + "\n", encoding="utf-8"
    )
    return learned


def _save_tuning_report(
    results: list[ValidationMetrics],
    selected: ValidationMetrics,
    config: VerhulstTuningConfig,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "VerhulstPINN SoH physics-weight tuning",
        "selection_rule": "Lowest validation SoH RMSE; physical residual RMSE is the tie-breaker",
        "selected_physics_weight": selected.physics_weight,
        "final_test_set_used": False,
        "shared_seed": config.base_experiment.training.seed,
        "candidate_weights": list(config.physics_weights),
        "results": [asdict(result) for result in results],
    }
    (output_dir / "tuning_report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "tuning_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ValidationMetrics.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    lines = [
        "# VerhulstPINN physics-weight tuning",
        "",
        "Selection used validation batteries only; final-test batteries were not evaluated.",
        "",
        "| Weight | Epoch | SoH RMSE | SoH R2 | Residual RMSE | Data loss | Physics loss | Weighted physics | Total loss |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        marker = " **selected**" if result.physics_weight == selected.physics_weight else ""
        lines.append(
            f"| {result.physics_weight:g}{marker} | {result.best_epoch} | "
            f"{result.soh_rmse:.8f} | {result.soh_r2:.6f} | "
            f"{result.physical_residual_rmse:.10f} | "
            f"{result.normalized_data_loss:.6f} | "
            f"{result.normalized_physics_loss:.6f} | "
            f"{result.weighted_physics_loss:.6f} | "
            f"{result.normalized_total_loss:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Selected physics weight: `{selected.physics_weight:g}`.",
            "",
            "This is a single-seed validation result, not a final-test result.",
        ]
    )
    (output_dir / "tuning_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_verhulst_tuning(
    config: VerhulstTuningConfig,
) -> tuple[list[ValidationMetrics], ValidationMetrics]:
    base = config.base_experiment
    device = resolve_device(base.training.device)
    bundle = load_battery_data(Path(base.data.dataset_path))
    validate_bundle_protocol(bundle, Path(config.reference_scalers))
    output_root = Path(config.output_dir)
    results: list[ValidationMetrics] = []
    for weight in config.physics_weights:
        print(f"\n=== Verhulst validation physics weight {weight:g} ===")
        set_random_seed(base.training.seed)
        loaders = create_data_loaders(
            bundle,
            batch_size=base.data.batch_size,
            seed=base.training.seed,
            num_workers=base.data.num_workers,
        )
        run_config = replace(
            base,
            training=replace(base.training, physics_weight=weight),
            output_dir=str(output_root / _weight_directory_name(weight)),
        )
        model = build_verhulst_model(run_config, bundle.scalers, config.parameters)
        training = fit(
            model=model,
            loaders=loaders,
            scalers=bundle.scalers,
            experiment_config=run_config,
            device=device,
        )
        _annotate_run(Path(run_config.output_dir), model, config.parameters)
        values = evaluate_validation(
            model=model,
            loader=loaders.validation,
            physics_weight=weight,
            device=device,
        )
        result = ValidationMetrics(
            physics_weight=weight,
            best_epoch=training.best_epoch,
            epochs_completed=len(training.history),
            stopped_early=training.stopped_early,
            **values,
        )
        results.append(result)
        print(
            f"Validation SoH RMSE={result.soh_rmse:.8f}, "
            f"R2={result.soh_r2:.6f}, residual RMSE={result.physical_residual_rmse:.10f}"
        )
    selected = select_best_result(results)
    _save_tuning_report(results, selected, config)
    return results, selected


def _complete_metrics(evaluation: Any) -> dict[str, float | int]:
    columns = evaluation.columns
    common = regression_metrics(columns["target_soh"], columns["predicted_soh"])
    physics = residual_metrics(columns["physics_residual"])
    return {
        "records": int(evaluation.metrics.records),
        "batteries": int(evaluation.metrics.batteries),
        **common,
        **physics,
        "soh_mae": common["mae"],
        "soh_rmse": common["rmse"],
        "soh_r2": common["r2"],
    }


def _save_complete_evaluation(evaluation: Any, metrics: dict[str, Any], output_dir: Path) -> None:
    save_evaluation(evaluation, output_dir)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def _load_reference(path: Path) -> dict[str, np.ndarray]:
    values = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    values = np.atleast_1d(values)
    return {name: np.asarray(values[name]) for name in values.dtype.names or ()}


def aggregate_verhulst_results(results: list[VerhulstRunResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot aggregate empty Verhulst results")
    aggregate: dict[str, Any] = {
        "num_seeds": len(results),
        "seeds": [result.seed for result in results],
        "standard_deviation": "sample (ddof=1)",
    }
    for name in COMMON_METRICS + PHYSICS_METRICS:
        values = np.asarray([float(result.metrics[name]) for result in results])
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    for name in ("rate_r_per_cycle", "capacity_K_pcl", "offset_C_pcl"):
        values = np.asarray(
            [float(result.learned_parameters[name]) for result in results]
        )
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    return aggregate


def _save_final_report(
    results: list[VerhulstRunResult],
    aggregate: dict[str, Any],
    config: VerhulstFinalConfig,
) -> None:
    output_dir = Path(config.output_dir)
    rows = [
        {
            "seed": result.seed,
            "best_epoch": result.best_epoch,
            "epochs_completed": result.epochs_completed,
            "stopped_early": result.stopped_early,
            **result.metrics,
            **{key: value for key, value in result.learned_parameters.items() if key not in {"state", "residual_units"}},
        }
        for result in results
    ]
    payload = {
        "experiment": "VerhulstPINN SoH final multi-seed evaluation",
        "model_type": ModelType.VERHULST.value,
        "physics_weight": config.base_experiment.training.physics_weight,
        "physics_weight_source": config.tuning_report,
        "model_selection_data": "validation split only",
        "test_evaluation_timing": "once per seed after best checkpoint restoration",
        "state_definition": "U = PCL = 1 - SoH",
        "per_seed": rows,
        "aggregate": aggregate,
    }
    (output_dir / "final_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "final_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Final VerhulstPINN SoH results",
        "",
        "The physics weight was selected using validation data only. The held-out test split was evaluated once per seed after restoring the validation-selected checkpoint.",
        "",
        "| Seed | Epoch | MAE | RMSE | R2 | Bias | Residual RMSE | r | K | C |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metric = result.metrics
        parameter = result.learned_parameters
        lines.append(
            f"| {result.seed} | {result.best_epoch} | {metric['mae']:.8f} | "
            f"{metric['rmse']:.8f} | {metric['r2']:.6f} | {metric['bias']:.8f} | "
            f"{metric['physical_residual_rmse']:.10f} | "
            f"{parameter['rate_r_per_cycle']:.8f} | "
            f"{parameter['capacity_K_pcl']:.6f} | {parameter['offset_C_pcl']:.6f} |"
        )
    lines.extend(["", "## Aggregate (mean +/- sample standard deviation)", ""])
    for name in COMMON_METRICS + PHYSICS_METRICS + (
        "rate_r_per_cycle", "capacity_K_pcl", "offset_C_pcl"
    ):
        values = aggregate[name]
        lines.append(f"- {name}: {values['mean']:.10f} +/- {values['std']:.10f}")
    (output_dir / "final_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_verhulst_final(
    config: VerhulstFinalConfig,
) -> tuple[list[VerhulstRunResult], dict[str, Any]]:
    config.validate()
    base = config.base_experiment
    device = resolve_device(base.training.device)
    bundle = load_battery_data(Path(base.data.dataset_path))
    validate_bundle_protocol(bundle, Path(config.reference_scalers))
    reference = _load_reference(Path(config.test_reference))
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[VerhulstRunResult] = []
    for seed in config.seeds:
        print(f"\n=== Final VerhulstPINN seed {seed} ===")
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
        model = build_verhulst_model(run_config, bundle.scalers, config.parameters)
        training = fit(
            model=model,
            loaders=loaders,
            scalers=bundle.scalers,
            experiment_config=run_config,
            device=device,
        )
        learned = _annotate_run(Path(run_config.output_dir), model, config.parameters)

        # First and only test-loader access for this seed, after checkpoint selection.
        evaluation = evaluate_model(model=model, loader=loaders.test, device=device)
        metrics = _complete_metrics(evaluation)
        run_dir = Path(run_config.output_dir)
        _save_complete_evaluation(evaluation, metrics, run_dir)
        validate_final_seed_artifacts(
            model_type=ModelType.VERHULST,
            seed=seed,
            run_dir=run_dir,
            reference_scalers=Path(config.reference_scalers),
            reference_columns=reference,
        )
        result = VerhulstRunResult(
            seed=seed,
            best_epoch=training.best_epoch,
            epochs_completed=len(training.history),
            stopped_early=training.stopped_early,
            learned_parameters=learned,
            metrics=metrics,
        )
        results.append(result)
        print(
            f"Test SoH RMSE={metrics['rmse']:.8f}, R2={metrics['r2']:.6f}, "
            f"residual RMSE={metrics['physical_residual_rmse']:.10f}"
        )
    aggregate = aggregate_verhulst_results(results)
    _save_final_report(results, aggregate, config)
    return results, aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    tune = subparsers.add_parser("tune", help="run validation-only weight tuning")
    tune.add_argument("--config", type=Path, required=True)
    final = subparsers.add_parser("final", help="run the final three-seed experiment")
    final.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "tune":
        config = load_verhulst_tuning_config(args.config)
        _, selected = run_verhulst_tuning(config)
        print(f"\nSelected Verhulst physics weight: {selected.physics_weight:g}")
        print(f"Report: {(Path(config.output_dir) / 'tuning_report.md').resolve()}")
    else:
        config = load_verhulst_final_config(args.config)
        _, aggregate = run_verhulst_final(config)
        print("\nFinal multi-seed VerhulstPINN experiment complete")
        print(
            f"SoH RMSE: {aggregate['rmse']['mean']:.8f} +/- "
            f"{aggregate['rmse']['std']:.8f}"
        )
        print(f"Report: {(Path(config.output_dir) / 'final_results.md').resolve()}")


if __name__ == "__main__":
    main()
