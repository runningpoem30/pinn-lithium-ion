"""Tune the DeepHPM physics-loss weight using validation batteries only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.data import create_data_loaders, load_battery_data
from src.models import DeepHPMNN, VerhulstPINN
from src.train import build_model, fit, resolve_device, set_random_seed


@dataclass(frozen=True)
class ValidationMetrics:
    physics_weight: float
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    records: int
    soh_rmse: float
    soh_r2: float
    physical_residual_mae: float
    physical_residual_rmse: float
    normalized_data_loss: float
    normalized_physics_loss: float
    weighted_physics_loss: float
    normalized_total_loss: float


@dataclass(frozen=True)
class PhysicsTuningConfig:
    physics_weights: tuple[float, ...]
    base_experiment: ExperimentConfig
    output_dir: str

    def validate(self) -> None:
        if not self.physics_weights:
            raise ValueError("physics_weights must not be empty")
        if any(weight < 0 for weight in self.physics_weights):
            raise ValueError("physics_weights must be non-negative")
        if len(set(self.physics_weights)) != len(self.physics_weights):
            raise ValueError("physics_weights must be unique")
        self.base_experiment.validate()
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be empty")


def _section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def load_tuning_config(path: Path) -> PhysicsTuningConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Tuning configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Tuning configuration root must be a mapping")
    base = _section(raw, "base_experiment")
    model_values = _section(base, "model")
    for name in ("surrogate_hidden_dims", "dynamics_hidden_dims"):
        if name in model_values:
            model_values[name] = tuple(model_values[name])

    base_config = ExperimentConfig(
        data=DataConfig(**_section(base, "data")),
        model=ModelConfig(**model_values),
        training=TrainingConfig(**_section(base, "training")),
        output_dir="unused_per_weight_output",
    )
    config = PhysicsTuningConfig(
        physics_weights=tuple(float(value) for value in raw["physics_weights"]),
        base_experiment=base_config,
        output_dir=str(raw.get("output_dir", "artifacts/deephpm/physics_tuning")),
    )
    config.validate()
    return config


def evaluate_validation(
    *,
    model: DeepHPMNN | VerhulstPINN,
    loader: torch.utils.data.DataLoader,
    physics_weight: float,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate one fitted model without accessing the final test loader."""

    model.eval()
    prediction_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    normalized_residual_parts: list[torch.Tensor] = []

    for batch in loader:
        output = model(
            batch["features_normalized"].to(device),
            batch["cycle_raw"].to(device),
            create_graph=False,
        )
        prediction_parts.append(output.pcl.detach().cpu())
        target_parts.append(batch["target_pcl"].cpu())
        residual_parts.append(output.residual.detach().cpu())
        normalized_residual_parts.append(output.residual_normalized.detach().cpu())

    prediction = torch.cat(prediction_parts).to(torch.float64)
    target = torch.cat(target_parts).to(torch.float64)
    residual = torch.cat(residual_parts).to(torch.float64)
    normalized_residual = torch.cat(normalized_residual_parts).to(torch.float64)
    errors = prediction - target
    squared_error_sum = torch.sum(errors.square())
    total_sum_of_squares = torch.sum((target - target.mean()).square())
    if total_sum_of_squares == 0:
        raise ValueError("Validation target is constant; R2 is undefined")

    target_scale = model.target_scale.detach().cpu().to(torch.float64)
    normalized_errors = errors / target_scale
    data_loss = torch.mean(normalized_errors.square())
    physics_loss = torch.mean(normalized_residual.square())
    weighted_physics_loss = physics_weight * physics_loss
    return {
        "records": target.numel(),
        # SoH = 1 - PCL, so PCL and SoH have identical errors and R2.
        "soh_rmse": float(torch.sqrt(squared_error_sum / target.numel())),
        "soh_r2": float(1.0 - squared_error_sum / total_sum_of_squares),
        "physical_residual_mae": float(torch.mean(torch.abs(residual))),
        "physical_residual_rmse": float(torch.sqrt(torch.mean(residual.square()))),
        "normalized_data_loss": float(data_loss),
        "normalized_physics_loss": float(physics_loss),
        "weighted_physics_loss": float(weighted_physics_loss),
        "normalized_total_loss": float(data_loss + weighted_physics_loss),
    }


def select_best_result(results: list[ValidationMetrics]) -> ValidationMetrics:
    if not results:
        raise ValueError("Cannot select from empty tuning results")
    if any(not math.isfinite(result.soh_rmse) for result in results):
        raise FloatingPointError("Cannot select from non-finite validation RMSE")
    return min(
        results,
        key=lambda result: (result.soh_rmse, result.physical_residual_rmse),
    )


def _weight_directory_name(weight: float) -> str:
    return f"lambda_{weight:g}".replace(".", "p")


def save_tuning_report(
    *,
    results: list[ValidationMetrics],
    selected: ValidationMetrics,
    output_dir: Path,
    config: PhysicsTuningConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "selection_rule": (
            "Lowest validation SoH RMSE; physical residual RMSE is the tie-breaker"
        ),
        "selected_physics_weight": selected.physics_weight,
        "final_test_set_used": False,
        "shared_seed": config.base_experiment.training.seed,
        "results": [asdict(result) for result in results],
    }
    (output_dir / "tuning_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    fieldnames = list(ValidationMetrics.__annotations__)
    with (output_dir / "tuning_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    lines = [
        "# DeepHPM physics-weight tuning",
        "",
        "Selection used validation batteries only. The final test batteries were not evaluated.",
        "",
        "| Physics weight | Epoch | SoH RMSE | SoH R2 | Residual RMSE | Data loss | Physics loss | Weighted physics | Total loss |",
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
            "This is a single-seed hyperparameter-tuning result, not a final test result.",
        ]
    )
    (output_dir / "tuning_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_tuning(config: PhysicsTuningConfig) -> tuple[list[ValidationMetrics], ValidationMetrics]:
    base = config.base_experiment
    device = resolve_device(base.training.device)
    bundle = load_battery_data(Path(base.data.dataset_path))
    results: list[ValidationMetrics] = []
    output_root = Path(config.output_dir)

    for physics_weight in config.physics_weights:
        print(f"\n=== Physics weight {physics_weight:g} ===")
        set_random_seed(base.training.seed)
        loaders = create_data_loaders(
            bundle,
            batch_size=base.data.batch_size,
            seed=base.training.seed,
            num_workers=base.data.num_workers,
        )
        run_config = replace(
            base,
            training=replace(base.training, physics_weight=physics_weight),
            output_dir=str(output_root / _weight_directory_name(physics_weight)),
        )
        model = build_model(run_config, bundle.scalers)
        training_result = fit(
            model=model,
            loaders=loaders,
            scalers=bundle.scalers,
            experiment_config=run_config,
            device=device,
        )
        metrics = evaluate_validation(
            model=model,
            loader=loaders.validation,
            physics_weight=physics_weight,
            device=device,
        )
        result = ValidationMetrics(
            physics_weight=physics_weight,
            best_epoch=training_result.best_epoch,
            epochs_completed=len(training_result.history),
            stopped_early=training_result.stopped_early,
            **metrics,
        )
        results.append(result)
        print(
            f"Validation SoH RMSE={result.soh_rmse:.8f}, "
            f"R2={result.soh_r2:.6f}, "
            f"physical residual RMSE={result.physical_residual_rmse:.10f}"
        )

    selected = select_best_result(results)
    save_tuning_report(
        results=results,
        selected=selected,
        output_dir=output_root,
        config=config,
    )
    return results, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_tuning_config(args.config)
    _, selected = run_tuning(config)
    print(f"\nSelected physics weight: {selected.physics_weight:g}")
    print(f"Report: {(Path(config.output_dir) / 'tuning_report.md').resolve()}")


if __name__ == "__main__":
    main()
