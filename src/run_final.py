"""Run the fixed, multi-seed DeepHPM SoH experiment."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.data import create_data_loaders, load_battery_data
from src.evaluate import EvaluationMetrics, evaluate_model, save_evaluation
from src.train import build_model, fit, resolve_device, set_random_seed


@dataclass(frozen=True)
class FinalRunResult:
    seed: int
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    metrics: EvaluationMetrics


@dataclass(frozen=True)
class FinalExperimentConfig:
    seeds: tuple[int, ...]
    base_experiment: ExperimentConfig
    output_dir: str

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.base_experiment.training.physics_weight != 0.001:
            raise ValueError("Final Step 6 physics_weight must be 0.001")
        if self.base_experiment.training.max_train_batches is not None:
            raise ValueError("Final training must use all training batches")
        if self.base_experiment.training.max_validation_batches is not None:
            raise ValueError("Final training must use all validation batches")
        self.base_experiment.validate()


def _section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def load_final_config(path: Path) -> FinalExperimentConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Final experiment configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Final experiment configuration root must be a mapping")
    base = _section(raw, "base_experiment")
    model_values = _section(base, "model")
    for name in ("surrogate_hidden_dims", "dynamics_hidden_dims"):
        if name in model_values:
            model_values[name] = tuple(model_values[name])
    base_config = ExperimentConfig(
        data=DataConfig(**_section(base, "data")),
        model=ModelConfig(**model_values),
        training=TrainingConfig(**_section(base, "training")),
        output_dir="unused_per_seed_output",
    )
    config = FinalExperimentConfig(
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        base_experiment=base_config,
        output_dir=str(raw.get("output_dir", "artifacts/deephpm/final")),
    )
    config.validate()
    return config


def _flat_result(result: FinalRunResult) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "stopped_early": result.stopped_early,
        **asdict(result.metrics),
    }


def aggregate_results(results: list[FinalRunResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot aggregate empty final results")
    metric_names = (
        "soh_mae",
        "soh_rmse",
        "soh_r2",
        "physical_residual_mae",
        "physical_residual_rmse",
    )
    aggregate: dict[str, Any] = {
        "num_seeds": len(results),
        "seeds": [result.seed for result in results],
        "standard_deviation": "sample (ddof=1) when more than one seed",
    }
    for name in metric_names:
        values = np.asarray([getattr(result.metrics, name) for result in results])
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    return aggregate


def save_final_report(
    results: list[FinalRunResult], aggregate: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_results = [_flat_result(result) for result in results]
    payload = {
        "experiment": "DeepHPM SoH final multi-seed evaluation",
        "physics_weight": 0.001,
        "model_selection_data": "validation split only",
        "test_evaluation_timing": "once per seed after best checkpoint restoration",
        "per_seed": flat_results,
        "aggregate": aggregate,
    }
    (output_dir / "final_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "final_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_results[0]))
        writer.writeheader()
        writer.writerows(flat_results)

    lines = [
        "# Final DeepHPM SoH results",
        "",
        "The final test split was evaluated once per seed after validation-based checkpoint selection.",
        "No best seed was selected or discarded.",
        "",
        "| Seed | Best epoch | SoH MAE | SoH RMSE | SoH R2 | Residual MAE | Residual RMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metric = result.metrics
        lines.append(
            f"| {result.seed} | {result.best_epoch} | {metric.soh_mae:.8f} | "
            f"{metric.soh_rmse:.8f} | {metric.soh_r2:.6f} | "
            f"{metric.physical_residual_mae:.10f} | "
            f"{metric.physical_residual_rmse:.10f} |"
        )
    lines.extend(["", "## Aggregate (mean +/- sample standard deviation)", ""])
    for name in (
        "soh_mae",
        "soh_rmse",
        "soh_r2",
        "physical_residual_mae",
        "physical_residual_rmse",
    ):
        values = aggregate[name]
        lines.append(f"- {name}: {values['mean']:.10f} +/- {values['std']:.10f}")
    (output_dir / "final_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_final_experiment(
    config: FinalExperimentConfig,
) -> tuple[list[FinalRunResult], dict[str, Any]]:
    base = config.base_experiment
    device = resolve_device(base.training.device)
    bundle = load_battery_data(Path(base.data.dataset_path))
    output_root = Path(config.output_dir)
    results: list[FinalRunResult] = []

    for seed in config.seeds:
        print(f"\n=== Final DeepHPM seed {seed} ===")
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
        model = build_model(run_config, bundle.scalers)
        training_result = fit(
            model=model,
            loaders=loaders,
            scalers=bundle.scalers,
            experiment_config=run_config,
            device=device,
        )

        # This is deliberately the first and only access to the test loader for
        # this seed. It occurs after validation selected and restored a checkpoint.
        evaluation = evaluate_model(
            model=model,
            loader=loaders.test,
            device=device,
        )
        run_output_dir = Path(run_config.output_dir)
        save_evaluation(evaluation, run_output_dir)
        result = FinalRunResult(
            seed=seed,
            best_epoch=training_result.best_epoch,
            epochs_completed=len(training_result.history),
            stopped_early=training_result.stopped_early,
            metrics=evaluation.metrics,
        )
        results.append(result)
        print(
            f"Test SoH RMSE={result.metrics.soh_rmse:.8f}, "
            f"R2={result.metrics.soh_r2:.6f}, "
            f"residual RMSE={result.metrics.physical_residual_rmse:.10f}"
        )

    aggregate = aggregate_results(results)
    save_final_report(results, aggregate, output_root)
    return results, aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_final_config(args.config)
    _, aggregate = run_final_experiment(config)
    print("\nFinal multi-seed DeepHPM experiment complete")
    print(
        f"SoH RMSE: {aggregate['soh_rmse']['mean']:.8f} +/- "
        f"{aggregate['soh_rmse']['std']:.8f}"
    )
    print(f"Report: {(Path(config.output_dir) / 'final_results.md').resolve()}")


if __name__ == "__main__":
    main()
