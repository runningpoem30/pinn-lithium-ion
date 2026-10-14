"""Compare frozen DataDrivenNN, VerhulstPINN, and DeepHPMNN SoH results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analyze_final import FORMATS, _sample_indices, _style, save_figure
from src.evaluation import (
    align_prediction_tables,
    load_prediction_csv,
    regression_metrics,
    residual_metrics,
)
from src.protocol import COMMON_METRICS, FIXED_SEEDS


MODEL_SPECS = {
    "data_driven": {
        "label": "DataDrivenNN",
        "path": Path("artifacts/data_driven/final"),
        "physics": False,
        "color": "#0072B2",
    },
    "verhulst": {
        "label": "VerhulstPINN",
        "path": Path("artifacts/verhulst/final"),
        "physics": True,
        "color": "#D55E00",
    },
    "deephpm": {
        "label": "DeepHPMNN",
        "path": Path("artifacts/deephpm/final"),
        "physics": True,
        "color": "#009E73",
    },
}

COMPARISON_FIGURES = (
    "01_metric_means_with_seed_error_bars",
    "02_measured_vs_predicted_soh",
    "03_representative_battery_trajectories",
    "04_per_battery_rmse",
    "05_error_distributions",
    "06_seed_stability",
    "07_physics_residuals",
)


@dataclass(frozen=True)
class ModelComparisonData:
    key: str
    label: str
    color: str
    seeds: tuple[int, ...]
    battery_id: np.ndarray
    cycle: np.ndarray
    target_soh: np.ndarray
    predicted_soh: np.ndarray
    physics_residual: np.ndarray | None

    @property
    def ensemble(self) -> np.ndarray:
        return self.predicted_soh.mean(axis=0)


@dataclass(frozen=True)
class ComparisonData:
    models: dict[str, ModelComparisonData]
    battery_id: np.ndarray
    cycle: np.ndarray
    target_soh: np.ndarray


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_comparison_data(
    model_specs: dict[str, dict[str, Any]] = MODEL_SPECS,
) -> ComparisonData:
    """Load and jointly align every model/seed table by battery and cycle."""

    tables = {}
    for key, spec in model_specs.items():
        for seed in FIXED_SEEDS:
            path = Path(spec["path"]) / f"seed_{seed}" / "test_predictions.csv"
            tables[f"{key}:{seed}"] = load_prediction_csv(
                path, require_physics=bool(spec["physics"])
            )
    aligned = align_prediction_tables(tables)
    reference = aligned[f"data_driven:{FIXED_SEEDS[0]}"].columns
    models = {}
    for key, spec in model_specs.items():
        seed_tables = [aligned[f"{key}:{seed}"].columns for seed in FIXED_SEEDS]
        models[key] = ModelComparisonData(
            key=key,
            label=str(spec["label"]),
            color=str(spec["color"]),
            seeds=FIXED_SEEDS,
            battery_id=reference["battery_id"].astype(np.int64),
            cycle=reference["cycle"].astype(np.float64),
            target_soh=reference["target_soh"].astype(np.float64),
            predicted_soh=np.stack(
                [table["predicted_soh"].astype(np.float64) for table in seed_tables]
            ),
            physics_residual=(
                np.stack(
                    [
                        table["physics_residual"].astype(np.float64)
                        for table in seed_tables
                    ]
                )
                if spec["physics"]
                else None
            ),
        )
    return ComparisonData(
        models=models,
        battery_id=reference["battery_id"].astype(np.int64),
        cycle=reference["cycle"].astype(np.float64),
        target_soh=reference["target_soh"].astype(np.float64),
    )


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(np.mean([float(row[name]) for row in rows])),
            "sample_std": float(np.std([float(row[name]) for row in rows], ddof=1)),
        }
        for name in COMMON_METRICS
    }


def calculate_comparison(data: ComparisonData) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    per_seed: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    ensembles: list[dict[str, Any]] = []
    per_battery: list[dict[str, Any]] = []
    physics_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}

    for model in data.models.values():
        seed_rows = []
        for index, seed in enumerate(model.seeds):
            metrics = regression_metrics(data.target_soh, model.predicted_soh[index])
            row = {
                "model": model.key,
                "model_label": model.label,
                "seed": seed,
                "records": int(data.target_soh.size),
                "batteries": int(np.unique(data.battery_id).size),
                **metrics,
            }
            per_seed.append(row)
            seed_rows.append(row)
            if model.physics_residual is not None:
                physics_rows.append(
                    {
                        "model": model.key,
                        "model_label": model.label,
                        "level": "seed",
                        "seed": seed,
                        **residual_metrics(model.physics_residual[index]),
                    }
                )

        seed_summary = _metric_summary(seed_rows)
        aggregate_row: dict[str, Any] = {
            "model": model.key,
            "model_label": model.label,
            "seeds": len(seed_rows),
        }
        for name, values in seed_summary.items():
            aggregate_row[f"{name}_mean"] = values["mean"]
            aggregate_row[f"{name}_sample_std"] = values["sample_std"]
        aggregate.append(aggregate_row)

        ensemble_metrics = regression_metrics(data.target_soh, model.ensemble)
        ensembles.append(
            {
                "model": model.key,
                "model_label": model.label,
                "records": int(data.target_soh.size),
                "batteries": int(np.unique(data.battery_id).size),
                **ensemble_metrics,
            }
        )
        for battery_id in np.unique(data.battery_id):
            mask = data.battery_id == battery_id
            per_battery.append(
                {
                    "model": model.key,
                    "model_label": model.label,
                    "battery_id": int(battery_id),
                    "records": int(mask.sum()),
                    **regression_metrics(
                        data.target_soh[mask], model.ensemble[mask]
                    ),
                }
            )

        physics_summary = None
        if model.physics_residual is not None:
            ensemble_residual = model.physics_residual.mean(axis=0)
            ensemble_residual_metrics = residual_metrics(ensemble_residual)
            physics_rows.append(
                {
                    "model": model.key,
                    "model_label": model.label,
                    "level": "ensemble_mean_residual",
                    "seed": "",
                    **ensemble_residual_metrics,
                }
            )
            seed_physics = [
                row
                for row in physics_rows
                if row["model"] == model.key and row["level"] == "seed"
            ]
            physics_summary = {
                "per_seed": seed_physics,
                "seed_mean": {
                    name: float(np.mean([row[name] for row in seed_physics]))
                    for name in ("physical_residual_mae", "physical_residual_rmse")
                },
                "seed_sample_std": {
                    name: float(np.std([row[name] for row in seed_physics], ddof=1))
                    for name in ("physical_residual_mae", "physical_residual_rmse")
                },
                "ensemble_mean_residual": ensemble_residual_metrics,
            }

        uncertainty = model.predicted_soh.std(axis=0, ddof=1)
        model_summaries[model.key] = {
            "label": model.label,
            "per_seed_metrics": seed_rows,
            "seed_aggregate_metrics": seed_summary,
            "ensemble_metrics": ensemble_metrics,
            "prediction_uncertainty": {
                "mean_seed_std": float(uncertainty.mean()),
                "p95_seed_std": float(np.percentile(uncertainty, 95)),
                "max_seed_std": float(uncertainty.max()),
            },
            "physics_residual_metrics": physics_summary,
        }

    score_by_battery = []
    for battery_id in np.unique(data.battery_id):
        rows = [row for row in per_battery if row["battery_id"] == battery_id]
        score_by_battery.append(
            (int(battery_id), float(np.mean([row["rmse"] for row in rows])))
        )
    ordered = sorted(score_by_battery, key=lambda item: item[1])
    positions = (0, round((len(ordered) - 1) / 2), len(ordered) - 1)
    representatives = [ordered[position][0] for position in positions]

    cohort_matrix = np.column_stack(
        (data.battery_id.astype(np.float64), data.cycle, data.target_soh)
    ).astype("<f8")
    baseline_seed = model_summaries["data_driven"]["seed_aggregate_metrics"]
    baseline_ensemble = model_summaries["data_driven"]["ensemble_metrics"]
    relative_to_baseline = {}
    for key, model_summary in model_summaries.items():
        relative_to_baseline[key] = {
            "seed_mean_mae_reduction_percent": 100.0
            * (
                baseline_seed["mae"]["mean"]
                - model_summary["seed_aggregate_metrics"]["mae"]["mean"]
            )
            / baseline_seed["mae"]["mean"],
            "seed_mean_rmse_reduction_percent": 100.0
            * (
                baseline_seed["rmse"]["mean"]
                - model_summary["seed_aggregate_metrics"]["rmse"]["mean"]
            )
            / baseline_seed["rmse"]["mean"],
            "ensemble_rmse_reduction_percent": 100.0
            * (baseline_ensemble["rmse"] - model_summary["ensemble_metrics"]["rmse"])
            / baseline_ensemble["rmse"],
        }
    summary = {
        "comparison_method": "saved predictions only; no model loading or retraining",
        "alignment": {
            "verified": True,
            "key": ["battery_id", "cycle"],
            "targets_identical": True,
            "records": int(data.target_soh.size),
            "batteries": int(np.unique(data.battery_id).size),
            "battery_ids": np.unique(data.battery_id).astype(int).tolist(),
            "seeds": list(FIXED_SEEDS),
            "cohort_sha256": hashlib.sha256(cohort_matrix.tobytes()).hexdigest(),
        },
        "metric_definition": "cycle-weighted metrics from the frozen shared protocol",
        "ensemble_definition": "equal-weight arithmetic mean of three aligned seed predictions",
        "models": model_summaries,
        "relative_to_data_driven": relative_to_baseline,
        "representative_batteries": {
            "selection": "lowest, median, and highest mean ensemble RMSE across models",
            "battery_ids": representatives,
        },
        "ranking_by_seed_mean_rmse": [
            row["model"] for row in sorted(aggregate, key=lambda row: row["rmse_mean"])
        ],
        "ranking_by_ensemble_rmse": [
            row["model"] for row in sorted(ensembles, key=lambda row: row["rmse"])
        ],
    }
    return summary, per_seed, aggregate, ensembles, per_battery, physics_rows


def _plot_metric_means(aggregate: list[dict[str, Any]]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    metrics = (("mae", "MAE"), ("rmse", "RMSE"), ("r2", r"$R^2$"), ("bias", "Bias"))
    x = np.arange(len(aggregate))
    colors = [MODEL_SPECS[row["model"]]["color"] for row in aggregate]
    labels = [row["model_label"] for row in aggregate]
    for axis, (name, title) in zip(axes.flat, metrics, strict=True):
        means = [row[f"{name}_mean"] for row in aggregate]
        errors = [row[f"{name}_sample_std"] for row in aggregate]
        axis.bar(x, means, yerr=errors, capsize=5, color=colors, alpha=0.9)
        axis.set(xticks=x, xticklabels=labels, title=title, ylabel=title)
        axis.tick_params(axis="x", rotation=10)
        if name == "bias":
            axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1)
    fig.suptitle("SoH test metrics: mean ± sample SD across three fixed seeds")
    fig.tight_layout()
    return fig


def _plot_parity(data: ComparisonData, ensembles: list[dict[str, Any]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    index = _sample_indices(data.target_soh.size)
    predictions = [model.ensemble for model in data.models.values()]
    lower = min(data.target_soh.min(), *(prediction.min() for prediction in predictions))
    upper = max(data.target_soh.max(), *(prediction.max() for prediction in predictions))
    metrics_by_model = {row["model"]: row for row in ensembles}
    for axis, model in zip(axes, data.models.values(), strict=True):
        metrics = metrics_by_model[model.key]
        axis.scatter(
            data.target_soh[index], model.ensemble[index], s=8, alpha=0.24,
            color=model.color, edgecolors="none",
        )
        axis.plot([lower, upper], [lower, upper], "--", color="#222222")
        axis.text(
            0.04, 0.96,
            f"RMSE {metrics['rmse']:.4f}\nMAE {metrics['mae']:.4f}\n$R^2$ {metrics['r2']:.3f}",
            transform=axis.transAxes, va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#BBBBBB"},
        )
        axis.set_title(model.label)
        axis.set_xlabel("Measured SoH")
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Three-seed ensemble predicted SoH")
    fig.suptitle("Measured versus predicted SoH on identical held-out cycles")
    fig.tight_layout()
    return fig


def _plot_trajectories(data: ComparisonData, representatives: list[int]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    descriptions = ("Lowest shared error", "Median shared error", "Highest shared error")
    line_styles = {"data_driven": "--", "verhulst": "-", "deephpm": ":"}
    for axis, battery_id, description in zip(axes, representatives, descriptions, strict=True):
        mask = data.battery_id == battery_id
        cycles = data.cycle[mask]
        axis.plot(cycles, data.target_soh[mask], color="#222222", linewidth=2.2, label="Measured")
        for model in data.models.values():
            predictions = model.predicted_soh[:, mask]
            mean = predictions.mean(axis=0)
            std = predictions.std(axis=0, ddof=1)
            axis.plot(
                cycles, mean, color=model.color, linestyle=line_styles[model.key],
                linewidth=2.0, label=model.label,
            )
            axis.fill_between(cycles, mean - std, mean + std, color=model.color, alpha=0.10)
        axis.set(title=f"Battery {battery_id}\n{description}", xlabel="Cycle")
    axes[0].set_ylabel("SoH")
    axes[-1].legend(fontsize=9)
    fig.suptitle("Representative held-out degradation trajectories (bands: ±1 seed SD)")
    fig.tight_layout()
    return fig


def _plot_per_battery(per_battery: list[dict[str, Any]]) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(13, 5.8))
    offsets = (-0.20, 0.0, 0.20)
    markers = ("o", "s", "^")
    for (key, spec), offset, marker in zip(
        MODEL_SPECS.items(), offsets, markers, strict=True
    ):
        rows = sorted(
            [row for row in per_battery if row["model"] == key],
            key=lambda row: row["battery_id"],
        )
        axis.plot(
            [row["battery_id"] + offset for row in rows],
            [row["rmse"] for row in rows],
            marker=marker, markersize=3.8, linewidth=1.25,
            color=spec["color"], label=spec["label"],
        )
    axis.set(
        xlabel="Held-out battery ID",
        ylabel="Ensemble SoH RMSE",
        title="Per-battery RMSE on the identical test cohort",
    )
    axis.set_xticks(sorted(np.unique([row["battery_id"] for row in per_battery]))[::2])
    axis.legend()
    return fig


def _plot_error_distributions(data: ComparisonData) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    line_styles = {"data_driven": "--", "verhulst": "-", "deephpm": ":"}
    for model in data.models.values():
        error = model.ensemble - data.target_soh
        axes[0].hist(
            error, bins=70, density=True, histtype="step", linewidth=1.8,
            color=model.color, linestyle=line_styles[model.key], label=model.label,
        )
        absolute = np.sort(np.abs(error))
        axes[1].plot(
            absolute, np.arange(1, absolute.size + 1) / absolute.size,
            color=model.color, linestyle=line_styles[model.key], linewidth=2.2,
            label=model.label,
        )
    axes[0].axvline(0.0, color="#222222", linestyle="--")
    axes[0].set(xlabel="Signed SoH error", ylabel="Density", title="Signed-error distribution")
    axes[1].set(xlabel="Absolute SoH error", ylabel="Cumulative fraction", title="Absolute-error ECDF")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Three-seed ensemble error distributions")
    fig.tight_layout()
    return fig


def _plot_seed_stability(per_seed: list[dict[str, Any]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(FIXED_SEEDS))
    width = 0.24
    for axis, (metric, title) in zip(
        axes, (("mae", "MAE"), ("rmse", "RMSE"), ("r2", r"$R^2$")), strict=True
    ):
        for model_index, (key, spec) in enumerate(MODEL_SPECS.items()):
            rows = sorted(
                [row for row in per_seed if row["model"] == key],
                key=lambda row: row["seed"],
            )
            axis.bar(
                x + (model_index - 1) * width,
                [row[metric] for row in rows],
                width, color=spec["color"], label=spec["label"],
            )
        axis.set(
            xticks=x, xticklabels=[str(seed) for seed in FIXED_SEEDS],
            xlabel="Fixed seed", ylabel=title, title=title,
        )
    axes[0].legend()
    fig.suptitle("Model stability across the same three random seeds")
    fig.tight_layout()
    return fig


def _plot_physics_residuals(data: ComparisonData) -> plt.Figure:
    physics_models = [model for model in data.models.values() if model.physics_residual is not None]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sample = _sample_indices(data.cycle.size, maximum=5000)
    x = np.arange(len(FIXED_SEEDS))
    width = 0.34
    for model_index, model in enumerate(physics_models):
        assert model.physics_residual is not None
        mean_residual = model.physics_residual.mean(axis=0)
        axes[0].scatter(
            data.cycle[sample], mean_residual[sample], s=7, alpha=0.18,
            color=model.color, edgecolors="none", label=model.label,
        )
        rmses = [
            residual_metrics(model.physics_residual[index])["physical_residual_rmse"]
            for index in range(len(FIXED_SEEDS))
        ]
        axes[1].bar(
            x + (model_index - 0.5) * width, rmses, width,
            color=model.color, label=model.label,
        )
    axes[0].axhline(0.0, color="#222222", linestyle="--")
    axes[0].set(
        xlabel="Cycle", ylabel="Ensemble-mean residual (PCL/cycle)",
        title="Residual across cycle life",
    )
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].set(
        xticks=x, xticklabels=[str(seed) for seed in FIXED_SEEDS],
        xlabel="Fixed seed", ylabel="Residual RMSE (PCL/cycle)",
        title="Physics residual by seed",
    )
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Physics consistency of the two physics-informed models")
    fig.tight_layout()
    return fig


def _save_aligned_predictions(data: ComparisonData, path: Path) -> None:
    names = ["battery_id", "cycle", "target_soh"]
    arrays: list[np.ndarray] = [data.battery_id, data.cycle, data.target_soh]
    for model in data.models.values():
        names.extend((f"{model.key}_predicted_soh_mean", f"{model.key}_predicted_soh_std"))
        arrays.extend((model.ensemble, model.predicted_soh.std(axis=0, ddof=1)))
        if model.physics_residual is not None:
            names.extend((f"{model.key}_physics_residual_mean", f"{model.key}_physics_residual_std"))
            arrays.extend(
                (
                    model.physics_residual.mean(axis=0),
                    model.physics_residual.std(axis=0, ddof=1),
                )
            )
    np.savetxt(
        path,
        np.column_stack(arrays),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt=("%d",) + ("%.10g",) * (len(names) - 1),
    )


def _write_markdown(
    summary: dict[str, Any], aggregate: list[dict[str, Any]], ensembles: list[dict[str, Any]], path: Path
) -> None:
    aggregate_by_model = {row["model"]: row for row in aggregate}
    ensemble_by_model = {row["model"]: row for row in ensembles}
    lines = [
        "# Final SoH model comparison",
        "",
        "This report was calculated from saved test predictions only; no model was retrained.",
        "All nine prediction files were aligned by `(battery_id, cycle)` and verified to contain identical targets.",
        "",
        "## Test metrics",
        "",
        "| Model | Seed-mean MAE ± SD | Seed-mean RMSE ± SD | Seed-mean R2 ± SD | Ensemble RMSE | Ensemble R2 | Ensemble bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, spec in MODEL_SPECS.items():
        aggregate_row = aggregate_by_model[key]
        ensemble_row = ensemble_by_model[key]
        lines.append(
            f"| {spec['label']} | {aggregate_row['mae_mean']:.6f} ± {aggregate_row['mae_sample_std']:.6f} | "
            f"{aggregate_row['rmse_mean']:.6f} ± {aggregate_row['rmse_sample_std']:.6f} | "
            f"{aggregate_row['r2_mean']:.4f} ± {aggregate_row['r2_sample_std']:.4f} | "
            f"{ensemble_row['rmse']:.6f} | {ensemble_row['r2']:.4f} | {ensemble_row['bias']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Relative to DataDrivenNN",
            "",
            f"- VerhulstPINN seed-mean RMSE reduction: {summary['relative_to_data_driven']['verhulst']['seed_mean_rmse_reduction_percent']:.3f}%",
            f"- VerhulstPINN ensemble RMSE reduction: {summary['relative_to_data_driven']['verhulst']['ensemble_rmse_reduction_percent']:.3f}%",
            f"- DeepHPMNN seed-mean RMSE reduction: {summary['relative_to_data_driven']['deephpm']['seed_mean_rmse_reduction_percent']:.3f}%",
            f"- DeepHPMNN ensemble RMSE reduction: {summary['relative_to_data_driven']['deephpm']['ensemble_rmse_reduction_percent']:.3f}%",
            "",
            "## Interpretation guardrails",
            "",
            "- Seed means describe repeated-training performance; ensemble metrics describe averaged predictions and are not interchangeable.",
            "- Lower physics residual indicates closer agreement with a model's physical equation, not automatically better prediction accuracy.",
            "- Results are cycle-weighted because every cycle is one evaluation record.",
            "- No statistical significance claim is made from three seeds.",
            "",
            f"Representative batteries: {', '.join(map(str, summary['representative_batteries']['battery_ids']))}.",
            "",
            "Figures are saved as 300-DPI PNG and vector SVG/PDF files.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_comparison(
    output_dir: Path = Path("artifacts/comparisons/soh"),
    model_specs: dict[str, dict[str, Any]] = MODEL_SPECS,
) -> dict[str, Any]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    data = load_comparison_data(model_specs)
    summary, per_seed, aggregate, ensembles, per_battery, physics_rows = calculate_comparison(data)

    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "alignment_report.json").write_text(
        json.dumps(summary["alignment"], indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "per_seed_metrics.csv", per_seed)
    _write_csv(output_dir / "aggregate_metrics.csv", aggregate)
    _write_csv(output_dir / "ensemble_metrics.csv", ensembles)
    _write_csv(output_dir / "per_battery_metrics.csv", per_battery)
    _write_csv(output_dir / "physics_residual_metrics.csv", physics_rows)
    _save_aligned_predictions(data, output_dir / "aligned_ensemble_predictions.csv")
    _write_markdown(summary, aggregate, ensembles, output_dir / "comparison_summary.md")

    representatives = summary["representative_batteries"]["battery_ids"]
    figures: Iterable[tuple[str, plt.Figure]] = (
        (COMPARISON_FIGURES[0], _plot_metric_means(aggregate)),
        (COMPARISON_FIGURES[1], _plot_parity(data, ensembles)),
        (COMPARISON_FIGURES[2], _plot_trajectories(data, representatives)),
        (COMPARISON_FIGURES[3], _plot_per_battery(per_battery)),
        (COMPARISON_FIGURES[4], _plot_error_distributions(data)),
        (COMPARISON_FIGURES[5], _plot_seed_stability(per_seed)),
        (COMPARISON_FIGURES[6], _plot_physics_residuals(data)),
    )
    for name, figure in figures:
        save_figure(figure, figure_dir, name)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/comparisons/soh")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_comparison(args.output_dir)
    print("Aligned all model predictions:", summary["alignment"])
    for key in summary["ranking_by_seed_mean_rmse"]:
        metrics = summary["models"][key]["seed_aggregate_metrics"]["rmse"]
        print(f"{summary['models'][key]['label']}: RMSE {metrics['mean']:.8f} +/- {metrics['sample_std']:.8f}")
    print(f"Report: {(args.output_dir / 'comparison_summary.md').resolve()}")


if __name__ == "__main__":
    main()
