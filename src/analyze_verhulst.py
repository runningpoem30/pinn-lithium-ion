"""Analyze frozen three-seed VerhulstPINN SoH results without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from src.analyze_final import (
    COLORS,
    AlignedPredictions,
    _load_histories,
    _plot_error_distributions,
    _plot_parity,
    _plot_per_battery,
    _plot_seed_stability,
    _plot_trajectories,
    _representative_batteries,
    _style,
    _write_csv,
    calculate_analysis,
    load_aligned_predictions,
    save_figure,
)


VERHULST_FIGURE_NAMES = (
    "01_actual_vs_predicted_soh",
    "02_representative_battery_trajectories",
    "03_per_battery_errors",
    "04_error_distributions",
    "05_training_curves",
    "06_physics_residuals",
    "07_verhulst_rhs_vs_autograd",
    "08_seed_stability",
)


def _plot_training_curves(input_dir: Path) -> plt.Figure:
    histories = _load_histories(input_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    definitions = (
        ("Data loss", "train_data", "validation_data"),
        ("Physics loss", "train_physics", "validation_physics"),
        ("Total loss", "train_total", "validation_total"),
    )
    for axis, (title, train_name, validation_name) in zip(
        axes, definitions, strict=True
    ):
        for index, (seed, history, best_epoch) in enumerate(histories):
            color = COLORS[index % len(COLORS)]
            axis.plot(
                history["epoch"], history[train_name],
                color=color, alpha=0.45, linestyle="--",
            )
            axis.plot(
                history["epoch"], history[validation_name],
                color=color, label=f"Seed {seed}",
            )
            if title == "Data loss":
                axis.axvline(best_epoch, color=color, linestyle=":", alpha=0.7)
        axis.set_yscale("log")
        axis.set(xlabel="Epoch", ylabel="Mean loss", title=title)
    axes[0].legend()
    fig.suptitle(
        "VerhulstPINN training histories "
        "(solid: validation, dashed: training)"
    )
    fig.tight_layout()
    return fig


def _plot_physics_residuals(aligned: AlignedPredictions) -> plt.Figure:
    residual = aligned.physics_residual.mean(axis=0)
    maximum = 10_000
    index = (
        np.arange(residual.size)
        if residual.size <= maximum
        else np.linspace(0, residual.size - 1, maximum, dtype=int)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(
        aligned.cycle[index], residual[index], s=8, alpha=0.25,
        color=COLORS[0], edgecolors="none",
    )
    axes[0].axhline(0.0, color="#222222", linestyle="--")
    axes[0].set(
        xlabel="Cycle",
        ylabel="Residual (PCL per cycle)",
        title="Residual across battery life",
    )
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].hist(residual, bins=60, color=COLORS[2], alpha=0.85)
    axes[1].axvline(0.0, color="#222222", linestyle="--")
    axes[1].set(
        xlabel="Residual (PCL per cycle)",
        ylabel="Cycles",
        title="Residual distribution",
    )
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle(r"Verhulst residual $dU/dcycle - f_{Verhulst}(U)$")
    fig.tight_layout()
    return fig


def _plot_verhulst_agreement(aligned: AlignedPredictions) -> plt.Figure:
    derivative = aligned.time_derivative.mean(axis=0)
    rhs = aligned.dynamics.mean(axis=0)
    maximum = 10_000
    index = (
        np.arange(derivative.size)
        if derivative.size <= maximum
        else np.linspace(0, derivative.size - 1, maximum, dtype=int)
    )
    lower = min(derivative[index].min(), rhs[index].min())
    upper = max(derivative[index].max(), rhs[index].max())
    fig, axis = plt.subplots(figsize=(8.5, 6))
    axis.scatter(
        derivative[index], rhs[index], s=9, alpha=0.3,
        color=COLORS[2], edgecolors="none",
    )
    axis.plot(
        [lower, upper], [lower, upper], "--", color="#222222",
        label="Exact agreement",
    )
    axis.set(
        xlabel=r"Autograd derivative $dU/dcycle$",
        ylabel="Verhulst right-hand side",
        title="Verhulst dynamics versus surrogate derivative",
    )
    axis.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    axis.legend()
    return fig


def _load_parameter_summary(input_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for directory in sorted(input_dir.glob("seed_*")):
        if not directory.is_dir():
            continue
        seed = int(directory.name.removeprefix("seed_"))
        values = json.loads(
            (directory / "verhulst_parameters.json").read_text(encoding="utf-8")
        )
        rows.append({"seed": seed, **values})
    if not rows:
        raise FileNotFoundError(f"No Verhulst parameter files under {input_dir}")
    numeric_names = ("rate_r_per_cycle", "capacity_K_pcl", "offset_C_pcl")
    aggregate = {}
    for name in numeric_names:
        values = np.asarray([float(row[name]) for row in rows])
        aggregate[name] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    return {
        "state": rows[0]["state"],
        "residual_units": rows[0]["residual_units"],
        "per_seed": rows,
        "aggregate": aggregate,
    }


def _save_aligned_predictions(aligned: AlignedPredictions, path: Path) -> None:
    matrix = np.column_stack(
        (
            aligned.battery_id,
            aligned.cycle,
            aligned.target_soh,
            aligned.predicted_soh.mean(axis=0),
            aligned.predicted_soh.std(axis=0, ddof=1),
            aligned.time_derivative.mean(axis=0),
            aligned.dynamics.mean(axis=0),
            aligned.physics_residual.mean(axis=0),
        )
    )
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header=(
            "battery_id,cycle,target_soh,predicted_soh_mean,"
            "predicted_soh_std,time_derivative_mean,"
            "verhulst_rhs_mean,physics_residual_mean"
        ),
        comments="",
        fmt=("%d",) + ("%.10g",) * 7,
    )


def _write_markdown(
    summary: dict[str, Any], representatives: list[int], path: Path
) -> None:
    metric = summary["ensemble_metrics"]
    residual = summary["physics_residual"]
    parameters = summary["verhulst_parameters"]["aggregate"]
    lines = [
        "# VerhulstPINN SoH analysis",
        "",
        "This is a read-only analysis of three frozen test-prediction files; no model was retrained.",
        "Predictions were aligned by battery ID and cycle before ensembling.",
        "",
        "## Ensemble test metrics",
        "",
        f"- MAE: {metric['mae']:.8f}",
        f"- RMSE: {metric['rmse']:.8f}",
        f"- R2: {metric['r2']:.6f}",
        f"- Bias: {metric['bias']:.8f}",
        f"- 95th-percentile absolute error: {metric['absolute_error_p95']:.8f}",
        f"- Physics-residual RMSE: {residual['rmse']:.10f} PCL/cycle",
        "",
        "## Learned physically constrained parameters",
        "",
        f"- r: {parameters['rate_r_per_cycle']['mean']:.8f} per cycle",
        f"- K: {parameters['capacity_K_pcl']['mean']:.8f} PCL",
        f"- C: {parameters['offset_C_pcl']['mean']:.8f} PCL",
        "",
        "## Representative batteries",
        "",
        f"Lowest, median, and highest per-battery RMSE: {', '.join(map(str, representatives))}.",
        "",
        "Figures are available as 300-DPI PNG plus vector SVG and PDF.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_verhulst_analysis(
    input_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Create Verhulst reports and figures from existing Step 11 files only."""

    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    aligned = load_aligned_predictions(input_dir)
    summary, per_battery = calculate_analysis(aligned)
    representatives = _representative_batteries(per_battery)
    residual = aligned.physics_residual.mean(axis=0)
    summary.update(
        {
            "model_type": "verhulst",
            "analysis_mode": "saved predictions and histories only; no retraining",
            "physics_residual": {
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "units": "PCL per physical cycle",
            },
            "verhulst_parameters": _load_parameter_summary(input_dir),
            "representative_batteries": {
                "selection": "lowest, median, and highest ensemble RMSE",
                "battery_ids": representatives,
            },
        }
    )

    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(summary, representatives, output_dir / "analysis_summary.md")
    _write_csv(output_dir / "per_battery_metrics.csv", per_battery)
    _write_csv(output_dir / "per_seed_metrics.csv", summary["per_seed_metrics"])
    _save_aligned_predictions(
        aligned, output_dir / "aligned_ensemble_predictions.csv"
    )

    figures: Iterable[tuple[str, plt.Figure]] = (
        (VERHULST_FIGURE_NAMES[0], _plot_parity(aligned, summary, "VerhulstPINN")),
        (
            VERHULST_FIGURE_NAMES[1],
            _plot_trajectories(aligned, representatives, "VerhulstPINN"),
        ),
        (
            VERHULST_FIGURE_NAMES[2],
            _plot_per_battery(per_battery, "VerhulstPINN"),
        ),
        (
            VERHULST_FIGURE_NAMES[3],
            _plot_error_distributions(aligned, "VerhulstPINN"),
        ),
        (VERHULST_FIGURE_NAMES[4], _plot_training_curves(input_dir)),
        (VERHULST_FIGURE_NAMES[5], _plot_physics_residuals(aligned)),
        (VERHULST_FIGURE_NAMES[6], _plot_verhulst_agreement(aligned)),
        (
            VERHULST_FIGURE_NAMES[7],
            _plot_seed_stability(summary, "VerhulstPINN"),
        ),
    )
    for name, figure in figures:
        save_figure(figure, figure_dir, name)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("artifacts/verhulst/final")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/verhulst/analysis")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_verhulst_analysis(args.input_dir, args.output_dir)
    metrics = summary["ensemble_metrics"]
    print(
        f"VerhulstPINN ensemble RMSE={metrics['rmse']:.8f}, "
        f"R2={metrics['r2']:.6f}"
    )
    print(f"Analysis: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
