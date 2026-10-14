"""Analyze the frozen three-seed DataDrivenNN SoH baseline without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from src.analyze_final import (
    COLORS,
    AlignedSoHPredictions,
    _load_histories,
    _plot_error_distributions,
    _plot_parity,
    _plot_per_battery,
    _plot_seed_stability,
    _plot_trajectories,
    _representative_batteries,
    _seed_directories,
    _style,
    _write_csv,
    calculate_analysis,
    save_figure,
)
from src.evaluation import align_prediction_tables, load_prediction_csv


BASELINE_FIGURE_NAMES = (
    "01_actual_vs_predicted_soh",
    "02_representative_battery_trajectories",
    "03_per_battery_errors",
    "04_error_distributions",
    "05_training_curves",
    "06_seed_stability",
)


def load_aligned_baseline_predictions(input_dir: Path) -> AlignedSoHPredictions:
    tables = {
        str(seed): load_prediction_csv(directory / "test_predictions.csv")
        for seed, directory in _seed_directories(input_dir)
    }
    aligned = align_prediction_tables(tables)
    seeds = tuple(int(name) for name in aligned)
    reference = aligned[str(seeds[0])]
    return AlignedSoHPredictions(
        seeds=seeds,
        battery_id=reference.columns["battery_id"].astype(np.int64),
        cycle=reference.columns["cycle"].astype(np.float64),
        target_soh=reference.columns["target_soh"].astype(np.float64),
        predicted_soh=np.stack(
            [aligned[str(seed)].columns["predicted_soh"] for seed in seeds]
        ),
    )


def _plot_baseline_training_curves(input_dir: Path) -> plt.Figure:
    histories = _load_histories(input_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for index, (seed, history, best_epoch) in enumerate(histories):
        color = COLORS[index % len(COLORS)]
        axes[0].plot(
            history["epoch"],
            history["train_data"],
            color=color,
            alpha=0.5,
            linestyle="--",
        )
        axes[0].plot(
            history["epoch"],
            history["validation_data"],
            color=color,
            label=f"Seed {seed}",
        )
        axes[0].axvline(best_epoch, color=color, linestyle=":", alpha=0.75)
        axes[1].plot(
            history["epoch"],
            history["gradient_norm"],
            color=color,
            label=f"Seed {seed}",
        )
    axes[0].set_yscale("log")
    axes[0].set(
        xlabel="Epoch",
        ylabel="Normalized PCL MSE",
        title="Data loss (solid: validation, dashed: training)",
    )
    axes[1].set(
        xlabel="Epoch", ylabel="Gradient norm", title="Training gradient norm"
    )
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("DataDrivenNN training histories")
    fig.tight_layout()
    return fig


def _save_aligned_predictions(
    aligned: AlignedSoHPredictions, path: Path
) -> None:
    ensemble = aligned.predicted_soh.mean(axis=0)
    standard_deviation = aligned.predicted_soh.std(axis=0, ddof=1)
    matrix = np.column_stack(
        (
            aligned.battery_id,
            aligned.cycle,
            aligned.target_soh,
            ensemble,
            standard_deviation,
        )
    )
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header=(
            "battery_id,cycle,target_soh,predicted_soh_mean,predicted_soh_std"
        ),
        comments="",
        fmt=("%d",) + ("%.10g",) * 4,
    )


def _write_markdown(
    summary: dict[str, Any], representatives: list[int], path: Path
) -> None:
    metric = summary["ensemble_metrics"]
    uncertainty = summary["prediction_uncertainty"]
    lines = [
        "# DataDrivenNN SoH analysis",
        "",
        "Results use the equal-weight mean of three predictions aligned by battery ID and cycle.",
        "No model was retrained and no test result was used for model selection.",
        "",
        "## Ensemble test metrics",
        "",
        f"- MAE: {metric['mae']:.8f}",
        f"- RMSE: {metric['rmse']:.8f}",
        f"- R2: {metric['r2']:.6f}",
        f"- Bias: {metric['bias']:.8f}",
        f"- 95th-percentile absolute error: {metric['absolute_error_p95']:.8f}",
        f"- Mean prediction standard deviation across seeds: {uncertainty['mean_seed_std']:.8f}",
        "",
        "## Representative batteries",
        "",
        f"Lowest, median, and highest per-battery RMSE: {', '.join(map(str, representatives))}.",
        "",
        "Figures are available as 300-DPI PNG plus SVG and PDF.",
        "Physics-residual figures are not applicable to this data-only model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_baseline_analysis(
    input_dir: Path, output_dir: Path
) -> dict[str, Any]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    aligned = load_aligned_baseline_predictions(input_dir)
    summary, per_battery = calculate_analysis(aligned)
    summary["model_type"] = "data_driven"
    summary["physics_diagnostics"] = "not applicable: data-loss-only model"
    representatives = _representative_batteries(per_battery)
    summary["representative_batteries"] = {
        "selection": "lowest, median, and highest ensemble RMSE",
        "battery_ids": representatives,
    }
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
        (
            BASELINE_FIGURE_NAMES[0],
            _plot_parity(aligned, summary, "DataDrivenNN"),
        ),
        (
            BASELINE_FIGURE_NAMES[1],
            _plot_trajectories(aligned, representatives, "DataDrivenNN"),
        ),
        (
            BASELINE_FIGURE_NAMES[2],
            _plot_per_battery(per_battery, "DataDrivenNN"),
        ),
        (
            BASELINE_FIGURE_NAMES[3],
            _plot_error_distributions(aligned, "DataDrivenNN"),
        ),
        (BASELINE_FIGURE_NAMES[4], _plot_baseline_training_curves(input_dir)),
        (
            BASELINE_FIGURE_NAMES[5],
            _plot_seed_stability(summary, "DataDrivenNN"),
        ),
    )
    for name, figure in figures:
        save_figure(figure, figure_dir, name)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("artifacts/data_driven/final")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/data_driven/analysis")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_baseline_analysis(args.input_dir, args.output_dir)
    metrics = summary["ensemble_metrics"]
    print(
        f"DataDrivenNN ensemble RMSE={metrics['rmse']:.8f}, "
        f"R2={metrics['r2']:.6f}"
    )
    print(f"Analysis: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
