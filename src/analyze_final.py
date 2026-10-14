"""Analyze and visualize frozen Step 6 DeepHPM SoH predictions.

This module never imports the training pipeline or model code. It operates only
on histories, summaries, and test predictions already written by Step 6.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PREDICTION_COLUMNS = (
    "battery_id",
    "cycle",
    "target_soh",
    "predicted_soh",
    "time_derivative",
    "dynamics",
    "physics_residual",
)
FIGURE_NAMES = (
    "01_actual_vs_predicted_soh",
    "02_representative_battery_trajectories",
    "03_per_battery_errors",
    "04_error_distributions",
    "05_training_curves",
    "06_physics_residuals",
    "07_learned_dynamics_vs_autograd",
    "08_seed_stability",
)
FORMATS = ("png", "svg", "pdf")
COLORS = ("#0072B2", "#D55E00", "#009E73")


@dataclass(frozen=True)
class AlignedPredictions:
    seeds: tuple[int, ...]
    battery_id: np.ndarray
    cycle: np.ndarray
    target_soh: np.ndarray
    predicted_soh: np.ndarray
    time_derivative: np.ndarray
    dynamics: np.ndarray
    physics_residual: np.ndarray


def _numeric_csv(path: Path) -> dict[str, np.ndarray]:
    values = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    values = np.atleast_1d(values)
    if values.dtype.names is None:
        raise ValueError(f"CSV has no header: {path}")
    missing = sorted(set(PREDICTION_COLUMNS) - set(values.dtype.names))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    result = {name: np.asarray(values[name]) for name in PREDICTION_COLUMNS}
    if not all(np.all(np.isfinite(column)) for column in result.values()):
        raise FloatingPointError(f"Non-finite values found in {path}")
    return result


def _seed_directories(input_dir: Path) -> list[tuple[int, Path]]:
    directories: list[tuple[int, Path]] = []
    for path in input_dir.glob("seed_*"):
        if path.is_dir():
            try:
                seed = int(path.name.removeprefix("seed_"))
            except ValueError as error:
                raise ValueError(f"Invalid seed directory name: {path.name}") from error
            directories.append((seed, path))
    directories.sort()
    if not directories:
        raise FileNotFoundError(f"No seed directories found under {input_dir}")
    return directories


def load_aligned_predictions(input_dir: Path) -> AlignedPredictions:
    """Load every seed and align rows by the stable battery/cycle key."""

    seed_data: list[tuple[int, dict[str, np.ndarray], list[tuple[int, float]]]] = []
    reference_keys: list[tuple[int, float]] | None = None
    for seed, directory in _seed_directories(input_dir):
        data = _numeric_csv(directory / "test_predictions.csv")
        keys = [
            (int(battery), float(cycle))
            for battery, cycle in zip(data["battery_id"], data["cycle"], strict=True)
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate battery/cycle keys for seed {seed}")
        if reference_keys is None:
            reference_keys = sorted(keys)
        elif set(keys) != set(reference_keys):
            missing = len(set(reference_keys) - set(keys))
            extra = len(set(keys) - set(reference_keys))
            raise ValueError(
                f"Seed {seed} prediction keys do not align: "
                f"missing={missing}, extra={extra}"
            )
        seed_data.append((seed, data, keys))

    assert reference_keys is not None
    stacked: dict[str, list[np.ndarray]] = {
        name: [] for name in PREDICTION_COLUMNS if name not in {"battery_id", "cycle"}
    }
    for seed, data, keys in seed_data:
        position = {key: index for index, key in enumerate(keys)}
        order = np.asarray([position[key] for key in reference_keys])
        for name in stacked:
            stacked[name].append(data[name][order])

    target_stack = np.stack(stacked.pop("target_soh"))
    if not np.allclose(target_stack, target_stack[0], rtol=0.0, atol=1e-8):
        raise ValueError("Target SoH differs between aligned seed files")
    return AlignedPredictions(
        seeds=tuple(seed for seed, _, _ in seed_data),
        battery_id=np.asarray([key[0] for key in reference_keys], dtype=np.int64),
        cycle=np.asarray([key[1] for key in reference_keys], dtype=np.float64),
        target_soh=target_stack[0],
        predicted_soh=np.stack(stacked["predicted_soh"]),
        time_derivative=np.stack(stacked["time_derivative"]),
        dynamics=np.stack(stacked["dynamics"]),
        physics_residual=np.stack(stacked["physics_residual"]),
    )


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Target and prediction must be non-empty arrays of equal shape")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Metrics require finite values")
    error = prediction - target
    denominator = np.square(target - target.mean()).sum()
    if denominator <= 0:
        raise ValueError("R2 is undefined for a constant target")
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / denominator),
        "bias": float(error.mean()),
        "signed_error_p05": float(np.percentile(error, 5)),
        "signed_error_p50": float(np.percentile(error, 50)),
        "signed_error_p95": float(np.percentile(error, 95)),
        "absolute_error_p50": float(np.percentile(absolute, 50)),
        "absolute_error_p90": float(np.percentile(absolute, 90)),
        "absolute_error_p95": float(np.percentile(absolute, 95)),
        "absolute_error_p99": float(np.percentile(absolute, 99)),
    }


def calculate_analysis(aligned: AlignedPredictions) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ensemble = aligned.predicted_soh.mean(axis=0)
    ensemble_metrics = regression_metrics(aligned.target_soh, ensemble)
    seed_metrics = []
    for index, seed in enumerate(aligned.seeds):
        seed_metrics.append(
            {"seed": seed, **regression_metrics(aligned.target_soh, aligned.predicted_soh[index])}
        )

    metric_names = ("mae", "rmse", "r2", "bias")
    seed_summary = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in seed_metrics])
        seed_summary[name] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }

    per_battery = []
    for battery_id in np.unique(aligned.battery_id):
        mask = aligned.battery_id == battery_id
        per_battery.append(
            {
                "battery_id": int(battery_id),
                "records": int(mask.sum()),
                **regression_metrics(aligned.target_soh[mask], ensemble[mask]),
            }
        )

    prediction_std = aligned.predicted_soh.std(axis=0, ddof=1)
    summary: dict[str, Any] = {
        "method": "equal-weight mean of aligned seed predictions",
        "alignment_key": ["battery_id", "cycle"],
        "seeds": list(aligned.seeds),
        "records": int(aligned.target_soh.size),
        "batteries": int(np.unique(aligned.battery_id).size),
        "ensemble_metrics": ensemble_metrics,
        "per_seed_metrics": seed_metrics,
        "across_seed_metrics": seed_summary,
        "prediction_uncertainty": {
            "mean_seed_std": float(prediction_std.mean()),
            "p95_seed_std": float(np.percentile(prediction_std, 95)),
            "max_seed_std": float(prediction_std.max()),
        },
    }
    return summary, per_battery


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        options = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output_dir / f"{name}.{extension}", **options)
    plt.close(fig)


def _sample_indices(size: int, maximum: int = 10000) -> np.ndarray:
    if size <= maximum:
        return np.arange(size)
    return np.linspace(0, size - 1, maximum, dtype=int)


def _representative_batteries(per_battery: list[dict[str, Any]]) -> list[int]:
    ordered = sorted(per_battery, key=lambda row: row["rmse"])
    indices = (0, round((len(ordered) - 1) / 2), len(ordered) - 1)
    return [int(ordered[index]["battery_id"]) for index in indices]


def _plot_parity(aligned: AlignedPredictions, summary: dict[str, Any]) -> plt.Figure:
    predicted = aligned.predicted_soh.mean(axis=0)
    index = _sample_indices(predicted.size)
    lower = min(aligned.target_soh.min(), predicted.min())
    upper = max(aligned.target_soh.max(), predicted.max())
    fig, axis = plt.subplots(figsize=(9.2, 6.2))
    axis.scatter(
        aligned.target_soh[index], predicted[index], s=10, alpha=0.3,
        color=COLORS[0], edgecolors="none", label="Test cycles",
    )
    axis.plot([lower, upper], [lower, upper], "--", color="#222222", label="Ideal")
    metrics = summary["ensemble_metrics"]
    axis.text(
        0.03, 0.97,
        f"RMSE = {metrics['rmse']:.4f}\nMAE = {metrics['mae']:.4f}\n$R^2$ = {metrics['r2']:.3f}",
        transform=axis.transAxes, va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#BBBBBB"},
    )
    axis.set(xlabel="Measured SoH", ylabel="Predicted SoH", title="DeepHPM SoH: measured vs predicted")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="lower right")
    return fig


def _plot_trajectories(
    aligned: AlignedPredictions, representatives: list[int]
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(representatives), figsize=(15, 4.8), sharey=True)
    for axis, battery_id, difficulty in zip(
        np.atleast_1d(axes), representatives, ("Lowest error", "Median error", "Highest error"), strict=True
    ):
        mask = aligned.battery_id == battery_id
        cycles = aligned.cycle[mask]
        predictions = aligned.predicted_soh[:, mask]
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0, ddof=1)
        axis.plot(cycles, aligned.target_soh[mask], color="#222222", linewidth=1.8, label="Measured")
        axis.plot(cycles, mean, color=COLORS[0], linewidth=1.8, label="3-seed mean")
        axis.fill_between(cycles, mean - std, mean + std, color=COLORS[0], alpha=0.2, label="±1 SD")
        axis.set_title(f"Battery {battery_id}\n{difficulty}")
        axis.set_xlabel("Cycle")
    axes[0].set_ylabel("SoH")
    axes[-1].legend(loc="best")
    fig.suptitle("Representative held-out battery degradation trajectories")
    fig.tight_layout()
    return fig


def _plot_per_battery(per_battery: list[dict[str, Any]]) -> plt.Figure:
    rows = sorted(per_battery, key=lambda row: row["battery_id"])
    batteries = [row["battery_id"] for row in rows]
    fig, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(batteries, [row["rmse"] for row in rows], "o-", color=COLORS[1], label="RMSE")
    axis.plot(batteries, [row["mae"] for row in rows], "s-", color=COLORS[0], label="MAE")
    axis.axhline(np.mean([row["rmse"] for row in rows]), color=COLORS[1], linestyle="--", alpha=0.7)
    axis.set(xlabel="Test battery ID", ylabel="SoH error", title="Per-battery DeepHPM error")
    axis.set_xticks(batteries[::2])
    axis.legend()
    return fig


def _plot_error_distributions(aligned: AlignedPredictions) -> plt.Figure:
    error = aligned.predicted_soh.mean(axis=0) - aligned.target_soh
    absolute = np.abs(error)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(error, bins=60, color=COLORS[0], alpha=0.85)
    axes[0].axvline(error.mean(), color=COLORS[1], linestyle="--", label=f"Bias = {error.mean():.4f}")
    axes[0].set(xlabel="Signed SoH error", ylabel="Cycles", title="Signed prediction error")
    axes[0].legend()
    ordered = np.sort(absolute)
    axes[1].plot(ordered, np.arange(1, ordered.size + 1) / ordered.size, color=COLORS[2], linewidth=2)
    axes[1].axvline(np.percentile(absolute, 95), color=COLORS[1], linestyle="--", label="95th percentile")
    axes[1].set(xlabel="Absolute SoH error", ylabel="Cumulative fraction", title="Absolute-error distribution")
    axes[1].legend()
    fig.suptitle("DeepHPM test error distribution")
    fig.tight_layout()
    return fig


def _load_histories(input_dir: Path) -> list[tuple[int, dict[str, np.ndarray], int]]:
    histories = []
    for seed, directory in _seed_directories(input_dir):
        values = np.genfromtxt(directory / "history.csv", delimiter=",", names=True)
        values = np.atleast_1d(values)
        history = {name: np.asarray(values[name]) for name in values.dtype.names or ()}
        summary = json.loads((directory / "training_summary.json").read_text(encoding="utf-8"))
        histories.append((seed, history, int(summary["best_epoch"])))
    return histories


def _plot_training_curves(input_dir: Path) -> plt.Figure:
    histories = _load_histories(input_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    definitions = (
        ("Data loss", "train_data", "validation_data"),
        ("Physics loss", "train_physics", "validation_physics"),
        ("Total loss", "train_total", "validation_total"),
    )
    for axis, (title, train_name, validation_name) in zip(axes, definitions, strict=True):
        for index, (seed, history, best_epoch) in enumerate(histories):
            color = COLORS[index % len(COLORS)]
            axis.plot(history["epoch"], history[train_name], color=color, alpha=0.45, linestyle="--")
            axis.plot(history["epoch"], history[validation_name], color=color, label=f"Seed {seed}")
            if title == "Data loss":
                axis.axvline(best_epoch, color=color, linestyle=":", alpha=0.7)
        axis.set_yscale("log")
        axis.set(xlabel="Epoch", ylabel="Mean loss", title=title)
    axes[0].legend()
    fig.suptitle("DeepHPM training and validation histories (solid: validation, dashed: training)")
    fig.tight_layout()
    return fig


def _plot_physics_residuals(aligned: AlignedPredictions) -> plt.Figure:
    residual = aligned.physics_residual.mean(axis=0)
    index = _sample_indices(residual.size)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(aligned.cycle[index], residual[index], s=8, alpha=0.25, color=COLORS[0], edgecolors="none")
    axes[0].axhline(0.0, color="#222222", linestyle="--")
    axes[0].set(xlabel="Cycle", ylabel="Residual (PCL per cycle)", title="Residual across battery life")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].hist(residual, bins=60, color=COLORS[2], alpha=0.85)
    axes[1].axvline(0.0, color="#222222", linestyle="--")
    axes[1].set(xlabel="Residual (PCL per cycle)", ylabel="Cycles", title="Residual distribution")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle(r"Physics residual $dU/dcycle - G(cycle, U)$")
    fig.tight_layout()
    return fig


def _plot_dynamics(aligned: AlignedPredictions) -> plt.Figure:
    derivative = aligned.time_derivative.mean(axis=0)
    dynamics = aligned.dynamics.mean(axis=0)
    index = _sample_indices(derivative.size)
    lower = min(derivative[index].min(), dynamics[index].min())
    upper = max(derivative[index].max(), dynamics[index].max())
    fig, axis = plt.subplots(figsize=(8.5, 6))
    axis.scatter(derivative[index], dynamics[index], s=9, alpha=0.3, color=COLORS[2], edgecolors="none")
    axis.plot([lower, upper], [lower, upper], "--", color="#222222", label="Exact agreement")
    axis.set(
        xlabel=r"Autograd derivative $dU/dcycle$",
        ylabel=r"Learned dynamics $G(cycle, U)$",
        title="Learned dynamics versus surrogate derivative",
    )
    axis.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    axis.legend()
    return fig


def _plot_seed_stability(summary: dict[str, Any]) -> plt.Figure:
    rows = summary["per_seed_metrics"]
    labels = [str(row["seed"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    width = 0.35
    axes[0].bar(x - width / 2, [row["mae"] for row in rows], width, color=COLORS[0], label="MAE")
    axes[0].bar(x + width / 2, [row["rmse"] for row in rows], width, color=COLORS[1], label="RMSE")
    axes[0].set(xticks=x, xticklabels=labels, xlabel="Seed", ylabel="SoH error", title="Error by seed")
    axes[0].legend()
    axes[1].bar(x, [row["r2"] for row in rows], width=0.55, color=COLORS[2])
    axes[1].set(xticks=x, xticklabels=labels, xlabel="Seed", ylabel=r"$R^2$", title=r"$R^2$ by seed")
    axes[1].set_ylim(0, 1)
    fig.suptitle("DeepHPM stability across fixed random seeds")
    fig.tight_layout()
    return fig


def _save_aligned_predictions(aligned: AlignedPredictions, path: Path) -> None:
    ensemble = aligned.predicted_soh.mean(axis=0)
    std = aligned.predicted_soh.std(axis=0, ddof=1)
    derivative = aligned.time_derivative.mean(axis=0)
    dynamics = aligned.dynamics.mean(axis=0)
    residual = aligned.physics_residual.mean(axis=0)
    header = "battery_id,cycle,target_soh,predicted_soh_mean,predicted_soh_std,time_derivative_mean,dynamics_mean,physics_residual_mean"
    matrix = np.column_stack((aligned.battery_id, aligned.cycle, aligned.target_soh, ensemble, std, derivative, dynamics, residual))
    np.savetxt(path, matrix, delimiter=",", header=header, comments="", fmt=("%d",) + ("%.10g",) * 7)


def _write_markdown(summary: dict[str, Any], representatives: list[int], path: Path) -> None:
    metric = summary["ensemble_metrics"]
    uncertainty = summary["prediction_uncertainty"]
    lines = [
        "# DeepHPM SoH analysis",
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
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_analysis(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Generate all Step 7 tables, reports, and figures from Step 6 files."""

    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    aligned = load_aligned_predictions(input_dir)
    summary, per_battery = calculate_analysis(aligned)
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
    _save_aligned_predictions(aligned, output_dir / "aligned_ensemble_predictions.csv")

    figures: Iterable[tuple[str, plt.Figure]] = (
        (FIGURE_NAMES[0], _plot_parity(aligned, summary)),
        (FIGURE_NAMES[1], _plot_trajectories(aligned, representatives)),
        (FIGURE_NAMES[2], _plot_per_battery(per_battery)),
        (FIGURE_NAMES[3], _plot_error_distributions(aligned)),
        (FIGURE_NAMES[4], _plot_training_curves(input_dir)),
        (FIGURE_NAMES[5], _plot_physics_residuals(aligned)),
        (FIGURE_NAMES[6], _plot_dynamics(aligned)),
        (FIGURE_NAMES[7], _plot_seed_stability(summary)),
    )
    for name, figure in figures:
        save_figure(figure, figure_dir, name)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/deephpm/final"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/deephpm/analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_analysis(args.input_dir, args.output_dir)
    metrics = summary["ensemble_metrics"]
    print(f"DeepHPM ensemble RMSE={metrics['rmse']:.8f}, R2={metrics['r2']:.6f}")
    print(f"Analysis: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
