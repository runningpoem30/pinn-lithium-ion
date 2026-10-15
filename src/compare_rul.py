"""Compare frozen DataDrivenNN, VerhulstPINN, and DeepHPMNN RUL results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analyze_final import _sample_indices, _style
from src.evaluation import regression_metrics
from src.protocol import COMMON_METRICS, FIXED_SEEDS


MODEL_SPECS = {
    "data_driven": {
        "label": "DataDrivenNN", "path": Path("artifacts/data_driven/rul/final"),
        "color": "#0072B2", "residual_column": None, "residual_units": None,
    },
    "verhulst": {
        "label": "VerhulstPINN", "path": Path("artifacts/verhulst/rul/final"),
        "color": "#D55E00", "residual_column": "physics_residual",
        "residual_units": "PCL per cycle",
    },
    "deephpm": {
        "label": "DeepHPMNN", "path": Path("artifacts/deephpm/rul/final"),
        "color": "#009E73", "residual_column": "physics_residual_rul_per_cycle",
        "residual_units": "cycles per cycle",
    },
}

FIGURE_NAMES = (
    "01_actual_vs_predicted_rul",
    "02_representative_rul_trajectories",
    "03_absolute_error_vs_cycle_and_life_fraction",
    "04_horizon_binned_error",
    "05_per_battery_rmse",
    "06_metric_means_with_seed_error_bars",
    "07_error_distributions",
    "08_physics_residuals",
)
REGIONS = (
    ("early", 0.0, 1.0 / 3.0),
    ("middle", 1.0 / 3.0, 2.0 / 3.0),
    ("late", 2.0 / 3.0, 1.0 + 1e-12),
)
HORIZONS = (
    ("0-99", 0.0, 100.0),
    ("100-249", 100.0, 250.0),
    ("250-499", 250.0, 500.0),
    ("500-999", 500.0, 1000.0),
    ("1000+", 1000.0, np.inf),
)


@dataclass(frozen=True)
class RULModelData:
    key: str
    label: str
    color: str
    predictions: np.ndarray
    residuals: np.ndarray | None
    residual_units: str | None

    @property
    def ensemble(self) -> np.ndarray:
        return self.predictions.mean(axis=0)


@dataclass(frozen=True)
class RULComparisonData:
    models: dict[str, RULModelData]
    battery_id: np.ndarray
    cycle: np.ndarray
    eol_cycle: np.ndarray
    target_rul: np.ndarray

    @property
    def life_fraction(self) -> np.ndarray:
        return self.cycle / self.eol_cycle


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    values = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    values = np.atleast_1d(values)
    if not values.dtype.names:
        raise ValueError(f"Prediction file has no named columns: {path}")
    required = {"battery_id", "cycle", "eol_cycle", "target_rul_cycles", "predicted_rul_cycles"}
    missing = required - set(values.dtype.names)
    if missing:
        raise ValueError(f"Missing RUL columns in {path}: {sorted(missing)}")
    columns = {name: np.asarray(values[name]) for name in values.dtype.names}
    if not all(np.all(np.isfinite(column)) for column in columns.values()):
        raise FloatingPointError(f"Non-finite prediction values in {path}")
    order = np.lexsort((columns["cycle"], columns["battery_id"]))
    return {name: column[order] for name, column in columns.items()}


def load_comparison_data(model_specs: dict[str, dict[str, Any]] = MODEL_SPECS) -> RULComparisonData:
    tables: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for key, spec in model_specs.items():
        for seed in FIXED_SEEDS:
            tables[(key, seed)] = _load_csv(Path(spec["path"]) / f"seed_{seed}" / "test_predictions.csv")
    reference = tables[("data_driven", FIXED_SEEDS[0])]
    key_matrix = np.column_stack((reference["battery_id"], reference["cycle"]))
    if np.unique(key_matrix, axis=0).shape[0] != key_matrix.shape[0]:
        raise ValueError("Reference predictions contain duplicate battery-cycle keys")
    for name, table in tables.items():
        if not np.array_equal(table["battery_id"], reference["battery_id"]) or not np.array_equal(table["cycle"], reference["cycle"]):
            raise ValueError(f"Prediction keys do not align for {name}")
        for target in ("eol_cycle", "target_rul_cycles"):
            if not np.array_equal(table[target], reference[target]):
                raise ValueError(f"Prediction targets do not align for {name}: {target}")
    models: dict[str, RULModelData] = {}
    for key, spec in model_specs.items():
        residual_column = spec.get("residual_column")
        models[key] = RULModelData(
            key=key, label=str(spec["label"]), color=str(spec["color"]),
            predictions=np.stack([tables[(key, seed)]["predicted_rul_cycles"] for seed in FIXED_SEEDS]),
            residuals=(np.stack([tables[(key, seed)][str(residual_column)] for seed in FIXED_SEEDS]) if residual_column else None),
            residual_units=spec.get("residual_units"),
        )
    return RULComparisonData(
        models=models,
        battery_id=reference["battery_id"].astype(np.int64),
        cycle=reference["cycle"], eol_cycle=reference["eol_cycle"],
        target_rul=reference["target_rul_cycles"],
    )


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return regression_metrics(target.astype(float), prediction.astype(float))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def calculate_comparison(data: RULComparisonData) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    ensembles: list[dict[str, Any]] = []
    per_battery: list[dict[str, Any]] = []
    life_regions: list[dict[str, Any]] = []
    horizons: list[dict[str, Any]] = []
    physics: list[dict[str, Any]] = []
    for model in data.models.values():
        seed_rows = []
        for index, seed in enumerate(FIXED_SEEDS):
            row = {"model": model.key, "model_label": model.label, "seed": seed,
                   "records": int(data.target_rul.size), **_metrics(data.target_rul, model.predictions[index])}
            per_seed.append(row); seed_rows.append(row)
            if model.residuals is not None:
                residual = model.residuals[index]
                physics.append({"model": model.key, "model_label": model.label, "level": "seed", "seed": seed,
                                "units": model.residual_units, "records": residual.size,
                                "residual_mae": float(np.mean(np.abs(residual))),
                                "residual_rmse": float(np.sqrt(np.mean(np.square(residual))))})
        aggregate_row: dict[str, Any] = {"model": model.key, "model_label": model.label, "seeds": 3}
        for metric in COMMON_METRICS:
            values = np.asarray([row[metric] for row in seed_rows])
            aggregate_row[f"{metric}_mean"] = float(values.mean())
            aggregate_row[f"{metric}_sample_std"] = float(values.std(ddof=1))
        aggregate.append(aggregate_row)
        ensembles.append({"model": model.key, "model_label": model.label,
                          "records": int(data.target_rul.size), **_metrics(data.target_rul, model.ensemble)})
        for battery in np.unique(data.battery_id):
            mask = data.battery_id == battery
            per_battery.append({"model": model.key, "model_label": model.label,
                                "battery_id": int(battery), "records": int(mask.sum()),
                                **_metrics(data.target_rul[mask], model.ensemble[mask])})
        for level, seed, prediction in [
            *[("seed", seed, model.predictions[index]) for index, seed in enumerate(FIXED_SEEDS)],
            ("ensemble", "", model.ensemble),
        ]:
            for region, lower, upper in REGIONS:
                mask = (data.life_fraction >= lower) & (data.life_fraction < upper)
                life_regions.append({"model": model.key, "model_label": model.label, "level": level,
                                     "seed": seed, "region": region, "life_fraction_lower": lower,
                                     "life_fraction_upper": min(upper, 1.0), "records": int(mask.sum()),
                                     **_metrics(data.target_rul[mask], prediction[mask])})
            for horizon, lower, upper in HORIZONS:
                mask = (data.target_rul >= lower) & (data.target_rul < upper)
                horizons.append({"model": model.key, "model_label": model.label, "level": level,
                                 "seed": seed, "horizon_bin": horizon, "rul_lower_cycles": lower,
                                 "rul_upper_cycles": "" if np.isinf(upper) else upper,
                                 "records": int(mask.sum()), **_metrics(data.target_rul[mask], prediction[mask])})
        if model.residuals is not None:
            residual = model.residuals.mean(axis=0)
            physics.append({"model": model.key, "model_label": model.label, "level": "ensemble_mean_residual", "seed": "",
                            "units": model.residual_units, "records": residual.size,
                            "residual_mae": float(np.mean(np.abs(residual))),
                            "residual_rmse": float(np.sqrt(np.mean(np.square(residual))))})
    battery_scores = []
    for battery in np.unique(data.battery_id):
        rows = [row for row in per_battery if row["battery_id"] == battery]
        battery_scores.append((int(battery), float(np.mean([row["rmse"] for row in rows]))))
    ordered = sorted(battery_scores, key=lambda item: item[1])
    representatives = [ordered[index][0] for index in (0, round((len(ordered) - 1) / 2), len(ordered) - 1)]
    cohort = np.column_stack((data.battery_id, data.cycle, data.eol_cycle, data.target_rul)).astype("<f8")
    summary = {
        "comparison_method": "saved final predictions only; no model loading or retraining",
        "alignment": {"verified": True, "key": ["battery_id", "cycle"], "targets_identical": True,
                      "records": int(data.target_rul.size), "batteries": int(np.unique(data.battery_id).size),
                      "seeds": list(FIXED_SEEDS), "cohort_sha256": hashlib.sha256(cohort.tobytes()).hexdigest()},
        "life_fraction_definition": "cycle / audited eol_cycle",
        "life_regions": {"early": "[0, 1/3)", "middle": "[1/3, 2/3)", "late": "[2/3, 1]"},
        "ensemble_definition": "equal-weight arithmetic mean of three aligned seed predictions",
        "representative_batteries": representatives,
        "ranking_by_seed_mean_rmse": [row["model"] for row in sorted(aggregate, key=lambda row: row["rmse_mean"])],
        "ranking_by_ensemble_rmse": [row["model"] for row in sorted(ensembles, key=lambda row: row["rmse"])],
        "physics_residual_warning": "Verhulst residual is PCL/cycle; DeepHPM RUL residual is cycles/cycle. Magnitudes are not directly comparable.",
    }
    summary["models"] = {}
    for model in data.models.values():
        summary["models"][model.key] = {
            "label": model.label,
            "seed_aggregate_metrics": next(row for row in aggregate if row["model"] == model.key),
            "ensemble_metrics": next(row for row in ensembles if row["model"] == model.key),
            "ensemble_life_region_metrics": [
                row for row in life_regions if row["model"] == model.key and row["level"] == "ensemble"
            ],
            "ensemble_horizon_metrics": [
                row for row in horizons if row["model"] == model.key and row["level"] == "ensemble"
            ],
            "physics_residual_metrics": [row for row in physics if row["model"] == model.key],
        }
    return {"summary": summary, "per_seed": per_seed, "aggregate": aggregate, "ensembles": ensembles,
            "per_battery": per_battery, "life_regions": life_regions, "horizons": horizons, "physics": physics}


def _save_png(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_parity(data: RULComparisonData, ensembles: list[dict[str, Any]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    sample = _sample_indices(data.target_rul.size, maximum=8000)
    metrics = {row["model"]: row for row in ensembles}; upper = float(data.target_rul.max())
    for axis, model in zip(axes, data.models.values(), strict=True):
        axis.scatter(data.target_rul[sample], model.ensemble[sample], s=8, alpha=0.22,
                     color=model.color, edgecolors="none")
        axis.plot([0, upper], [0, upper], "--", color="#222222")
        row = metrics[model.key]
        axis.text(.04, .96, f"RMSE {row['rmse']:.1f}\nMAE {row['mae']:.1f}\n$R^2$ {row['r2']:.3f}",
                  transform=axis.transAxes, va="top", bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#BBBBBB"})
        axis.set(title=model.label, xlabel="Actual RUL (cycles)")
    axes[0].set_ylabel("Ensemble predicted RUL (cycles)")
    fig.suptitle("Actual versus predicted RUL on aligned held-out cycles"); fig.tight_layout(); return fig


def _plot_trajectories(data: RULComparisonData, batteries: list[int]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    descriptions = ("Lowest shared error", "Median shared error", "Highest shared error")
    styles = {"data_driven": "--", "verhulst": "-.", "deephpm": ":"}
    for axis, battery, description in zip(axes, batteries, descriptions, strict=True):
        mask = data.battery_id == battery
        axis.plot(data.cycle[mask], data.target_rul[mask], color="#222222", lw=2.2, label="Actual")
        for model in data.models.values():
            values = model.predictions[:, mask]; mean = values.mean(axis=0); std = values.std(axis=0, ddof=1)
            axis.plot(data.cycle[mask], mean, color=model.color, ls=styles[model.key], lw=2, label=model.label)
            axis.fill_between(data.cycle[mask], mean - std, mean + std, color=model.color, alpha=.10)
        axis.set(title=f"Battery {battery}\n{description}", xlabel="Cycle")
    axes[0].set_ylabel("RUL (cycles)"); axes[-1].legend(fontsize=9)
    fig.suptitle("Representative RUL trajectories (bands: ±1 seed SD)"); fig.tight_layout(); return fig


def _binned_curve(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = (bins[:-1] + bins[1:]) / 2; medians = []; p90 = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        values = y[(x >= lower) & (x < upper)]
        medians.append(float(np.median(values)) if values.size else np.nan)
        p90.append(float(np.percentile(values, 90)) if values.size else np.nan)
    return centers, np.asarray(medians), np.asarray(p90)


def _plot_absolute_error(data: RULComparisonData) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    cycle_bins = np.linspace(data.cycle.min(), data.cycle.max() + 1e-9, 25)
    fraction_bins = np.linspace(0, 1.000001, 21)
    for model in data.models.values():
        error = np.abs(model.ensemble - data.target_rul)
        for axis, x, bins in ((axes[0], data.cycle, cycle_bins), (axes[1], data.life_fraction, fraction_bins)):
            center, median, p90 = _binned_curve(x, error, bins)
            axis.plot(center, median, color=model.color, lw=2, label=f"{model.label} median")
            axis.plot(center, p90, color=model.color, lw=1.3, ls="--", alpha=.8)
    axes[0].set(xlabel="Cycle", ylabel="Absolute RUL error (cycles)", title="Error versus absolute cycle")
    axes[1].set(xlabel="Elapsed life fraction", ylabel="Absolute RUL error (cycles)", title="Error versus battery life fraction")
    axes[0].legend(fontsize=8); fig.suptitle("Binned absolute error (solid: median, dashed: 90th percentile)"); fig.tight_layout(); return fig


def _plot_horizons(horizons: list[dict[str, Any]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2)); x = np.arange(len(HORIZONS)); width = .24
    for model_index, (key, spec) in enumerate(MODEL_SPECS.items()):
        rows = [row for row in horizons if row["model"] == key and row["level"] == "ensemble"]
        rows.sort(key=lambda row: [h[0] for h in HORIZONS].index(row["horizon_bin"]))
        for axis, metric in zip(axes, ("mae", "rmse"), strict=True):
            axis.bar(x + (model_index - 1) * width, [row[metric] for row in rows], width,
                     color=spec["color"], label=spec["label"])
    for axis, title in zip(axes, ("MAE", "RMSE"), strict=True):
        axis.set(xticks=x, xticklabels=[h[0] for h in HORIZONS], xlabel="Actual RUL horizon (cycles)",
                 ylabel=f"{title} (cycles)", title=title)
    axes[0].legend(); fig.suptitle("Ensemble error by remaining-life horizon"); fig.tight_layout(); return fig


def _plot_per_battery(rows: list[dict[str, Any]]) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(13, 5.8)); offsets = (-.2, 0, .2); markers = ("o", "s", "^")
    for (key, spec), offset, marker in zip(MODEL_SPECS.items(), offsets, markers, strict=True):
        values = sorted((row for row in rows if row["model"] == key), key=lambda row: row["battery_id"])
        axis.plot([row["battery_id"] + offset for row in values], [row["rmse"] for row in values],
                  marker=marker, ms=3.8, lw=1.2, color=spec["color"], label=spec["label"])
    axis.set(xlabel="Held-out battery ID", ylabel="Ensemble RUL RMSE (cycles)", title="Per-battery RUL error")
    axis.legend(); fig.tight_layout(); return fig


def _plot_metric_means(rows: list[dict[str, Any]]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5)); x = np.arange(3)
    colors = [MODEL_SPECS[row["model"]]["color"] for row in rows]
    labels = [row["model_label"] for row in rows]
    for axis, (metric, title) in zip(axes.flat, (("mae", "MAE (cycles)"), ("rmse", "RMSE (cycles)"), ("r2", "$R^2$"), ("bias", "Bias (cycles)")), strict=True):
        axis.bar(x, [row[f"{metric}_mean"] for row in rows],
                 yerr=[row[f"{metric}_sample_std"] for row in rows], capsize=5, color=colors)
        axis.set(xticks=x, xticklabels=labels, title=title, ylabel=title); axis.tick_params(axis="x", rotation=10)
        if metric == "bias": axis.axhline(0, color="#222222", ls="--", lw=1)
    fig.suptitle("RUL metrics: mean ± sample SD across three fixed seeds"); fig.tight_layout(); return fig


def _plot_distributions(data: RULComparisonData) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model in data.models.values():
        error = model.ensemble - data.target_rul
        axes[0].hist(error, bins=80, density=True, histtype="step", lw=1.8, color=model.color, label=model.label)
        absolute = np.sort(np.abs(error)); axes[1].plot(absolute, np.arange(1, absolute.size + 1) / absolute.size,
                                                       color=model.color, lw=2, label=model.label)
    axes[0].axvline(0, color="#222222", ls="--"); axes[0].set(xlabel="Signed RUL error (cycles)", ylabel="Density", title="Signed-error distribution")
    axes[1].set(xlabel="Absolute RUL error (cycles)", ylabel="Cumulative fraction", title="Absolute-error ECDF")
    axes[0].legend(); axes[1].legend(); fig.suptitle("Three-seed ensemble error distributions"); fig.tight_layout(); return fig


def _plot_residuals(data: RULComparisonData) -> plt.Figure:
    physics = [model for model in data.models.values() if model.residuals is not None]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    sample = _sample_indices(data.target_rul.size, maximum=6000)
    for row, model in enumerate(physics):
        assert model.residuals is not None
        residual = model.residuals.mean(axis=0)
        axes[row, 0].scatter(data.life_fraction[sample], residual[sample], s=7, alpha=.2,
                             color=model.color, edgecolors="none")
        axes[row, 0].axhline(0, color="#222222", ls="--")
        axes[row, 0].set(xlabel="Elapsed life fraction", ylabel=f"Residual ({model.residual_units})",
                         title=f"{model.label}: residual across life")
        axes[row, 1].hist(residual, bins=80, density=True, color=model.color, alpha=.75)
        axes[row, 1].axvline(0, color="#222222", ls="--")
        axes[row, 1].set(xlabel=f"Residual ({model.residual_units})", ylabel="Density",
                         title=f"{model.label}: residual distribution")
    fig.suptitle("Physics residuals shown separately because their units differ"); fig.tight_layout(); return fig


def _save_aligned(data: RULComparisonData, path: Path) -> None:
    names = ["battery_id", "cycle", "eol_cycle", "life_fraction", "target_rul_cycles"]
    arrays: list[np.ndarray] = [data.battery_id, data.cycle, data.eol_cycle, data.life_fraction, data.target_rul]
    for model in data.models.values():
        names += [f"{model.key}_predicted_rul_mean", f"{model.key}_predicted_rul_std"]
        arrays += [model.ensemble, model.predictions.std(axis=0, ddof=1)]
        if model.residuals is not None:
            names += [f"{model.key}_physics_residual_mean", f"{model.key}_physics_residual_std"]
            arrays += [model.residuals.mean(axis=0), model.residuals.std(axis=0, ddof=1)]
    np.savetxt(path, np.column_stack(arrays), delimiter=",", header=",".join(names), comments="",
               fmt=("%d",) + ("%.17g",) * (len(names) - 1))


def _notebook_payload(relative_root: str) -> dict[str, Any]:
    def markdown(text: str) -> dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}
    def code(text: str) -> dict[str, Any]:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.splitlines()]}
    return {
        "cells": [
            markdown("# Step 17 — Final RUL model comparison\n\nThis notebook presents saved comparison artifacts only. It contains no training pipeline and does not select models using test data."),
            code(f"from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path({relative_root!r})"),
            markdown("## Alignment and aggregate metrics"),
            code("summary = json.loads((ROOT / 'comparison_summary.json').read_text())\nprint(summary['alignment'])\ndisplay(pd.read_csv(ROOT / 'aggregate_metrics.csv'))\ndisplay(pd.read_csv(ROOT / 'ensemble_metrics.csv'))"),
            markdown("## Early-, middle-, and late-life performance"),
            code("regions = pd.read_csv(ROOT / 'life_region_metrics.csv')\ndisplay(regions[regions.level == 'ensemble'])"),
            markdown("## Presentation figures"),
            code("for path in sorted((ROOT / 'figures').glob('*.png')):\n    print(path.name)\n    display(Image(filename=str(path)))"),
            markdown("## Interpretation\n\nPhysics residual magnitudes are not directly comparable: Verhulst uses PCL/cycle, while DeepHPM RUL uses cycles/cycle. A lower residual does not by itself imply lower RUL prediction error."),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def generate_comparison(output_dir: Path = Path("artifacts/comparisons/rul"),
                        model_specs: dict[str, dict[str, Any]] = MODEL_SPECS,
                        notebook_path: Path | None = None) -> dict[str, Any]:
    _style(); output_dir.mkdir(parents=True, exist_ok=True); figures = output_dir / "figures"
    data = load_comparison_data(model_specs); result = calculate_comparison(data); summary = result["summary"]
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "alignment_report.json").write_text(json.dumps(summary["alignment"], indent=2) + "\n")
    for filename, key in (("per_seed_metrics.csv", "per_seed"), ("aggregate_metrics.csv", "aggregate"),
                          ("ensemble_metrics.csv", "ensembles"), ("per_battery_metrics.csv", "per_battery"),
                          ("life_region_metrics.csv", "life_regions"), ("horizon_binned_metrics.csv", "horizons"),
                          ("physics_residual_metrics.csv", "physics")):
        _write_csv(output_dir / filename, result[key])
    _save_aligned(data, output_dir / "aligned_ensemble_predictions.csv")
    plots = (
        (FIGURE_NAMES[0], _plot_parity(data, result["ensembles"])),
        (FIGURE_NAMES[1], _plot_trajectories(data, summary["representative_batteries"])),
        (FIGURE_NAMES[2], _plot_absolute_error(data)),
        (FIGURE_NAMES[3], _plot_horizons(result["horizons"])),
        (FIGURE_NAMES[4], _plot_per_battery(result["per_battery"])),
        (FIGURE_NAMES[5], _plot_metric_means(result["aggregate"])),
        (FIGURE_NAMES[6], _plot_distributions(data)),
        (FIGURE_NAMES[7], _plot_residuals(data)),
    )
    for name, figure in plots: _save_png(figure, figures, name)
    local_notebook = output_dir / "17_rul_model_comparison.ipynb"
    local_notebook.write_text(json.dumps(_notebook_payload("."), indent=1) + "\n")
    if notebook_path is not None:
        notebook_path.parent.mkdir(parents=True, exist_ok=True)
        notebook_path.write_text(json.dumps(_notebook_payload("../artifacts/comparisons/rul"), indent=1) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/comparisons/rul"))
    parser.add_argument("--notebook", type=Path, default=Path("notebooks/17_rul_model_comparison.ipynb"))
    args = parser.parse_args(); summary = generate_comparison(args.output_dir, notebook_path=args.notebook)
    print("Aligned:", summary["alignment"])
    print("Seed-mean RMSE ranking:", summary["ranking_by_seed_mean_rmse"])
    print("Output:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
