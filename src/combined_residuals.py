"""Create one scientifically valid SoH-and-RUL physics-residual figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analyze_final import _style


def _load(path: Path) -> dict[str, np.ndarray]:
    values = np.genfromtxt(path, delimiter=",", names=True)
    if not values.dtype.names:
        raise ValueError(f"No named columns in {path}")
    columns = {name: np.asarray(values[name]) for name in values.dtype.names}
    if not all(np.all(np.isfinite(value)) for value in columns.values()):
        raise FloatingPointError(f"Non-finite values in {path}")
    return columns


def _binned_rows(
    *, task: str, model: str, units: str, life_fraction: np.ndarray,
    residual: np.ndarray, edges: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (life_fraction >= lower) & (life_fraction < upper)
        values = residual[mask]
        if values.size == 0:
            raise ValueError(f"Empty life-fraction bin {lower}-{upper}")
        rows.append({
            "task": task, "model": model, "units": units,
            "life_fraction_lower": float(lower), "life_fraction_upper": float(min(upper, 1.0)),
            "life_fraction_center": float((lower + min(upper, 1.0)) / 2),
            "records": int(values.size), "residual_median": float(np.median(values)),
            "residual_p10": float(np.percentile(values, 10)),
            "residual_p90": float(np.percentile(values, 90)),
            "residual_mae": float(np.mean(np.abs(values))),
            "residual_rmse": float(np.sqrt(np.mean(np.square(values)))),
        })
    return rows


def generate_combined_residual_analysis(
    soh_predictions: Path,
    rul_predictions: Path,
    output_png: Path,
    output_csv: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    soh = _load(soh_predictions); rul = _load(rul_predictions)
    for key in ("battery_id", "cycle"):
        if not np.array_equal(soh[key], rul[key]):
            raise ValueError(f"SoH and RUL cohorts differ in {key}")
    required_soh = ("verhulst_physics_residual_mean", "deephpm_physics_residual_mean")
    required_rul = ("life_fraction", "verhulst_physics_residual_mean", "deephpm_physics_residual_mean")
    for name in required_soh:
        if name not in soh: raise ValueError(f"Missing SoH residual column: {name}")
    for name in required_rul:
        if name not in rul: raise ValueError(f"Missing RUL residual column: {name}")
    life = rul["life_fraction"]
    edges = np.linspace(0.0, 1.000001, 26)
    series = (
        ("SoH", "VerhulstPINN", "PCL per cycle", soh["verhulst_physics_residual_mean"], "#D55E00"),
        ("SoH", "DeepHPMNN", "PCL per cycle", soh["deephpm_physics_residual_mean"], "#009E73"),
        ("RUL", "VerhulstPINN", "PCL per cycle", rul["verhulst_physics_residual_mean"], "#D55E00"),
        ("RUL", "DeepHPMNN", "cycles per cycle", rul["deephpm_physics_residual_mean"], "#009E73"),
    )
    rows: list[dict[str, Any]] = []
    global_metrics: list[dict[str, Any]] = []
    for task, model, units, residual, _ in series:
        rows.extend(_binned_rows(task=task, model=model, units=units, life_fraction=life, residual=residual, edges=edges))
        global_metrics.append({
            "task": task, "model": model, "units": units, "records": int(residual.size),
            "residual_mae": float(np.mean(np.abs(residual))),
            "residual_rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "residual_mean": float(np.mean(residual)),
        })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    _style(); fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    panel_series = ((series[0], series[1]), (series[2],), (series[3],))
    titles = ("SoH residuals", "RUL Verhulst residual", "RUL DeepHPM residual")
    for axis, members, title in zip(axes, panel_series, titles, strict=True):
        for task, model, units, residual, color in members:
            selected = [row for row in rows if row["task"] == task and row["model"] == model]
            x = np.asarray([row["life_fraction_center"] for row in selected])
            median = np.asarray([row["residual_median"] for row in selected])
            lower = np.asarray([row["residual_p10"] for row in selected])
            upper = np.asarray([row["residual_p90"] for row in selected])
            metric = next(row for row in global_metrics if row["task"] == task and row["model"] == model)
            axis.plot(x, median, color=color, lw=2.2, label=f"{model} (RMSE {metric['residual_rmse']:.2e})")
            axis.fill_between(x, lower, upper, color=color, alpha=.16)
        axis.axhline(0, color="#222222", ls="--", lw=1)
        axis.set(xlabel="Elapsed battery life fraction", title=title)
        axis.set_ylabel(f"Physics residual ({members[0][2]})")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.legend(fontsize=8, loc="best")
    fig.suptitle("Physics residual behavior across SoH and RUL\n(lines: median; bands: 10th–90th percentile across held-out cycles)")
    fig.tight_layout(); output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_png.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "records": int(life.size), "batteries": int(np.unique(rul["battery_id"]).size),
        "x_axis": "elapsed life fraction = cycle / audited eol_cycle",
        "band_definition": "10th to 90th percentile of ensemble-mean residual within 25 life-fraction bins",
        "global_metrics": global_metrics,
        "interpretation": [
            "SoH residuals share PCL/cycle units, so their magnitudes may be compared as internal consistency diagnostics.",
            "DeepHPM has the smaller SoH residual, while Verhulst has the better SoH prediction metrics; residual size is not prediction accuracy.",
            "RUL Verhulst and RUL DeepHPM residuals have different units and must not be numerically compared.",
            "Residuals describe agreement with each model's equation, not proof that the learned dynamics are physically correct.",
        ],
    }
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    metrics = {(row["task"], row["model"]): row for row in global_metrics}
    output_markdown.write_text(
        "# Combined SoH and RUL physics-residual analysis\n\n"
        "The graph uses elapsed life fraction on a shared x-axis. Shaded bands show the 10th–90th percentile.\n\n"
        f"- SoH Verhulst residual RMSE: {metrics[('SoH', 'VerhulstPINN')]['residual_rmse']:.8g} PCL/cycle.\n"
        f"- SoH DeepHPM residual RMSE: {metrics[('SoH', 'DeepHPMNN')]['residual_rmse']:.8g} PCL/cycle.\n"
        f"- RUL Verhulst residual RMSE: {metrics[('RUL', 'VerhulstPINN')]['residual_rmse']:.8g} PCL/cycle.\n"
        f"- RUL DeepHPM residual RMSE: {metrics[('RUL', 'DeepHPMNN')]['residual_rmse']:.8g} cycles/cycle.\n\n"
        "DeepHPM is more internally consistent for SoH, but Verhulst predicts SoH more accurately. "
        "This demonstrates why physics residual and prediction error must be reported separately. "
        "The two RUL residual magnitudes cannot be directly compared because their units differ.\n",
        encoding="utf-8",
    )
    return summary
