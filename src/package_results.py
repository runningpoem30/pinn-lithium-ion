"""Build the final presentation package from saved SoH and RUL comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analyze_final import _style
from src.combined_residuals import generate_combined_residual_analysis


MODEL_ORDER = ("data_driven", "verhulst", "deephpm")
MODEL_LABELS = {
    "data_driven": "DataDrivenNN", "verhulst": "VerhulstPINN", "deephpm": "DeepHPMNN"
}
COLORS = {"data_driven": "#0072B2", "verhulst": "#D55E00", "deephpm": "#009E73"}
FIGURE_SELECTION = (
    (2, "soh", "01_metric_means_with_seed_error_bars.png", "02_soh_metric_comparison.png",
     "SoH accuracy across three fixed seeds",
     "VerhulstPINN has the lowest SoH error, but all three models remain close."),
    (2, "soh", "03_representative_battery_trajectories.png", "02b_soh_trajectories.png",
     "Representative held-out SoH trajectories",
     "Measured and predicted degradation paths with ±1 seed standard deviation."),
    (3, "rul", "01_actual_vs_predicted_rul.png", "03_rul_actual_vs_predicted.png",
     "Actual versus predicted RUL",
     "All models underpredict long-horizon RUL; the data-driven ensemble has the lowest cycle-weighted RMSE."),
    (4, "rul", "03_absolute_error_vs_cycle_and_life_fraction.png", "04_rul_error_across_life.png",
     "RUL error across battery life",
     "Early-life RUL is substantially harder than late-life prediction."),
    (4, "rul", "04_horizon_binned_error.png", "04b_rul_horizon_error.png",
     "RUL error by prediction horizon",
     "Prediction error rises strongly as the remaining-life horizon increases."),
    (5, "soh", "07_physics_residuals.png", "05_soh_physics_residuals.png",
     "SoH physics residual diagnostics",
     "Residual consistency is a diagnostic and is not equivalent to predictive accuracy."),
    (5, "rul", "08_physics_residuals.png", "05b_rul_physics_residuals.png",
     "RUL physics residual diagnostics",
     "Verhulst and DeepHPM residuals are separated because their physical units differ."),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty required table: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _row_by_model(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    values = {row["model"]: row for row in rows}
    if set(values) != set(MODEL_ORDER):
        raise ValueError(f"Expected exactly the three frozen models; found {sorted(values)}")
    return values


def _alignment(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("verified") or not value.get("targets_identical"):
        raise ValueError(f"Unverified comparison alignment: {path}")
    if value.get("seeds") != [1234, 2345, 3456]:
        raise ValueError(f"Unexpected comparison seeds: {path}")
    return value


def _cohort_keys(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.genfromtxt(path, delimiter=",", names=True)
    return np.asarray(values["battery_id"], dtype=np.int64), np.asarray(values["cycle"], dtype=float)


def _compact_rows(soh_dir: Path, rul_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, directory in (("SoH", soh_dir), ("RUL", rul_dir)):
        aggregate = _row_by_model(_read_csv(directory / "aggregate_metrics.csv"))
        ensemble = _row_by_model(_read_csv(directory / "ensemble_metrics.csv"))
        per_battery = _read_csv(directory / "per_battery_metrics.csv")
        for model in MODEL_ORDER:
            battery_rmse = [float(row["rmse"]) for row in per_battery if row["model"] == model]
            rows.append({
                "task": task, "model": model, "model_label": MODEL_LABELS[model],
                "seed_mean_mae": float(aggregate[model]["mae_mean"]),
                "seed_sd_mae": float(aggregate[model]["mae_sample_std"]),
                "seed_mean_rmse": float(aggregate[model]["rmse_mean"]),
                "seed_sd_rmse": float(aggregate[model]["rmse_sample_std"]),
                "seed_mean_r2": float(aggregate[model]["r2_mean"]),
                "seed_sd_r2": float(aggregate[model]["r2_sample_std"]),
                "seed_mean_bias": float(aggregate[model]["bias_mean"]),
                "seed_sd_bias": float(aggregate[model]["bias_sample_std"]),
                "ensemble_mae": float(ensemble[model]["mae"]),
                "ensemble_rmse": float(ensemble[model]["rmse"]),
                "ensemble_r2": float(ensemble[model]["r2"]),
                "ensemble_bias": float(ensemble[model]["bias"]),
                "ensemble_absolute_error_p95": float(ensemble[model]["absolute_error_p95"]),
                "mean_per_battery_ensemble_rmse": float(np.mean(battery_rmse)),
            })
    return rows


def _life_region_rows(rul_dir: Path) -> list[dict[str, Any]]:
    source = _read_csv(rul_dir / "life_region_metrics.csv")
    return [
        {
            "model": row["model"], "model_label": row["model_label"], "region": row["region"],
            "records": int(row["records"]), "mae_cycles": float(row["mae"]),
            "rmse_cycles": float(row["rmse"]), "r2": float(row["r2"]), "bias_cycles": float(row["bias"]),
        }
        for row in source if row["level"] == "ensemble"
    ]


def _key_findings(compact: list[dict[str, Any]], regions: list[dict[str, Any]]) -> list[dict[str, str]]:
    by = {(row["task"], row["model"]): row for row in compact}
    soh_reduction = 100 * (by[("SoH", "data_driven")]["ensemble_rmse"] - by[("SoH", "verhulst")]["ensemble_rmse"]) / by[("SoH", "data_driven")]["ensemble_rmse"]
    region = {(row["model"], row["region"]): row for row in regions}
    return [
        {"id": "soh_result", "finding": f"VerhulstPINN has the lowest ensemble SoH RMSE ({by[('SoH', 'verhulst')]['ensemble_rmse']:.5f}), {soh_reduction:.2f}% below DataDrivenNN."},
        {"id": "rul_cycle_weighted", "finding": f"DataDrivenNN has the lowest cycle-weighted ensemble RUL RMSE ({by[('RUL', 'data_driven')]['ensemble_rmse']:.2f} cycles)."},
        {"id": "rul_battery_weighted", "finding": f"VerhulstPINN has the lowest mean per-battery ensemble RUL RMSE ({by[('RUL', 'verhulst')]['mean_per_battery_ensemble_rmse']:.2f} cycles), showing that weighting changes the ranking."},
        {"id": "early_life", "finding": f"DeepHPMNN is strongest in early-life ensemble RUL RMSE ({region[('deephpm', 'early')]['rmse_cycles']:.2f} cycles), but DataDrivenNN is strongest late in life ({region[('data_driven', 'late')]['rmse_cycles']:.2f} cycles)."},
        {"id": "bias", "finding": "All RUL ensembles have substantial negative bias, especially for long-horizon early-life predictions."},
        {"id": "physics_guardrail", "finding": "Physics residual magnitude measures internal equation consistency, not prediction accuracy; the RUL residuals also have different units across PINNs."},
    ]


def _limitations() -> list[str]:
    return [
        "Only three random seeds were run, so uncertainty estimates are descriptive rather than inferential.",
        "Evaluation uses one fixed 40-battery held-out cohort from the same dataset; external-dataset generalization was not tested.",
        "RUL ground truth follows the audited operational label RUL_Flt = cycle_life - cycle. The legacy intended 80% SoH threshold was not persisted consistently in preprocessing.",
        "RUL rankings differ between cycle-weighted and battery-weighted metrics; neither should be presented as a universal winner.",
        "All RUL models underpredict long-horizon remaining life, producing substantial negative bias and weak early-life accuracy.",
        "A small PINN residual does not prove that the learned dynamics are physically correct or improve held-out prediction.",
        "Verhulst RUL residuals use PCL/cycle, whereas DeepHPM RUL residuals use cycles/cycle; their numerical magnitudes cannot be compared directly.",
        "DeepHPM RUL learned dynamics did not recover the audited ideal derivative dRUL/dcycle = -1, revealing an identifiability limitation.",
        "The comparison describes frozen test performance only. It must not be used for further model or hyperparameter selection.",
    ]


def _overview_figure(compact: list[dict[str, Any]], path: Path) -> None:
    _style(); fig, axes = plt.subplots(2, 2, figsize=(12, 8.2)); x = np.arange(3)
    for row_index, task in enumerate(("SoH", "RUL")):
        rows = [next(row for row in compact if row["task"] == task and row["model"] == model) for model in MODEL_ORDER]
        axes[row_index, 0].bar(x, [row["ensemble_rmse"] for row in rows], color=[COLORS[m] for m in MODEL_ORDER])
        axes[row_index, 1].bar(x, [row["ensemble_r2"] for row in rows], color=[COLORS[m] for m in MODEL_ORDER])
        units = "" if task == "SoH" else " (cycles)"
        axes[row_index, 0].set(title=f"{task} ensemble RMSE", ylabel=f"RMSE{units}")
        axes[row_index, 1].set(title=f"{task} ensemble $R^2$", ylabel="$R^2$")
        for axis in axes[row_index]:
            axis.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=10)
    fig.suptitle("Final held-out SoH and RUL comparison\n(descriptive test results; not a model-selection step)")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _slide_plan(captions: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Recommended results slides", "",
        "The package reports frozen held-out results. Test rankings are descriptive and were not used to retune or select checkpoints.", "",
        "## Slide 1 — Experimental protocol and headline results", "",
        "Use `01_cross_task_results_overview.png` with the compact results table.",
        "State the battery-wise split, three fixed seeds, validation-only checkpoint/hyperparameter selection, and single final test evaluation.", "",
        "## Slide 2 — SoH results", "",
        "Use the SoH metric and trajectory figures. Headline: VerhulstPINN has the best SoH metrics, while improvements over the other models are modest.", "",
        "## Slide 3 — Overall RUL results", "",
        "Use actual-versus-predicted RUL. Headline: DataDrivenNN is best cycle-weighted; all models underpredict long remaining-life horizons.", "",
        "## Slide 4 — Where RUL models succeed and fail", "",
        "Use life-fraction and horizon figures. Headline: DeepHPM is strongest early, while DataDrivenNN is strongest in middle/late life.", "",
        "## Slide 5 — Physics diagnostics", "",
        "Use the residual figures. Do not compare Verhulst and DeepHPM RUL residual magnitudes because their units differ.", "",
        "## Slide 6 — Conclusions and limitations", "",
        "Use `key_findings.md` and `limitations.md`. Avoid claiming a universal winner or causal physical discovery.", "",
        "## Figure captions", "",
    ]
    for row in captions:
        lines += [f"### Slide {row['slide']} — {row['title']}", "", row["caption"], "", f"File: `{row['file']}`", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_results_package(
    soh_dir: Path = Path("artifacts/comparisons/soh"),
    rul_dir: Path = Path("artifacts/comparisons/rul"),
    output_dir: Path = Path("artifacts/presentation/final_results"),
) -> dict[str, Any]:
    soh_alignment = _alignment(soh_dir / "alignment_report.json")
    rul_alignment = _alignment(rul_dir / "alignment_report.json")
    soh_ids, soh_cycles = _cohort_keys(soh_dir / "aligned_ensemble_predictions.csv")
    rul_ids, rul_cycles = _cohort_keys(rul_dir / "aligned_ensemble_predictions.csv")
    if not np.array_equal(soh_ids, rul_ids) or not np.array_equal(soh_cycles, rul_cycles):
        raise ValueError("SoH and RUL comparison cohorts do not share identical battery-cycle keys")
    if soh_alignment["records"] != rul_alignment["records"] or soh_alignment["batteries"] != rul_alignment["batteries"]:
        raise ValueError("SoH and RUL alignment reports describe different cohorts")
    output_dir.mkdir(parents=True, exist_ok=True); figure_dir = output_dir / "figures"; figure_dir.mkdir(exist_ok=True)
    compact = _compact_rows(soh_dir, rul_dir); regions = _life_region_rows(rul_dir)
    findings = _key_findings(compact, regions); limitations = _limitations()
    _write_csv(output_dir / "compact_results_table.csv", compact)
    _write_csv(output_dir / "rul_life_region_summary.csv", regions)
    _write_csv(output_dir / "key_findings.csv", findings)
    _overview_figure(compact, figure_dir / "01_cross_task_results_overview.png")
    captions: list[dict[str, Any]] = [{
        "slide": 1, "task": "SoH + RUL", "file": "figures/01_cross_task_results_overview.png",
        "title": "Cross-task result overview",
        "caption": "Ensemble SoH and RUL metrics on the same 40 held-out batteries. Results are descriptive and were not used for model selection.",
    }]
    for slide, task, source_name, target_name, title, caption in FIGURE_SELECTION:
        source_dir = soh_dir if task == "soh" else rul_dir
        target = figure_dir / target_name; shutil.copy2(source_dir / "figures" / source_name, target)
        captions.append({"slide": slide, "task": task.upper(), "file": f"figures/{target_name}", "title": title, "caption": caption})
    combined_name = "06_combined_soh_rul_physics_residuals.png"
    generate_combined_residual_analysis(
        soh_dir / "aligned_ensemble_predictions.csv",
        rul_dir / "aligned_ensemble_predictions.csv",
        figure_dir / combined_name,
        output_dir / "combined_physics_residual_bins.csv",
        output_dir / "combined_physics_residual_summary.json",
        output_dir / "combined_physics_residual_analysis.md",
    )
    captions.append({
        "slide": 5, "task": "SoH + RUL", "file": f"figures/{combined_name}",
        "title": "Combined SoH and RUL physics residuals",
        "caption": "Residuals across elapsed battery life. SoH residuals share PCL/cycle units; the two RUL panels use different units and must be interpreted separately.",
    })
    _write_csv(output_dir / "figure_captions.csv", captions)
    _slide_plan(captions, output_dir / "slide_plan.md")
    (output_dir / "key_findings.md").write_text("# Key findings\n\n" + "\n".join(f"- {row['finding']}" for row in findings) + "\n")
    (output_dir / "limitations.md").write_text("# Limitations and interpretation guardrails\n\n" + "\n".join(f"- {value}" for value in limitations) + "\n")
    commands = """# Reproducibility commands

These commands regenerate reports from saved predictions only; they do not train or select models.

```bash
PYTHONPATH=. .venv/bin/python -m src.compare_soh
PYTHONPATH=. .venv/bin/python -m src.compare_rul
PYTHONPATH=. .venv/bin/python -m src.package_results
PYTHONPATH=. .venv/bin/python -m pytest -q
```

Frozen experimental controls: official battery-wise splits, eight features,
training-only scalers, seeds 1234/2345/3456, and validation-only checkpoint and
physics-weight selection. Test predictions are consumed only for final descriptive evaluation.
"""
    (output_dir / "reproducibility.md").write_text(commands, encoding="utf-8")
    payload = {
        "package": "Final SoH and RUL results package", "created_from_saved_artifacts_only": True,
        "retraining_performed": False, "test_results_used_for_model_selection": False,
        "cohort": {"records": soh_alignment["records"], "batteries": soh_alignment["batteries"],
                   "seeds": soh_alignment["seeds"], "keys_identical_across_tasks": True},
        "compact_results": compact, "rul_life_regions": regions,
        "key_findings": findings, "limitations": limitations, "figure_captions": captions,
    }
    (output_dir / "results_package.json").write_text(json.dumps(payload, indent=2) + "\n")
    readme = """# Final presentation-ready SoH and RUL results

This directory was built entirely from frozen Step 12 and Step 17 comparison
artifacts. No network was loaded or retrained, and no test result was used to
select a model, checkpoint, or hyperparameter.

- `compact_results_table.csv`: cross-task seed and ensemble metrics
- `rul_life_region_summary.csv`: early/middle/late-life RUL performance
- `slide_plan.md`: recommended six-slide narrative
- `figure_captions.csv`: captions matched to packaged figures
- `combined_physics_residual_analysis.md`: joint SoH/RUL residual interpretation
- `combined_physics_residual_bins.csv`: plotted life-fraction statistics
- `key_findings.md`: defensible presentation statements
- `limitations.md`: interpretation guardrails
- `reproducibility.md`: saved-artifact regeneration commands
- `results_package.json`: machine-readable package
- `package_manifest.json`: source and output checksums
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    sources = [soh_dir / "comparison_summary.json", soh_dir / "aggregate_metrics.csv", soh_dir / "ensemble_metrics.csv",
               rul_dir / "comparison_summary.json", rul_dir / "aggregate_metrics.csv", rul_dir / "ensemble_metrics.csv",
               rul_dir / "life_region_metrics.csv"]
    outputs = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "package_manifest.json")
    manifest = {
        "source_artifacts": [{"path": str(path), "sha256": _sha256(path)} for path in sources],
        "packaged_files": [{"path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in outputs],
    }
    (output_dir / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soh-dir", type=Path, default=Path("artifacts/comparisons/soh"))
    parser.add_argument("--rul-dir", type=Path, default=Path("artifacts/comparisons/rul"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/presentation/final_results"))
    args = parser.parse_args(); payload = build_results_package(args.soh_dir, args.rul_dir, args.output_dir)
    print("Created final results package without retraining:", args.output_dir.resolve())
    print("Cohort:", payload["cohort"])


if __name__ == "__main__":
    main()
