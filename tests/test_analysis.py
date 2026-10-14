import csv
import json
from pathlib import Path

import numpy as np

from src.analyze_baseline import (
    BASELINE_FIGURE_NAMES,
    generate_baseline_analysis,
    load_aligned_baseline_predictions,
)
from src.analyze_final import (
    FIGURE_NAMES,
    FORMATS,
    generate_analysis,
    load_aligned_predictions,
    regression_metrics,
)


FIELDNAMES = (
    "battery_id",
    "cycle",
    "target_pcl",
    "predicted_pcl",
    "target_soh",
    "predicted_soh",
    "soh_error",
    "time_derivative",
    "dynamics",
    "physics_residual",
)

BASELINE_FIELDNAMES = FIELDNAMES[:7]


def _prediction_rows(seed_offset: float) -> list[dict[str, float | int]]:
    rows = []
    for battery_id in (85, 86, 87):
        for cycle in range(1, 5):
            target = 1.0 - 0.02 * cycle - 0.001 * (battery_id - 85)
            prediction = target + seed_offset + 0.001 * cycle
            derivative = -0.02 + seed_offset
            dynamics = derivative - 0.0005
            rows.append(
                {
                    "battery_id": battery_id,
                    "cycle": cycle,
                    "target_pcl": 1.0 - target,
                    "predicted_pcl": 1.0 - prediction,
                    "target_soh": target,
                    "predicted_soh": prediction,
                    "soh_error": prediction - target,
                    "time_derivative": derivative,
                    "dynamics": dynamics,
                    "physics_residual": derivative - dynamics,
                }
            )
    return rows


def _write_seed(root: Path, seed: int, rows: list[dict[str, float | int]]) -> None:
    directory = root / f"seed_{seed}"
    directory.mkdir(parents=True)
    with (directory / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "history.csv").open("w", newline="") as stream:
        names = (
            "epoch", "train_total", "train_data", "train_physics",
            "validation_total", "validation_data", "validation_physics",
            "gradient_norm", "improved",
        )
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for epoch in range(1, 4):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_total": 1 / epoch,
                    "train_data": 0.9 / epoch,
                    "train_physics": 0.1 / epoch,
                    "validation_total": 1.1 / epoch,
                    "validation_data": 1.0 / epoch,
                    "validation_physics": 0.1 / epoch,
                    "gradient_norm": 1.0,
                    "improved": True,
                }
            )
    (directory / "training_summary.json").write_text(json.dumps({"best_epoch": 3}))


def _write_baseline_seed(
    root: Path, seed: int, rows: list[dict[str, float | int]]
) -> None:
    directory = root / f"seed_{seed}"
    directory.mkdir(parents=True)
    with (directory / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=BASELINE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {name: row[name] for name in BASELINE_FIELDNAMES} for row in rows
        )
    with (directory / "history.csv").open("w", newline="") as stream:
        names = (
            "epoch", "train_total", "train_data", "train_physics",
            "validation_total", "validation_data", "validation_physics",
            "gradient_norm", "improved",
        )
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for epoch in range(1, 4):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_total": 1 / epoch,
                    "train_data": 1 / epoch,
                    "train_physics": 0.0,
                    "validation_total": 1.1 / epoch,
                    "validation_data": 1.1 / epoch,
                    "validation_physics": 0.0,
                    "gradient_norm": 1 / epoch,
                    "improved": True,
                }
            )
    (directory / "training_summary.json").write_text(json.dumps({"best_epoch": 3}))


def test_alignment_uses_battery_and_cycle_not_csv_row_order(tmp_path: Path):
    first = _prediction_rows(0.0)
    second = list(reversed(_prediction_rows(0.01)))
    _write_seed(tmp_path, 11, first)
    _write_seed(tmp_path, 22, second)

    aligned = load_aligned_predictions(tmp_path)

    assert aligned.seeds == (11, 22)
    assert np.all(np.diff(aligned.battery_id) >= 0)
    assert np.allclose(aligned.predicted_soh[1] - aligned.predicted_soh[0], 0.01)


def test_regression_metrics_include_bias_r2_and_error_percentiles():
    metrics = regression_metrics(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([0.1, 0.9, 2.2, 2.8]),
    )

    assert np.isclose(metrics["mae"], 0.15)
    assert np.isclose(metrics["rmse"], np.sqrt(0.025))
    assert np.isclose(metrics["bias"], 0.0)
    assert np.isclose(metrics["r2"], 0.98)
    assert metrics["absolute_error_p99"] >= metrics["absolute_error_p95"]
    assert metrics["absolute_error_p95"] >= metrics["absolute_error_p50"]


def test_generation_creates_all_tables_reports_and_figure_formats(tmp_path: Path):
    input_dir = tmp_path / "final"
    output_dir = tmp_path / "analysis"
    for seed, offset in ((11, -0.002), (22, 0.0), (33, 0.002)):
        _write_seed(input_dir, seed, _prediction_rows(offset))

    summary = generate_analysis(input_dir, output_dir)

    assert summary["records"] == 12
    assert summary["batteries"] == 3
    for name in (
        "analysis_summary.json",
        "analysis_summary.md",
        "per_battery_metrics.csv",
        "per_seed_metrics.csv",
        "aligned_ensemble_predictions.csv",
    ):
        assert (output_dir / name).is_file()
    for figure in FIGURE_NAMES:
        for extension in FORMATS:
            path = output_dir / "figures" / f"{figure}.{extension}"
            assert path.is_file()
            assert path.stat().st_size > 0


def test_baseline_alignment_uses_battery_and_cycle_not_csv_order(tmp_path: Path):
    _write_baseline_seed(tmp_path, 11, _prediction_rows(0.0))
    _write_baseline_seed(tmp_path, 22, list(reversed(_prediction_rows(0.01))))

    aligned = load_aligned_baseline_predictions(tmp_path)

    assert aligned.seeds == (11, 22)
    assert np.all(np.diff(aligned.battery_id) >= 0)
    assert np.allclose(aligned.predicted_soh[1] - aligned.predicted_soh[0], 0.01)


def test_baseline_generation_creates_expected_outputs(tmp_path: Path):
    input_dir = tmp_path / "final"
    output_dir = tmp_path / "analysis"
    for seed, offset in ((11, -0.002), (22, 0.0), (33, 0.002)):
        _write_baseline_seed(input_dir, seed, _prediction_rows(offset))

    summary = generate_baseline_analysis(input_dir, output_dir)

    assert summary["model_type"] == "data_driven"
    assert summary["physics_diagnostics"].startswith("not applicable")
    assert summary["records"] == 12
    for name in (
        "analysis_summary.json",
        "analysis_summary.md",
        "per_battery_metrics.csv",
        "per_seed_metrics.csv",
        "aligned_ensemble_predictions.csv",
    ):
        assert (output_dir / name).is_file()
    for figure in BASELINE_FIGURE_NAMES:
        for extension in FORMATS:
            path = output_dir / "figures" / f"{figure}.{extension}"
            assert path.is_file()
            assert path.stat().st_size > 0
