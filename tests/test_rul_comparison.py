from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.compare_rul import FIGURE_NAMES, calculate_comparison, generate_comparison, load_comparison_data
from src.protocol import FIXED_SEEDS


def _rows(model: str, seed_index: int) -> list[dict[str, float | int]]:
    rows = []
    model_offset = {"data_driven": 5.0, "verhulst": 2.0, "deephpm": 0.0}[model]
    for battery in (85, 86, 87):
        eol = 1500 + 10 * (battery - 85)
        for cycle in (1, 300, 600, 900, 1200, 1300, 1450):
            target = eol - cycle
            prediction = target + model_offset + (seed_index - 1) * 2 + 0.002 * cycle
            row: dict[str, float | int] = {
                "battery_id": battery, "cycle": cycle, "eol_cycle": eol,
                "target_rul_cycles": target, "predicted_rul_cycles": prediction,
                "rul_error_cycles": prediction - target,
            }
            if model == "verhulst":
                row.update(target_pcl=.1, predicted_pcl=.11, time_derivative=.001,
                           verhulst_rhs=.0008, physics_residual=.0002,
                           threshold_crossing_possible=1, threshold_reached=0, non_crossing=0)
            elif model == "deephpm":
                row.update(time_derivative_rul_per_cycle=-.9,
                           learned_dynamics_rul_per_cycle=-.92,
                           physics_residual_rul_per_cycle=.02)
            rows.append(row)
    return rows


def _specs(root: Path, *, reverse_one: bool = False) -> dict[str, dict[str, object]]:
    specs = {
        "data_driven": {"label": "DataDrivenNN", "path": root / "data", "color": "#0072B2", "residual_column": None, "residual_units": None},
        "verhulst": {"label": "VerhulstPINN", "path": root / "verhulst", "color": "#D55E00", "residual_column": "physics_residual", "residual_units": "PCL per cycle"},
        "deephpm": {"label": "DeepHPMNN", "path": root / "deephpm", "color": "#009E73", "residual_column": "physics_residual_rul_per_cycle", "residual_units": "cycles per cycle"},
    }
    for model, spec in specs.items():
        for seed_index, seed in enumerate(FIXED_SEEDS):
            directory = Path(spec["path"]) / f"seed_{seed}"; directory.mkdir(parents=True)
            rows = _rows(model, seed_index)
            if reverse_one and model == "deephpm" and seed == FIXED_SEEDS[-1]: rows.reverse()
            with (directory / "test_predictions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return specs


def test_alignment_uses_battery_and_cycle_and_checks_targets(tmp_path: Path) -> None:
    specs = _specs(tmp_path, reverse_one=True)
    data = load_comparison_data(specs)
    assert data.target_rul.size == 21
    assert np.all(np.diff(data.battery_id) >= 0)
    path = Path(specs["verhulst"]["path"]) / f"seed_{FIXED_SEEDS[0]}" / "test_predictions.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["target_rul_cycles"] = str(float(rows[0]["target_rul_cycles"]) - 1)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="targets"):
        load_comparison_data(specs)


def test_metrics_cover_regions_horizons_batteries_and_physics(tmp_path: Path) -> None:
    result = calculate_comparison(load_comparison_data(_specs(tmp_path)))
    assert len(result["per_seed"]) == 9
    assert len(result["aggregate"]) == 3
    assert len(result["ensembles"]) == 3
    assert len(result["per_battery"]) == 9
    assert len(result["life_regions"]) == 3 * 4 * 3
    assert len(result["horizons"]) == 3 * 4 * 5
    assert len(result["physics"]) == 8
    assert result["summary"]["alignment"]["verified"] is True
    assert set(result["summary"]["models"]) == {"data_driven", "verhulst", "deephpm"}


def test_generation_writes_requested_outputs_and_notebook(tmp_path: Path) -> None:
    output = tmp_path / "comparison"; notebook = tmp_path / "17_rul.ipynb"
    generate_comparison(output, _specs(tmp_path / "inputs"), notebook)
    for name in (
        "comparison_summary.json", "alignment_report.json", "per_seed_metrics.csv",
        "aggregate_metrics.csv", "ensemble_metrics.csv", "per_battery_metrics.csv",
        "life_region_metrics.csv", "horizon_binned_metrics.csv",
        "physics_residual_metrics.csv", "aligned_ensemble_predictions.csv",
        "17_rul_model_comparison.ipynb",
    ):
        assert (output / name).is_file()
    for name in FIGURE_NAMES:
        path = output / "figures" / f"{name}.png"
        assert path.is_file() and path.stat().st_size > 0
    payload = json.loads(notebook.read_text())
    assert payload["nbformat"] == 4
    assert all("fit(" not in "".join(cell["source"]) for cell in payload["cells"])
