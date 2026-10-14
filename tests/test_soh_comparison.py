import csv
from pathlib import Path

import numpy as np
import pytest

from src.analyze_final import FORMATS
from src.compare_soh import (
    COMPARISON_FIGURES,
    calculate_comparison,
    generate_comparison,
    load_comparison_data,
)
from src.protocol import CORE_PREDICTION_COLUMNS, FIXED_SEEDS, PHYSICS_PREDICTION_COLUMNS


def _rows(model_offset: float, seed_offset: float, *, physics: bool):
    rows = []
    for battery_id in (85, 86, 87):
        for cycle in (1.0, 2.0, 3.0):
            target_soh = 1.0 - 0.02 * cycle - 0.001 * (battery_id - 85)
            predicted_soh = target_soh + model_offset + seed_offset * cycle
            row = {
                "battery_id": battery_id,
                "cycle": cycle,
                "target_pcl": 1.0 - target_soh,
                "predicted_pcl": 1.0 - predicted_soh,
                "target_soh": target_soh,
                "predicted_soh": predicted_soh,
                "soh_error": predicted_soh - target_soh,
            }
            if physics:
                row.update(
                    {
                        "time_derivative": 0.001,
                        "dynamics": 0.0008,
                        "physics_residual": 0.0002,
                    }
                )
            rows.append(row)
    return rows


def _write_predictions(path: Path, rows, *, physics: bool, reverse: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = CORE_PREDICTION_COLUMNS + (
        PHYSICS_PREDICTION_COLUMNS if physics else ()
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reversed(rows) if reverse else rows)


def _specs(root: Path):
    return {
        "data_driven": {
            "label": "DataDrivenNN",
            "path": root / "data_driven",
            "physics": False,
            "color": "#0072B2",
        },
        "verhulst": {
            "label": "VerhulstPINN",
            "path": root / "verhulst",
            "physics": True,
            "color": "#D55E00",
        },
        "deephpm": {
            "label": "DeepHPMNN",
            "path": root / "deephpm",
            "physics": True,
            "color": "#009E73",
        },
    }


def _write_all(root: Path):
    specs = _specs(root)
    offsets = {"data_driven": 0.003, "verhulst": 0.002, "deephpm": 0.001}
    for model_index, (key, spec) in enumerate(specs.items()):
        for seed_index, seed in enumerate(FIXED_SEEDS):
            rows = _rows(
                offsets[key],
                (seed_index - 1) * 0.0001,
                physics=spec["physics"],
            )
            _write_predictions(
                Path(spec["path"]) / f"seed_{seed}" / "test_predictions.csv",
                rows,
                physics=spec["physics"],
                reverse=(model_index + seed_index) % 2 == 1,
            )
    return specs


def test_all_models_and_seeds_align_by_battery_and_cycle(tmp_path: Path):
    data = load_comparison_data(_write_all(tmp_path))

    assert tuple(data.models) == ("data_driven", "verhulst", "deephpm")
    assert data.target_soh.size == 9
    assert np.all(np.diff(data.battery_id) >= 0)
    for model in data.models.values():
        assert model.seeds == FIXED_SEEDS
        assert model.predicted_soh.shape == (3, 9)
        assert np.array_equal(model.battery_id, data.battery_id)
        assert np.array_equal(model.cycle, data.cycle)
        assert np.array_equal(model.target_soh, data.target_soh)


def test_alignment_rejects_a_changed_target(tmp_path: Path):
    specs = _write_all(tmp_path)
    path = Path(specs["verhulst"]["path"]) / "seed_2345" / "test_predictions.csv"
    rows = _rows(0.002, 0.0, physics=True)
    rows[0]["target_pcl"] += 0.01
    rows[0]["target_soh"] -= 0.01
    rows[0]["soh_error"] = rows[0]["predicted_soh"] - rows[0]["target_soh"]
    _write_predictions(path, rows, physics=True)

    with pytest.raises(ValueError, match="targets"):
        load_comparison_data(specs)


def test_metrics_cover_seed_aggregate_ensemble_and_per_battery(tmp_path: Path):
    data = load_comparison_data(_write_all(tmp_path))
    summary, per_seed, aggregate, ensembles, per_battery, physics = calculate_comparison(data)

    assert summary["alignment"]["verified"] is True
    assert len(per_seed) == 9
    assert len(aggregate) == 3
    assert len(ensembles) == 3
    assert len(per_battery) == 9
    assert len(physics) == 8  # Three seeds plus one ensemble row for two PINNs.
    assert summary["ranking_by_seed_mean_rmse"][0] == "deephpm"


def test_comparison_generation_creates_all_reports_and_figure_formats(tmp_path: Path):
    specs = _write_all(tmp_path / "inputs")
    output = tmp_path / "comparison"

    generate_comparison(output, specs)

    for name in (
        "comparison_summary.json",
        "comparison_summary.md",
        "alignment_report.json",
        "per_seed_metrics.csv",
        "aggregate_metrics.csv",
        "ensemble_metrics.csv",
        "per_battery_metrics.csv",
        "physics_residual_metrics.csv",
        "aligned_ensemble_predictions.csv",
    ):
        assert (output / name).is_file()
    for figure in COMPARISON_FIGURES:
        for extension in FORMATS:
            path = output / "figures" / f"{figure}.{extension}"
            assert path.is_file()
            assert path.stat().st_size > 0
