import json
from pathlib import Path

import numpy as np
import pytest

from src.data_audit import FEATURE_NAMES
from src.evaluation import (
    align_prediction_tables,
    calculate_soh_metrics,
    prediction_table_from_columns,
    regression_metrics,
)
from src.protocol import (
    ArtifactLayout,
    CORE_PREDICTION_COLUMNS,
    FIXED_SEEDS,
    ModelType,
    OFFICIAL_SPLITS,
    PHYSICS_PREDICTION_COLUMNS,
    SPLIT_RECORD_COUNTS,
    validate_final_seed_artifacts,
)


def _columns(
    *, prediction_offset: float = 0.0, target_offset: float = 0.0
) -> dict[str, np.ndarray]:
    battery = np.asarray([85, 85, 86, 86])
    cycle = np.asarray([1.0, 2.0, 1.0, 2.0])
    target_pcl = np.asarray([0.02, 0.04, 0.03, 0.06]) + target_offset
    predicted_pcl = target_pcl + prediction_offset
    target_soh = 1.0 - target_pcl
    predicted_soh = 1.0 - predicted_pcl
    derivative = np.full(4, 0.001)
    dynamics = np.full(4, 0.0008)
    return {
        "battery_id": battery,
        "cycle": cycle,
        "target_pcl": target_pcl,
        "predicted_pcl": predicted_pcl,
        "target_soh": target_soh,
        "predicted_soh": predicted_soh,
        "soh_error": predicted_soh - target_soh,
        "time_derivative": derivative,
        "dynamics": dynamics,
        "physics_residual": derivative - dynamics,
    }


def test_protocol_freezes_features_splits_seeds_and_layout():
    assert len(FEATURE_NAMES) == 8
    assert FIXED_SEEDS == (1234, 2345, 3456)
    assert len(OFFICIAL_SPLITS["train"]) == 41
    assert len(OFFICIAL_SPLITS["validation"]) == 43
    assert len(OFFICIAL_SPLITS["test"]) == 40
    assert SPLIT_RECORD_COUNTS == {
        "train": 27584,
        "validation": 30457,
        "test": 41240,
    }
    layout = ArtifactLayout()
    assert layout.seed_dir(ModelType.DATA_DRIVEN, 1234) == Path(
        "artifacts/data_driven/final/seed_1234"
    )
    assert layout.seed_dir(ModelType.VERHULST, 2345) == Path(
        "artifacts/verhulst/final/seed_2345"
    )
    assert layout.seed_dir(ModelType.DEEPHPM, 3456) == Path(
        "artifacts/deephpm/final/seed_3456"
    )


def test_all_model_types_align_to_identical_ids_cycles_and_targets():
    baseline_columns = _columns(prediction_offset=0.01)
    for name in PHYSICS_PREDICTION_COLUMNS:
        baseline_columns.pop(name)
    baseline = prediction_table_from_columns(baseline_columns)
    verhulst = prediction_table_from_columns(_columns(prediction_offset=-0.01))
    deephpm_columns = {
        name: values[::-1] for name, values in _columns().items()
    }
    deephpm = prediction_table_from_columns(deephpm_columns)

    aligned = align_prediction_tables(
        {
            ModelType.DATA_DRIVEN.value: baseline,
            ModelType.VERHULST.value: verhulst,
            ModelType.DEEPHPM.value: deephpm,
        }
    )

    reference = aligned[ModelType.DATA_DRIVEN.value]
    assert tuple(reference.columns) == CORE_PREDICTION_COLUMNS
    for table in aligned.values():
        assert table.keys == reference.keys
        assert np.array_equal(table.columns["battery_id"], reference.columns["battery_id"])
        assert np.array_equal(table.columns["cycle"], reference.columns["cycle"])
        assert np.array_equal(table.columns["target_pcl"], reference.columns["target_pcl"])
        assert np.array_equal(table.columns["target_soh"], reference.columns["target_soh"])


def test_alignment_rejects_target_or_cohort_changes():
    reference = prediction_table_from_columns(_columns())
    changed_target = prediction_table_from_columns(_columns(target_offset=0.001))
    with pytest.raises(ValueError, match="targets"):
        align_prediction_tables({"reference": reference, "changed": changed_target})

    missing_row = prediction_table_from_columns(
        {name: values[:-1] for name, values in _columns().items()}
    )
    with pytest.raises(ValueError, match="cohort"):
        align_prediction_tables({"reference": reference, "missing": missing_row})


def test_evaluation_definition_is_shared_for_every_model_type():
    expected = regression_metrics(
        _columns()["target_soh"], _columns()["predicted_soh"]
    )
    for offset in (0.0, 0.01, -0.01):
        table = prediction_table_from_columns(_columns(prediction_offset=offset))
        metrics = calculate_soh_metrics(table)
        direct = regression_metrics(
            table.columns["target_soh"], table.columns["predicted_soh"]
        )
        assert {name: metrics[name] for name in direct} == direct
        if offset == 0.0:
            assert direct == expected


def test_completed_deephpm_runs_satisfy_frozen_protocol():
    reference_values = np.genfromtxt(
        "artifacts/protocol/soh_test_reference.csv",
        delimiter=",",
        names=True,
        dtype=np.float64,
    )
    reference = {
        name: np.asarray(reference_values[name]) for name in reference_values.dtype.names
    }
    layout = ArtifactLayout()
    for seed in FIXED_SEEDS:
        validate_final_seed_artifacts(
            model_type=ModelType.DEEPHPM,
            seed=seed,
            run_dir=layout.seed_dir(ModelType.DEEPHPM, seed),
            reference_scalers=Path("artifacts/data_scalers.npz"),
            reference_columns=reference,
        )

    manifest = json.loads(
        Path("artifacts/protocol/soh_comparison_protocol.json").read_text()
    )
    assert manifest["seeds"] == list(FIXED_SEEDS)
    assert manifest["features"] == list(FEATURE_NAMES)
    assert manifest["prediction_schema"]["required_common_columns"] == list(
        CORE_PREDICTION_COLUMNS
    )

    final_results = json.loads(
        Path("artifacts/deephpm/final/final_results.json").read_text()
    )
    from src.evaluation import load_prediction_csv

    for saved in final_results["per_seed"]:
        table = load_prediction_csv(
            layout.seed_dir(ModelType.DEEPHPM, saved["seed"])
            / "test_predictions.csv",
            require_physics=True,
        )
        current = calculate_soh_metrics(table)
        assert np.isclose(current["mae"], saved["soh_mae"], atol=1e-10)
        assert np.isclose(current["rmse"], saved["soh_rmse"], atol=1e-10)
        assert np.isclose(current["r2"], saved["soh_r2"], atol=1e-10)
