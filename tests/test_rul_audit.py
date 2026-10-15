import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from src.rul_audit import (
    RUL_PREDICTION_COLUMNS,
    audit_rul_arrays,
    build_rul_audit_artifacts,
    reconstruct_rul_labels,
    rul_evaluation_protocol,
)


def _arrays() -> dict[str, np.ndarray]:
    records = 6
    return {
        "Features_mov_Flt": np.arange(records * 8, dtype=float).reshape(records, 8),
        "RUL_Flt": np.asarray([[2], [1], [2], [1], [0], [0]], dtype=float),
        "PCL_Flt": np.asarray([[0.10], [0.15], [0.10], [0.15], [0.21], [0.20]]),
        "Cycles_Flt": np.asarray([[2], [3], [1], [2], [3], [1]], dtype=float),
        "Num_Cycles_Flt": np.asarray([[2], [3], [1]], dtype=float),
        "train_ind": np.asarray([[1]], dtype=float),
        "test_ind": np.asarray([[2]], dtype=float),
        "secondary_test_ind": np.asarray([[3]], dtype=float),
    }


def test_reconstructs_labels_and_preserves_unrecorded_endpoint():
    arrays = _arrays()
    labels = reconstruct_rul_labels(
        cycles=arrays["Cycles_Flt"],
        stored_rul=arrays["RUL_Flt"],
        num_cycles=arrays["Num_Cycles_Flt"],
    )

    assert labels.target_rul.tolist() == [2, 1, 2, 1, 0, 0]
    assert labels.eol_cycle.tolist() == [4, 4, 3, 3, 3, 1]
    assert labels.endpoint_recorded.tolist() == [False, False, True, True, True, True]


def test_reconstruction_rejects_inconsistent_cycle_plus_rul():
    arrays = _arrays()
    arrays["RUL_Flt"][1, 0] = 0
    with pytest.raises(ValueError, match=r"inconsistent cycle \+ RUL"):
        reconstruct_rul_labels(
            cycles=arrays["Cycles_Flt"],
            stored_rul=arrays["RUL_Flt"],
            num_cycles=arrays["Num_Cycles_Flt"],
        )


def test_audit_verifies_monotonicity_endpoints_and_splits():
    report, batteries, labels = audit_rul_arrays(_arrays())

    assert report["reconstruction"]["matches_stored_labels"] is True
    assert report["reconstruction"]["strictly_decreasing_within_every_battery"] is True
    assert report["endpoint_observability"]["endpoint_recorded_batteries"] == 2
    assert report["endpoint_observability"]["endpoint_one_cycle_beyond_records_batteries"] == 1
    assert report["endpoint_observability"]["processed_soh_below_0p8_visible_batteries"] == 1
    assert report["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert batteries[0]["last_rul_cycles"] == 1
    assert labels.target_rul.shape == (6,)


def test_protocol_freezes_batterywise_selection_and_prediction_schema():
    audit, _, _ = audit_rul_arrays(_arrays())
    protocol = rul_evaluation_protocol(audit)

    assert protocol["target"]["target_clipping_or_capping"] is False
    assert protocol["selection"]["primary_metric"] == "battery-macro RMSE in cycles"
    assert protocol["evaluation"]["alignment_key"] == ["battery_id", "cycle"]
    assert protocol["prediction_schema"] == list(RUL_PREDICTION_COLUMNS)
    assert "battery ID as a predictive feature" in protocol["inputs"]["forbidden"]
    assert protocol["test_cohort_sha256"] == audit["reconstruction"][
        "split_cohort_sha256"
    ]["test"]
    assert protocol["artifact_layout"]["deephpm"].startswith("artifacts/deephpm/rul")


def test_artifact_generation_writes_auditable_labels_and_protocol(tmp_path: Path):
    dataset = tmp_path / "sample.mat"
    output = tmp_path / "rul"
    savemat(dataset, _arrays())

    report = build_rul_audit_artifacts(dataset_path=dataset, output_dir=output)

    expected = (
        "rul_task_audit.json",
        "rul_evaluation_protocol.json",
        "rul_task_definition.md",
        "rul_battery_summary.csv",
        "rul_cycle_labels.csv",
        "rul_test_reference.csv",
    )
    assert all((output / name).is_file() for name in expected)
    assert json.loads((output / "rul_task_audit.json").read_text())["records"] == 6
    with (output / "rul_cycle_labels.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert rows[0]["target_rul_cycles"] == "2"
    assert report["censoring"]["right_censored_batteries"] == 0
