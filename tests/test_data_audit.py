import numpy as np
import pytest

from src.data_audit import audit_arrays, reconstruct_battery_boundaries


def sample_arrays():
    num_records = 6
    features = np.arange(num_records * 8, dtype=float).reshape(num_records, 8)
    return {
        "Features_mov_Flt": features,
        "RUL_Flt": np.array([[1], [0], [2], [1], [0], [0]], dtype=float),
        "PCL_Flt": np.linspace(0.0, 0.2, num_records)[:, None],
        "Cycles_Flt": np.array([[1], [2], [1], [2], [3], [1]], dtype=float),
        "Num_Cycles_Flt": np.array([[2], [3], [1]], dtype=float),
        "train_ind": np.array([[1]], dtype=float),
        "test_ind": np.array([[2]], dtype=float),
        "secondary_test_ind": np.array([[3]], dtype=float),
    }


def test_reconstruct_battery_boundaries():
    boundaries = reconstruct_battery_boundaries(np.array([[2], [3], [1]]))
    assert [(item.battery_id, item.start, item.stop) for item in boundaries] == [
        (1, 0, 2),
        (2, 2, 5),
        (3, 5, 6),
    ]


def test_audit_reports_expected_shape_and_split():
    report = audit_arrays(sample_arrays())
    assert report["num_records"] == 6
    assert report["num_batteries"] == 3
    assert report["num_features"] == 8
    assert report["model_input_width_with_cycle"] == 9
    assert report["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["split_record_counts"] == {"train": 2, "validation": 3, "test": 1}
    assert report["total_invalid_values"] == 0
    assert report["unassigned_battery_ids"] == []


def test_audit_rejects_battery_leakage():
    arrays = sample_arrays()
    arrays["secondary_test_ind"] = np.array([[1]])
    with pytest.raises(ValueError, match="Battery leakage"):
        audit_arrays(arrays)


def test_audit_rejects_cycle_count_mismatch():
    arrays = sample_arrays()
    arrays["Num_Cycles_Flt"] = np.array([[2], [2], [1]])
    with pytest.raises(ValueError, match="sums to"):
        audit_arrays(arrays)
