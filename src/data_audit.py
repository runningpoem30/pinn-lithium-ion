"""Validate and summarize the processed Severson battery dataset.

This module performs no model training. It checks the MATLAB arrays, rebuilds
the per-battery boundaries, validates the official battery-wise splits, and
reports data-quality statistics needed before implementing DeepHPM.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import loadmat


FEATURE_NAMES = (
    "slope",
    "intercept",
    "average_temperature",
    "internal_resistance",
    "charge_time",
    "dQdV_max",
    "dQdV_min",
    "dQdV_variance",
)

REQUIRED_ARRAYS = (
    "Features_mov_Flt",
    "RUL_Flt",
    "PCL_Flt",
    "Cycles_Flt",
    "Num_Cycles_Flt",
    "train_ind",
    "test_ind",
    "secondary_test_ind",
)


@dataclass(frozen=True)
class BatteryBoundary:
    """Half-open record range belonging to one battery."""

    battery_id: int
    start: int
    stop: int

    @property
    def num_cycles(self) -> int:
        return self.stop - self.start


def _as_vector(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim > 2 or (array.ndim == 2 and 1 not in array.shape):
        raise ValueError(f"{name} must be a row or column vector; got {array.shape}")
    return array.reshape(-1)


def _as_integer_vector(array: np.ndarray, name: str) -> np.ndarray:
    vector = _as_vector(array, name)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all(vector == np.floor(vector)):
        raise ValueError(f"{name} must contain integer values")
    return vector.astype(np.int64)


def reconstruct_battery_boundaries(num_cycles: np.ndarray) -> list[BatteryBoundary]:
    """Convert per-battery cycle counts into half-open flattened-data ranges."""

    counts = _as_integer_vector(num_cycles, "Num_Cycles_Flt")
    if counts.size == 0:
        raise ValueError("Num_Cycles_Flt is empty")
    if np.any(counts <= 0):
        raise ValueError("Every battery must contain at least one cycle")

    boundaries: list[BatteryBoundary] = []
    start = 0
    for battery_id, count in enumerate(counts, start=1):
        stop = start + int(count)
        boundaries.append(BatteryBoundary(battery_id, start, stop))
        start = stop
    return boundaries


def _numeric_stats(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(values)
    finite_values = values[finite]
    stats: dict[str, Any] = {
        "shape": list(values.shape),
        "nan_count": int(np.isnan(values).sum()),
        "positive_infinity_count": int(np.isposinf(values).sum()),
        "negative_infinity_count": int(np.isneginf(values).sum()),
    }
    if finite_values.size:
        stats.update(
            minimum=float(finite_values.min()),
            maximum=float(finite_values.max()),
            mean=float(finite_values.mean()),
            standard_deviation=float(finite_values.std()),
        )
    else:
        stats.update(minimum=None, maximum=None, mean=None, standard_deviation=None)
    return stats


def _validate_split(
    arrays: Mapping[str, np.ndarray], num_batteries: int
) -> tuple[dict[str, list[int]], list[int]]:
    source_names = {
        "train": "train_ind",
        "validation": "test_ind",
        "test": "secondary_test_ind",
    }
    splits: dict[str, list[int]] = {}
    for split_name, source_name in source_names.items():
        ids = _as_integer_vector(arrays[source_name], source_name)
        if np.any((ids < 1) | (ids > num_batteries)):
            raise ValueError(
                f"{source_name} contains an ID outside 1..{num_batteries}"
            )
        if np.unique(ids).size != ids.size:
            raise ValueError(f"{source_name} contains duplicate battery IDs")
        splits[split_name] = ids.tolist()

    names = tuple(splits)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = sorted(set(splits[left_name]) & set(splits[right_name]))
            if overlap:
                raise ValueError(
                    f"Battery leakage between {left_name} and {right_name}: {overlap}"
                )

    assigned = set().union(*(set(ids) for ids in splits.values()))
    unassigned = sorted(set(range(1, num_batteries + 1)) - assigned)
    return splits, unassigned


def audit_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Validate already-loaded MATLAB arrays and return a JSON-safe report."""

    missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
    if missing:
        raise KeyError(f"Missing required MATLAB arrays: {', '.join(missing)}")

    features = np.asarray(arrays["Features_mov_Flt"])
    if features.ndim != 2:
        raise ValueError(f"Features_mov_Flt must be two-dimensional; got {features.shape}")
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, found {features.shape[1]}"
        )

    vectors = {
        name: _as_vector(np.asarray(arrays[name]), name)
        for name in ("RUL_Flt", "PCL_Flt", "Cycles_Flt")
    }
    num_records = features.shape[0]
    for name, vector in vectors.items():
        if vector.size != num_records:
            raise ValueError(
                f"{name} has {vector.size} records but features have {num_records}"
            )

    boundaries = reconstruct_battery_boundaries(arrays["Num_Cycles_Flt"])
    reconstructed_records = boundaries[-1].stop
    if reconstructed_records != num_records:
        raise ValueError(
            "Num_Cycles_Flt sums to "
            f"{reconstructed_records}, but the dataset contains {num_records} records"
        )

    splits, unassigned = _validate_split(arrays, len(boundaries))
    cycles_by_battery = {
        boundary.battery_id: boundary.num_cycles for boundary in boundaries
    }
    split_record_counts = {
        split_name: sum(cycles_by_battery[battery_id] for battery_id in ids)
        for split_name, ids in splits.items()
    }

    feature_quality = []
    for column, feature_name in enumerate(FEATURE_NAMES):
        column_values = features[:, column]
        stats = _numeric_stats(column_values)
        finite_values = column_values[np.isfinite(column_values)]
        stats["name"] = feature_name
        stats["constant"] = bool(
            finite_values.size > 0 and np.ptp(finite_values) == 0
        )
        feature_quality.append(stats)

    per_battery_checks = []
    for boundary in boundaries:
        battery_cycles = vectors["Cycles_Flt"][boundary.start : boundary.stop]
        battery_rul = vectors["RUL_Flt"][boundary.start : boundary.stop]
        per_battery_checks.append(
            {
                **asdict(boundary),
                "num_cycles": boundary.num_cycles,
                "cycle_strictly_increasing": bool(
                    battery_cycles.size < 2 or np.all(np.diff(battery_cycles) > 0)
                ),
                "negative_rul_count": int((battery_rul < 0).sum()),
            }
        )

    array_quality = {
        "Features_mov_Flt": _numeric_stats(features),
        **{name: _numeric_stats(vector) for name, vector in vectors.items()},
    }
    invalid_total = sum(
        stats["nan_count"]
        + stats["positive_infinity_count"]
        + stats["negative_infinity_count"]
        for stats in array_quality.values()
    )

    return {
        "num_records": num_records,
        "num_batteries": len(boundaries),
        "num_features": features.shape[1],
        "model_input_width_with_cycle": features.shape[1] + 1,
        "feature_names": list(FEATURE_NAMES),
        "array_quality": array_quality,
        "feature_quality": feature_quality,
        "constant_features": [
            item["name"] for item in feature_quality if item["constant"]
        ],
        "total_invalid_values": invalid_total,
        "battery_cycle_counts": [boundary.num_cycles for boundary in boundaries],
        "minimum_cycles_per_battery": min(b.num_cycles for b in boundaries),
        "maximum_cycles_per_battery": max(b.num_cycles for b in boundaries),
        "split_source_mapping": {
            "train": "train_ind",
            "validation": "test_ind",
            "test": "secondary_test_ind",
        },
        "splits": splits,
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "split_record_counts": split_record_counts,
        "unassigned_battery_ids": unassigned,
        "batteries_with_non_increasing_cycles": [
            item["battery_id"]
            for item in per_battery_checks
            if not item["cycle_strictly_increasing"]
        ],
        "batteries_with_negative_rul": [
            item["battery_id"]
            for item in per_battery_checks
            if item["negative_rul_count"] > 0
        ],
        "battery_boundaries": per_battery_checks,
    }


def load_and_audit(path: Path) -> dict[str, Any]:
    """Load a MATLAB file and audit its required arrays."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    arrays = loadmat(path)
    report = audit_arrays(arrays)
    report["dataset_path"] = str(path.resolve())
    report["dataset_size_bytes"] = path.stat().st_size
    return report


def format_report(report: Mapping[str, Any]) -> str:
    """Render the most important audit findings for terminal use."""

    lines = [
        "Severson battery dataset audit",
        f"Dataset: {report['dataset_path']}",
        f"Records: {report['num_records']:,}",
        f"Batteries: {report['num_batteries']}",
        f"Engineered features: {report['num_features']}",
        f"Model inputs with cycle: {report['model_input_width_with_cycle']}",
        (
            "Cycles per battery: "
            f"{report['minimum_cycles_per_battery']}.."
            f"{report['maximum_cycles_per_battery']}"
        ),
        f"Invalid numeric values: {report['total_invalid_values']}",
        "Constant features: "
        + (", ".join(report["constant_features"]) or "none"),
        "Official split mapping:",
    ]
    for split_name in ("train", "validation", "test"):
        source = report["split_source_mapping"][split_name]
        ids = report["splits"][split_name]
        records = report["split_record_counts"][split_name]
        lines.append(
            f"  {split_name}: {len(ids)} batteries / {records:,} records "
            f"from {source} -> {ids}"
        )
    lines.extend(
        [
            "Unassigned battery IDs: "
            + (str(report["unassigned_battery_ids"]) if report["unassigned_battery_ids"] else "none"),
            "Batteries with non-increasing cycle values: "
            + (
                str(report["batteries_with_non_increasing_cycles"])
                if report["batteries_with_non_increasing_cycles"]
                else "none"
            ),
            "Batteries with negative RUL: "
            + (
                str(report["batteries_with_negative_rul"])
                if report["batteries_with_negative_rul"]
                else "none"
            ),
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("SeversonBattery.mat"),
        help="Path to the processed MATLAB dataset",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/data_audit.json"),
        help="Where to save the complete machine-readable audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_and_audit(args.data)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_report(report))
    print(f"Full JSON report: {args.json_output.resolve()}")


if __name__ == "__main__":
    main()
