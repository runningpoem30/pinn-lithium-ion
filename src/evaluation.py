"""Model-independent prediction validation, alignment, and SoH metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.protocol import CORE_PREDICTION_COLUMNS, PHYSICS_PREDICTION_COLUMNS


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    bias: float
    signed_error_p05: float
    signed_error_p50: float
    signed_error_p95: float
    absolute_error_p50: float
    absolute_error_p90: float
    absolute_error_p95: float
    absolute_error_p99: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionTable:
    """Validated cycle-level predictions from any comparison model."""

    columns: dict[str, np.ndarray]

    @property
    def keys(self) -> tuple[tuple[int, float], ...]:
        return tuple(
            (int(battery), float(cycle))
            for battery, cycle in zip(
                self.columns["battery_id"], self.columns["cycle"], strict=True
            )
        )

    @property
    def records(self) -> int:
        return int(self.columns["battery_id"].size)

    @property
    def battery_ids(self) -> set[int]:
        return set(self.columns["battery_id"].astype(np.int64).tolist())

    @property
    def has_physics(self) -> bool:
        return all(name in self.columns for name in PHYSICS_PREDICTION_COLUMNS)


def regression_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    """Return the single frozen regression-metric definition."""

    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Target and prediction must be non-empty arrays of equal shape")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Metrics require finite values")
    error = prediction - target
    denominator = np.square(target - target.mean()).sum()
    if denominator <= 0:
        raise ValueError("R2 is undefined for a constant target")
    absolute = np.abs(error)
    return RegressionMetrics(
        mae=float(absolute.mean()),
        rmse=float(np.sqrt(np.square(error).mean())),
        r2=float(1.0 - np.square(error).sum() / denominator),
        bias=float(error.mean()),
        signed_error_p05=float(np.percentile(error, 5)),
        signed_error_p50=float(np.percentile(error, 50)),
        signed_error_p95=float(np.percentile(error, 95)),
        absolute_error_p50=float(np.percentile(absolute, 50)),
        absolute_error_p90=float(np.percentile(absolute, 90)),
        absolute_error_p95=float(np.percentile(absolute, 95)),
        absolute_error_p99=float(np.percentile(absolute, 99)),
    ).to_dict()


def residual_metrics(residual: np.ndarray) -> dict[str, float]:
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise FloatingPointError("Physics residuals must be non-empty and finite")
    return {
        "physical_residual_mae": float(np.abs(values).mean()),
        "physical_residual_rmse": float(np.sqrt(np.square(values).mean())),
    }


def prediction_table_from_columns(
    columns: Mapping[str, np.ndarray], *, require_physics: bool = False
) -> PredictionTable:
    arrays = {name: np.asarray(values).reshape(-1) for name, values in columns.items()}
    missing = sorted(set(CORE_PREDICTION_COLUMNS) - set(arrays))
    if missing:
        raise ValueError(f"Prediction data is missing core columns: {missing}")
    if require_physics:
        missing_physics = sorted(set(PHYSICS_PREDICTION_COLUMNS) - set(arrays))
        if missing_physics:
            raise ValueError(
                f"Prediction data is missing physics columns: {missing_physics}"
            )
    present_physics = set(PHYSICS_PREDICTION_COLUMNS) & set(arrays)
    if present_physics and present_physics != set(PHYSICS_PREDICTION_COLUMNS):
        raise ValueError("Physics prediction columns must be present together or omitted")

    lengths = {name: values.size for name, values in arrays.items()}
    if not lengths or len(set(lengths.values())) != 1:
        raise ValueError(f"Prediction columns have inconsistent lengths: {lengths}")
    for name in CORE_PREDICTION_COLUMNS:
        if not np.all(np.isfinite(arrays[name])):
            raise FloatingPointError(f"Non-finite values in prediction column {name}")
    for name in present_physics:
        if not np.all(np.isfinite(arrays[name])):
            raise FloatingPointError(f"Non-finite values in prediction column {name}")

    battery_ids = arrays["battery_id"]
    if not np.all(battery_ids == np.floor(battery_ids)):
        raise ValueError("battery_id must contain integers")
    keys = [
        (int(battery), float(cycle))
        for battery, cycle in zip(battery_ids, arrays["cycle"], strict=True)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (battery_id, cycle) prediction keys")

    checks = {
        "target_soh": 1.0 - arrays["target_pcl"],
        "predicted_soh": 1.0 - arrays["predicted_pcl"],
        "soh_error": arrays["predicted_soh"] - arrays["target_soh"],
    }
    for name, expected in checks.items():
        if not np.allclose(arrays[name], expected, rtol=0.0, atol=2e-7):
            raise ValueError(f"Prediction column {name} violates its frozen definition")
    if present_physics:
        expected_residual = arrays["time_derivative"] - arrays["dynamics"]
        if not np.allclose(
            arrays["physics_residual"], expected_residual, rtol=1e-5, atol=1e-10
        ):
            raise ValueError("physics_residual must equal time_derivative - dynamics")
    return PredictionTable(arrays)


def load_prediction_csv(path: Path, *, require_physics: bool = False) -> PredictionTable:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    values = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    values = np.atleast_1d(values)
    if values.dtype.names is None:
        raise ValueError(f"Prediction CSV has no header: {path}")
    columns = {name: np.asarray(values[name]) for name in values.dtype.names}
    return prediction_table_from_columns(columns, require_physics=require_physics)


def align_prediction_tables(
    tables: Mapping[str, PredictionTable],
) -> dict[str, PredictionTable]:
    """Sort by battery/cycle and require identical targets across models."""

    if not tables:
        raise ValueError("At least one prediction table is required")
    aligned: dict[str, PredictionTable] = {}
    reference_keys: tuple[tuple[int, float], ...] | None = None
    reference_targets: tuple[np.ndarray, np.ndarray] | None = None
    for name, table in tables.items():
        positions = {key: index for index, key in enumerate(table.keys)}
        keys = tuple(sorted(positions))
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            missing = len(set(reference_keys) - set(keys))
            extra = len(set(keys) - set(reference_keys))
            raise ValueError(
                f"Prediction cohort {name} differs from the reference: "
                f"missing={missing}, extra={extra}"
            )
        order = np.asarray([positions[key] for key in reference_keys])
        ordered = PredictionTable(
            {column: values[order] for column, values in table.columns.items()}
        )
        targets = (ordered.columns["target_pcl"], ordered.columns["target_soh"])
        if reference_targets is None:
            reference_targets = targets
        elif not all(
            np.allclose(current, reference, rtol=0.0, atol=2e-7)
            for current, reference in zip(targets, reference_targets, strict=True)
        ):
            raise ValueError(f"Prediction targets for {name} differ from the reference")
        aligned[name] = ordered
    return aligned


def calculate_soh_metrics(table: PredictionTable) -> dict[str, float | int]:
    return {
        "records": table.records,
        "batteries": len(table.battery_ids),
        **regression_metrics(
            table.columns["target_soh"], table.columns["predicted_soh"]
        ),
    }
