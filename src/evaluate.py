"""Evaluation and prediction export for fitted DeepHPM models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation import (
    prediction_table_from_columns,
    regression_metrics,
    residual_metrics,
)
from src.models import DeepHPMNN, VerhulstPINN
from src.protocol import CORE_PREDICTION_COLUMNS, PHYSICS_PREDICTION_COLUMNS


@dataclass(frozen=True)
class EvaluationMetrics:
    records: int
    batteries: int
    soh_mae: float
    soh_rmse: float
    soh_r2: float
    physical_residual_mae: float
    physical_residual_rmse: float


@dataclass(frozen=True)
class EvaluationResult:
    metrics: EvaluationMetrics
    columns: dict[str, np.ndarray]


def evaluate_model(
    *,
    model: DeepHPMNN | VerhulstPINN,
    loader: DataLoader,
    device: torch.device,
) -> EvaluationResult:
    """Evaluate a fitted model and retain cycle-level predictions."""

    model.eval()
    collected: dict[str, list[torch.Tensor]] = {
        "battery_id": [],
        "cycle": [],
        "target_pcl": [],
        "predicted_pcl": [],
        "time_derivative": [],
        "dynamics": [],
        "physics_residual": [],
    }
    for batch in loader:
        output = model(
            batch["features_normalized"].to(device),
            batch["cycle_raw"].to(device),
            create_graph=False,
        )
        collected["battery_id"].append(batch["battery_id"].detach().cpu())
        collected["cycle"].append(batch["cycle_raw"].detach().cpu())
        collected["target_pcl"].append(batch["target_pcl"].detach().cpu())
        collected["predicted_pcl"].append(output.pcl.detach().cpu())
        collected["time_derivative"].append(output.time_derivative.detach().cpu())
        collected["dynamics"].append(output.dynamics.detach().cpu())
        collected["physics_residual"].append(output.residual.detach().cpu())

    if not collected["target_pcl"]:
        raise ValueError("Evaluation loader produced no batches")
    columns = {
        name: torch.cat(parts).reshape(-1).numpy()
        for name, parts in collected.items()
    }
    columns["target_soh"] = 1.0 - columns["target_pcl"]
    columns["predicted_soh"] = 1.0 - columns["predicted_pcl"]
    columns["soh_error"] = columns["predicted_soh"] - columns["target_soh"]

    table = prediction_table_from_columns(columns, require_physics=True)
    common_metrics = regression_metrics(
        table.columns["target_soh"], table.columns["predicted_soh"]
    )
    physics_metrics = residual_metrics(table.columns["physics_residual"])
    metrics = EvaluationMetrics(
        records=table.records,
        batteries=len(table.battery_ids),
        soh_mae=common_metrics["mae"],
        soh_rmse=common_metrics["rmse"],
        soh_r2=common_metrics["r2"],
        physical_residual_mae=physics_metrics["physical_residual_mae"],
        physical_residual_rmse=physics_metrics["physical_residual_rmse"],
    )
    metric_values = [
        value
        for value in asdict(metrics).values()
        if isinstance(value, float)
    ]
    if not all(np.isfinite(metric_values)):
        raise FloatingPointError("Non-finite evaluation metrics detected")
    if not all(np.all(np.isfinite(values)) for values in columns.values()):
        raise FloatingPointError("Non-finite evaluation predictions detected")
    return EvaluationResult(metrics=metrics, columns=columns)


def save_evaluation(result: EvaluationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(asdict(result.metrics), indent=2) + "\n", encoding="utf-8"
    )
    column_order = CORE_PREDICTION_COLUMNS + PHYSICS_PREDICTION_COLUMNS
    matrix = np.column_stack([result.columns[name] for name in column_order])
    np.savetxt(
        output_dir / "test_predictions.csv",
        matrix,
        delimiter=",",
        header=",".join(column_order),
        comments="",
        fmt=("%d",) + ("%.10g",) * (len(column_order) - 1),
    )
