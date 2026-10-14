"""Evaluation and prediction export for fitted DeepHPM models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import DeepHPMNN


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
    model: DeepHPMNN,
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

    error = columns["soh_error"].astype(np.float64)
    target = columns["target_soh"].astype(np.float64)
    residual = columns["physics_residual"].astype(np.float64)
    total_sum_of_squares = np.square(target - target.mean()).sum()
    if total_sum_of_squares == 0:
        raise ValueError("Evaluation target is constant; R2 is undefined")
    metrics = EvaluationMetrics(
        records=int(target.size),
        batteries=int(np.unique(columns["battery_id"]).size),
        soh_mae=float(np.abs(error).mean()),
        soh_rmse=float(np.sqrt(np.square(error).mean())),
        soh_r2=float(1.0 - np.square(error).sum() / total_sum_of_squares),
        physical_residual_mae=float(np.abs(residual).mean()),
        physical_residual_rmse=float(np.sqrt(np.square(residual).mean())),
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
    column_order = (
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
    matrix = np.column_stack([result.columns[name] for name in column_order])
    np.savetxt(
        output_dir / "test_predictions.csv",
        matrix,
        delimiter=",",
        header=",".join(column_order),
        comments="",
        fmt=("%d",) + ("%.10g",) * (len(column_order) - 1),
    )
