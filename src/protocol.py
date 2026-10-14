"""Frozen protocol shared by all SoH model-comparison experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from src.data_audit import FEATURE_NAMES


PROTOCOL_VERSION = 1
FIXED_SEEDS = (1234, 2345, 3456)
SPLIT_SOURCE_MAPPING = {
    "train": "train_ind",
    "validation": "test_ind",
    "test": "secondary_test_ind",
}
OFFICIAL_SPLITS = {
    "train": tuple(range(2, 83, 2)),
    "validation": tuple(range(1, 84, 2)) + (84,),
    "test": tuple(range(85, 125)),
}
SPLIT_RECORD_COUNTS = {"train": 27584, "validation": 30457, "test": 41240}
CORE_PREDICTION_COLUMNS = (
    "battery_id",
    "cycle",
    "target_pcl",
    "predicted_pcl",
    "target_soh",
    "predicted_soh",
    "soh_error",
)
PHYSICS_PREDICTION_COLUMNS = (
    "time_derivative",
    "dynamics",
    "physics_residual",
)
COMMON_METRICS = (
    "mae",
    "rmse",
    "r2",
    "bias",
    "signed_error_p05",
    "signed_error_p50",
    "signed_error_p95",
    "absolute_error_p50",
    "absolute_error_p90",
    "absolute_error_p95",
    "absolute_error_p99",
)
PER_SEED_REQUIRED_FILES = (
    "best_checkpoint.pt",
    "scalers.npz",
    "resolved_config.json",
    "history.csv",
    "training_summary.json",
    "test_metrics.json",
    "test_predictions.csv",
)


class ModelType(str, Enum):
    DATA_DRIVEN = "data_driven"
    VERHULST = "verhulst"
    DEEPHPM = "deephpm"


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path = Path("artifacts")

    def final_dir(self, model_type: ModelType) -> Path:
        return self.root / model_type.value / "final"

    def seed_dir(self, model_type: ModelType, seed: int) -> Path:
        if seed not in FIXED_SEEDS:
            raise ValueError(f"Seed {seed} is outside the frozen protocol")
        return self.final_dir(model_type) / f"seed_{seed}"

    def comparison_dir(self, task: str = "soh") -> Path:
        if task not in {"soh", "rul"}:
            raise ValueError(f"Unsupported comparison task: {task}")
        return self.root / "comparisons" / task


def scaler_fingerprint(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Scaler artifact not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort_fingerprint(
    battery_id: np.ndarray,
    cycle: np.ndarray,
    target_pcl: np.ndarray,
) -> str:
    matrix = np.column_stack(
        (
            np.asarray(battery_id, dtype=np.float64).reshape(-1),
            np.asarray(cycle, dtype=np.float64).reshape(-1),
            np.asarray(target_pcl, dtype=np.float64).reshape(-1),
        )
    )
    order = np.lexsort((matrix[:, 1], matrix[:, 0]))
    canonical = np.ascontiguousarray(matrix[order].astype("<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def validate_bundle_protocol(bundle: Any, reference_scalers: Path) -> None:
    """Require the data bundle to satisfy the frozen splits and scalers."""

    report = bundle.audit_report
    if tuple(report["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Dataset features differ from the frozen eight features")
    for split, expected in OFFICIAL_SPLITS.items():
        if tuple(report["splits"][split]) != expected:
            raise ValueError(f"{split} battery IDs differ from the frozen protocol")
        if int(report["split_record_counts"][split]) != SPLIT_RECORD_COUNTS[split]:
            raise ValueError(f"{split} record count differs from the frozen protocol")
    if report["split_source_mapping"] != SPLIT_SOURCE_MAPPING:
        raise ValueError("Official MATLAB split mapping differs from the protocol")

    from src.data import DatasetScalers

    expected_scalers = DatasetScalers.load(reference_scalers)
    pairs = (
        (bundle.scalers.features.mean, expected_scalers.features.mean),
        (bundle.scalers.features.scale, expected_scalers.features.scale),
        (bundle.scalers.cycle.mean, expected_scalers.cycle.mean),
        (bundle.scalers.cycle.scale, expected_scalers.cycle.scale),
        (bundle.scalers.target_pcl.mean, expected_scalers.target_pcl.mean),
        (bundle.scalers.target_pcl.scale, expected_scalers.target_pcl.scale),
    )
    if not all(np.array_equal(actual, expected) for actual, expected in pairs):
        raise ValueError("Scalers differ from the training-only reference scalers")


def _reference_columns(bundle: Any) -> dict[str, np.ndarray]:
    tensors = bundle.test.tensors
    battery_id = tensors["battery_id"].numpy().reshape(-1).astype(np.int64)
    cycle = tensors["cycle_raw"].numpy().reshape(-1).astype(np.float64)
    target_pcl = tensors["target_pcl"].numpy().reshape(-1).astype(np.float64)
    order = np.lexsort((cycle, battery_id))
    return {
        "battery_id": battery_id[order],
        "cycle": cycle[order],
        "target_pcl": target_pcl[order],
        "target_soh": 1.0 - target_pcl[order],
    }


def save_test_reference(bundle: Any, path: Path) -> dict[str, np.ndarray]:
    columns = _reference_columns(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.column_stack(tuple(columns.values()))
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt=("%d", "%.10g", "%.10g", "%.10g"),
    )
    return columns


def validate_prediction_against_reference(
    prediction_path: Path,
    reference_columns: dict[str, np.ndarray],
    *,
    require_physics: bool,
) -> None:
    from src.evaluation import (
        PredictionTable,
        align_prediction_tables,
        load_prediction_csv,
    )

    prediction = load_prediction_csv(
        prediction_path, require_physics=require_physics
    )
    target_pcl = reference_columns["target_pcl"]
    reference = PredictionTable(
        {
            "battery_id": reference_columns["battery_id"],
            "cycle": reference_columns["cycle"],
            "target_pcl": target_pcl,
            "predicted_pcl": target_pcl,
            "target_soh": reference_columns["target_soh"],
            "predicted_soh": reference_columns["target_soh"],
            "soh_error": np.zeros_like(target_pcl),
        }
    )
    aligned = align_prediction_tables({"reference": reference, "candidate": prediction})
    candidate = aligned["candidate"]
    if candidate.records != SPLIT_RECORD_COUNTS["test"]:
        raise ValueError("Prediction record count differs from the official test split")
    if candidate.battery_ids != set(OFFICIAL_SPLITS["test"]):
        raise ValueError("Prediction battery IDs differ from the official test split")


def validate_final_seed_artifacts(
    *,
    model_type: ModelType,
    seed: int,
    run_dir: Path,
    reference_scalers: Path,
    reference_columns: dict[str, np.ndarray],
) -> None:
    """Validate the complete per-seed handoff required for fair comparison."""

    if seed not in FIXED_SEEDS:
        raise ValueError(f"Seed {seed} is outside the frozen protocol")
    missing = [
        name for name in PER_SEED_REQUIRED_FILES if not (run_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{run_dir} is missing required artifacts: {missing}")
    if scaler_fingerprint(run_dir / "scalers.npz") != scaler_fingerprint(
        reference_scalers
    ):
        raise ValueError(f"{run_dir} does not use the frozen training-only scalers")

    config = json.loads(
        (run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    training = config.get("training", {})
    if int(training.get("seed", -1)) != seed:
        raise ValueError(f"{run_dir} resolved configuration has the wrong seed")
    if training.get("early_stopping_monitor") != "data":
        raise ValueError(f"{run_dir} must select checkpoints by validation data loss")
    if training.get("max_train_batches") is not None:
        raise ValueError(f"{run_dir} does not use the complete training split")
    if training.get("max_validation_batches") is not None:
        raise ValueError(f"{run_dir} does not use the complete validation split")

    training_summary = json.loads(
        (run_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    if training_summary.get("early_stopping_monitor") != "data":
        raise ValueError(f"{run_dir} training summary violates checkpoint selection")
    validate_prediction_against_reference(
        run_dir / "test_predictions.csv",
        reference_columns,
        require_physics=model_type is not ModelType.DATA_DRIVEN,
    )


def protocol_manifest(
    *, scaler_path: Path, reference_columns: dict[str, np.ndarray]
) -> dict[str, Any]:
    layout = ArtifactLayout()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task": "soh",
        "model_types": [model.value for model in ModelType],
        "seeds": list(FIXED_SEEDS),
        "features": list(FEATURE_NAMES),
        "input_definition": "eight normalized features plus cycle normalized inside the model",
        "target_definition": {
            "pcl": "PCL_Flt",
            "soh": "1 - PCL_Flt",
        },
        "splits": {
            name: {
                "matlab_source": SPLIT_SOURCE_MAPPING[name],
                "battery_ids": list(ids),
                "records": SPLIT_RECORD_COUNTS[name],
            }
            for name, ids in OFFICIAL_SPLITS.items()
        },
        "scalers": {
            "fitted_from": "training batteries only",
            "reference_path": str(scaler_path),
            "sha256": scaler_fingerprint(scaler_path),
        },
        "selection": {
            "checkpoint_data": "validation split only",
            "early_stopping_monitor": "validation data loss",
            "test_timing": "once per seed after best checkpoint restoration",
            "test_based_model_or_seed_selection": False,
        },
        "metrics": {
            "common": list(COMMON_METRICS),
            "aggregation": "cycle-weighted per seed; report mean and sample standard deviation across seeds",
            "r2_definition": "1 - sum((prediction-target)^2) / sum((target-mean(target))^2)",
            "bias_definition": "mean(prediction - target)",
        },
        "prediction_schema": {
            "required_common_columns": list(CORE_PREDICTION_COLUMNS),
            "physics_extension_columns": list(PHYSICS_PREDICTION_COLUMNS),
            "alignment_key": ["battery_id", "cycle"],
            "physics_extension_policy": "all three fields together; required for physics-informed models and omitted for data-only baseline",
        },
        "test_cohort_sha256": cohort_fingerprint(
            reference_columns["battery_id"],
            reference_columns["cycle"],
            reference_columns["target_pcl"],
        ),
        "artifact_layout": {
            model.value: str(layout.final_dir(model)) for model in ModelType
        },
        "per_seed_required_files": list(PER_SEED_REQUIRED_FILES),
    }


def _write_protocol_readme(path: Path) -> None:
    lines = [
        "# Frozen SoH comparison protocol",
        "",
        "This directory is the machine-checkable contract for comparing DataDrivenNN,",
        "VerhulstPINN, and DeepHPMNN. Creating it loads data and validates existing",
        "artifacts but never trains or modifies a model.",
        "",
        "- Seeds: 1234, 2345, 3456",
        "- Features: the eight engineered features listed in the JSON manifest",
        "- Splits: 41 training, 43 validation, and 40 final-test batteries",
        "- Scalers: fitted on training batteries only",
        "- Checkpoint selection: validation data loss only",
        "- Test use: once per seed after checkpoint restoration",
        "- Alignment key: `(battery_id, cycle)`",
        "- Common metrics: MAE, RMSE, R2, bias, and signed/absolute error percentiles",
        "",
        "`soh_test_reference.csv` contains only frozen test keys and targets; it contains",
        "no model predictions. `soh_comparison_protocol.json` contains the full schema,",
        "split membership, scaler fingerprint, metric definitions, and artifact layout.",
        "",
        "Regenerate and validate with `python -m src.protocol`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_protocol_artifacts(
    *, dataset_path: Path, scaler_path: Path, output_dir: Path
) -> dict[str, Any]:
    from src.data import load_battery_data

    bundle = load_battery_data(dataset_path)
    validate_bundle_protocol(bundle, scaler_path)
    reference_path = output_dir / "soh_test_reference.csv"
    reference = save_test_reference(bundle, reference_path)

    layout = ArtifactLayout()
    for seed in FIXED_SEEDS:
        validate_final_seed_artifacts(
            model_type=ModelType.DEEPHPM,
            seed=seed,
            run_dir=layout.seed_dir(ModelType.DEEPHPM, seed),
            reference_scalers=scaler_path,
            reference_columns=reference,
        )
    manifest = protocol_manifest(scaler_path=scaler_path, reference_columns=reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "soh_comparison_protocol.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _write_protocol_readme(output_dir / "README.md")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("SeversonBattery.mat"))
    parser.add_argument("--scalers", type=Path, default=Path("artifacts/data_scalers.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/protocol"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_protocol_artifacts(
        dataset_path=args.data, scaler_path=args.scalers, output_dir=args.output_dir
    )
    print(
        f"Frozen SoH comparison protocol v{manifest['protocol_version']}: "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
