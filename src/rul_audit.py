"""Audit and formally define the cycle-level RUL prediction task.

This module contains no neural-network code. It reconstructs the legacy RUL
labels battery by battery, verifies the processed MATLAB dataset, and writes a
frozen protocol for later DataDrivenNN, VerhulstPINN, and DeepHPMNN runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import loadmat

from src.data_audit import FEATURE_NAMES, audit_arrays, reconstruct_battery_boundaries
from src.protocol import FIXED_SEEDS


RUL_PROTOCOL_VERSION = 1
NOMINAL_CAPACITY_AH = 1.1
EOL_DISCHARGE_CAPACITY_AH = 0.88
EOL_SOH_THRESHOLD = EOL_DISCHARGE_CAPACITY_AH / NOMINAL_CAPACITY_AH
RUL_METRICS = (
    "mae_cycles",
    "rmse_cycles",
    "r2",
    "bias_cycles",
    "signed_error_p05_cycles",
    "signed_error_p50_cycles",
    "signed_error_p95_cycles",
    "absolute_error_p50_cycles",
    "absolute_error_p90_cycles",
    "absolute_error_p95_cycles",
    "absolute_error_p99_cycles",
)
RUL_PREDICTION_COLUMNS = (
    "battery_id",
    "cycle",
    "eol_cycle",
    "target_rul_cycles",
    "predicted_rul_cycles",
    "rul_error_cycles",
)


@dataclass(frozen=True)
class ReconstructedRUL:
    """Canonical cycle-level labels and battery endpoint metadata."""

    battery_id: np.ndarray
    cycle: np.ndarray
    eol_cycle: np.ndarray
    target_rul: np.ndarray
    stored_rul: np.ndarray
    endpoint_recorded: np.ndarray


def _integer_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all(array == np.floor(array)):
        raise ValueError(f"{name} must contain integer-valued cycles")
    return array.astype(np.int64)


def reconstruct_rul_labels(
    *, cycles: np.ndarray, stored_rul: np.ndarray, num_cycles: np.ndarray
) -> ReconstructedRUL:
    """Recover each EOL cycle and rebuild ``RUL = EOL cycle - cycle``.

    The endpoint is inferred independently within every battery from the
    invariant ``cycle + RUL``. This preserves known endpoints that are one
    cycle beyond the last processed record.
    """

    cycle_values = _integer_vector(cycles, "Cycles_Flt")
    rul_values = _integer_vector(stored_rul, "RUL_Flt")
    if cycle_values.size != rul_values.size:
        raise ValueError("Cycles_Flt and RUL_Flt have different record counts")
    boundaries = reconstruct_battery_boundaries(num_cycles)
    if boundaries[-1].stop != cycle_values.size:
        raise ValueError("Num_Cycles_Flt does not cover all RUL records")

    battery_ids = np.empty(cycle_values.size, dtype=np.int64)
    eol_cycles = np.empty(cycle_values.size, dtype=np.int64)
    endpoint_recorded = np.empty(cycle_values.size, dtype=bool)
    rebuilt = np.empty(cycle_values.size, dtype=np.int64)
    for boundary in boundaries:
        selection = slice(boundary.start, boundary.stop)
        battery_cycles = cycle_values[selection]
        battery_rul = rul_values[selection]
        if np.any(np.diff(battery_cycles) <= 0):
            raise ValueError(
                f"Battery {boundary.battery_id} cycles are not strictly increasing"
            )
        implied_endpoints = battery_cycles + battery_rul
        unique_endpoints = np.unique(implied_endpoints)
        if unique_endpoints.size != 1:
            raise ValueError(
                f"Battery {boundary.battery_id} has inconsistent cycle + RUL values"
            )
        endpoint = int(unique_endpoints[0])
        reconstructed = endpoint - battery_cycles
        if np.any(reconstructed < 0):
            raise ValueError(f"Battery {boundary.battery_id} has negative RUL")
        if not np.array_equal(reconstructed, battery_rul):
            raise ValueError(
                f"Battery {boundary.battery_id} stored RUL cannot be reconstructed"
            )
        battery_ids[selection] = boundary.battery_id
        eol_cycles[selection] = endpoint
        endpoint_recorded[selection] = bool(battery_cycles[-1] == endpoint)
        rebuilt[selection] = reconstructed

    return ReconstructedRUL(
        battery_id=battery_ids,
        cycle=cycle_values,
        eol_cycle=eol_cycles,
        target_rul=rebuilt,
        stored_rul=rul_values,
        endpoint_recorded=endpoint_recorded,
    )


def _cohort_sha256(labels: ReconstructedRUL) -> str:
    values = np.column_stack(
        (labels.battery_id, labels.cycle, labels.eol_cycle, labels.target_rul)
    )
    return hashlib.sha256(
        np.ascontiguousarray(values.astype("<i8")).tobytes()
    ).hexdigest()


def _subset_sha256(labels: ReconstructedRUL, battery_ids: list[int]) -> str:
    mask = np.isin(labels.battery_id, np.asarray(battery_ids))
    values = np.column_stack(
        (
            labels.battery_id[mask],
            labels.cycle[mask],
            labels.eol_cycle[mask],
            labels.target_rul[mask],
        )
    )
    return hashlib.sha256(
        np.ascontiguousarray(values.astype("<i8")).tobytes()
    ).hexdigest()


def _split_lookup(splits: Mapping[str, list[int]]) -> dict[int, str]:
    return {
        int(battery_id): split
        for split, battery_ids in splits.items()
        for battery_id in battery_ids
    }


def audit_rul_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], ReconstructedRUL]:
    """Validate the RUL target and return report, battery rows, and labels."""

    dataset = audit_arrays(arrays)
    labels = reconstruct_rul_labels(
        cycles=arrays["Cycles_Flt"],
        stored_rul=arrays["RUL_Flt"],
        num_cycles=arrays["Num_Cycles_Flt"],
    )
    pcl = np.asarray(arrays["PCL_Flt"], dtype=np.float64).reshape(-1)
    soh = 1.0 - pcl
    if soh.size != labels.cycle.size:
        raise ValueError("PCL_Flt and reconstructed RUL have different lengths")

    split_for_battery = _split_lookup(dataset["splits"])
    battery_rows: list[dict[str, Any]] = []
    threshold_visible_count = 0
    endpoint_recorded_count = 0
    affected_if_final_row_used = 0
    boundaries = reconstruct_battery_boundaries(arrays["Num_Cycles_Flt"])
    for boundary in boundaries:
        selection = slice(boundary.start, boundary.stop)
        cycles = labels.cycle[selection]
        target = labels.target_rul[selection]
        battery_soh = soh[selection]
        endpoint = int(labels.eol_cycle[boundary.start])
        crossings = np.flatnonzero(battery_soh < EOL_SOH_THRESHOLD)
        first_visible_crossing = (
            int(cycles[crossings[0]]) if crossings.size else None
        )
        endpoint_is_recorded = bool(cycles[-1] == endpoint)
        endpoint_recorded_count += int(endpoint_is_recorded)
        threshold_visible_count += int(crossings.size > 0)
        if not endpoint_is_recorded:
            affected_if_final_row_used += boundary.num_cycles
        if not np.all(np.diff(target) < 0):
            raise ValueError(
                f"Battery {boundary.battery_id} RUL is not strictly decreasing"
            )
        if not np.array_equal(np.diff(target), -np.diff(cycles)):
            raise ValueError(
                f"Battery {boundary.battery_id} RUL decrement does not match cycles"
            )
        battery_rows.append(
            {
                "battery_id": boundary.battery_id,
                "split": split_for_battery.get(boundary.battery_id, "unassigned"),
                "records": boundary.num_cycles,
                "first_recorded_cycle": int(cycles[0]),
                "last_recorded_cycle": int(cycles[-1]),
                "eol_cycle": endpoint,
                "first_rul_cycles": int(target[0]),
                "last_rul_cycles": int(target[-1]),
                "endpoint_recorded": endpoint_is_recorded,
                "processed_soh_threshold_visible": bool(crossings.size),
                "first_processed_soh_below_0p8_cycle": first_visible_crossing,
                "final_processed_soh": float(battery_soh[-1]),
                "minimum_processed_soh": float(battery_soh.min()),
                "cycle_strictly_increasing": True,
                "rul_strictly_decreasing": True,
                "rul_decrement_matches_cycle_increment": True,
            }
        )

    final_rul = np.asarray([row["last_rul_cycles"] for row in battery_rows])
    report: dict[str, Any] = {
        "audit_version": RUL_PROTOCOL_VERSION,
        "records": int(labels.cycle.size),
        "batteries": len(battery_rows),
        "units": "physical cycles",
        "legacy_definition": {
            "source_endpoint": "batch_combined(i).cycle_life",
            "threshold_interpretation": "first raw QDischarge < 0.88 Ah",
            "threshold_operator": "strictly less than",
            "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
            "equivalent_soh_threshold": EOL_SOH_THRESHOLD,
            "provenance_note": (
                "LoadData.m computes bat_label as the first QDischarge < 0.88 Ah "
                "and calls it the modeling output, but the supplied script does not "
                "assign bat_label back to cycle_life. ProcessData.m uses the existing "
                "cycle_life field directly."
            ),
            "operational_conclusion": (
                "RUL_Flt and cycle_life are authoritative; 0.88 Ah is the documented "
                "intended EOL interpretation, not a threshold recomputed from PCL_Flt."
            ),
            "target_formula": "RUL(cycle) = cycle_life - cycle",
            "matlab_statement": "RUL_i = batch_combined(i).cycle_life - Cycles_i",
            "authoritative_array": "RUL_Flt",
            "not_defined_as": "last recorded cycle - current cycle",
        },
        "reconstruction": {
            "formula": "eol_cycle = cycle + stored_RUL, constant within battery",
            "matches_stored_labels": bool(
                np.array_equal(labels.target_rul, labels.stored_rul)
            ),
            "integer_labels": True,
            "nonnegative_labels": bool(np.all(labels.target_rul >= 0)),
            "strictly_decreasing_within_every_battery": True,
            "decrement_equals_negative_cycle_increment": True,
            "minimum_rul_cycles": int(labels.target_rul.min()),
            "maximum_rul_cycles": int(labels.target_rul.max()),
            "cohort_sha256": _cohort_sha256(labels),
            "split_cohort_sha256": {
                name: _subset_sha256(labels, battery_ids)
                for name, battery_ids in dataset["splits"].items()
            },
        },
        "endpoint_observability": {
            "endpoint_recorded_batteries": endpoint_recorded_count,
            "endpoint_one_cycle_beyond_records_batteries": (
                len(battery_rows) - endpoint_recorded_count
            ),
            "final_rul_zero_batteries": int(np.sum(final_rul == 0)),
            "final_rul_one_batteries": int(np.sum(final_rul == 1)),
            "records_whose_labels_would_change_if_final_row_were_forced_to_zero": (
                affected_if_final_row_used
            ),
            "processed_soh_below_0p8_visible_batteries": threshold_visible_count,
            "explanation": (
                "The EOL metadata remain known even when the EOL row is absent. "
                "This is endpoint truncation in the processed rows, not an unknown/censored label."
            ),
        },
        "censoring": {
            "final_protocol_assumption": "all 124 RUL endpoints are treated as observed",
            "right_censored_batteries": 0,
            "source_filtering": [
                "five unfinished batch-1 batteries were removed in LoadData.m",
                "batch-3 cells ending above 0.885 Ah were removed in LoadData.m",
            ],
            "warning": (
                "A threshold crossing cannot be reliably reconstructed from processed "
                "PCL alone because preprocessing/outlier replacement and endpoint-row "
                "retention differ from the source cycle_life metadata."
            ),
        },
        "splits": dataset["splits"],
        "split_counts": dataset["split_counts"],
        "split_record_counts": dataset["split_record_counts"],
        "split_source_mapping": dataset["split_source_mapping"],
        "legacy_script_findings": {
            "loader": "functions.py stacks targets as [PCL_Flt, RUL_Flt]",
            "rul_target_selection": "legacy RUL scripts select the second target column",
            "units": "cycles",
            "case_a": "selected batteries 91 and 100 for training, 124 for test",
            "case_b": "selected batteries 101, 108, and 120 for training, 116 for test",
            "case_c": (
                "randomly splits cycle rows from the legacy validation group; this is "
                "not battery-wise and is rejected for the final protocol"
            ),
        },
    }
    return report, battery_rows, labels


def rul_evaluation_protocol(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen, model-independent protocol for future RUL runs."""

    return {
        "protocol_version": RUL_PROTOCOL_VERSION,
        "task": "cycle-level remaining useful life regression",
        "model_types": ["data_driven", "verhulst", "deephpm"],
        "units": "physical cycles",
        "target": {
            "column": "RUL_Flt",
            "formula": "target_rul_cycles = eol_cycle - current_cycle",
            "eol_definition": (
                "authoritative legacy cycle_life; LoadData.m documents the intended "
                "interpretation as first raw QDischarge < 0.88 Ah"
            ),
            "equivalent_soh_threshold": EOL_SOH_THRESHOLD,
            "endpoint_policy": "preserve cycle_life even if its row is absent",
            "censoring_policy": "all retained batteries are treated as fully observed",
            "target_clipping_or_capping": False,
        },
        "inputs": {
            "features": list(FEATURE_NAMES),
            "cycle": "raw physical cycle retained; normalized using training batteries only",
            "scaling": "fit every input and RUL target scaler on training batteries only",
            "forbidden": [
                "eol_cycle as a model input",
                "future measurements",
                "battery ID as a predictive feature",
            ],
        },
        "splits": {
            name: {
                "matlab_source": audit["split_source_mapping"][name],
                "battery_ids": ids,
                "batteries": audit["split_counts"][name],
                "records": audit["split_record_counts"][name],
            }
            for name, ids in audit["splits"].items()
        },
        "seeds": list(FIXED_SEEDS),
        "selection": {
            "hyperparameters_and_checkpoint": "validation batteries only",
            "primary_metric": "battery-macro RMSE in cycles",
            "definition": (
                "compute RMSE within each validation battery, then average equally "
                "over batteries"
            ),
            "tie_breaker": "cycle-weighted validation RMSE in cycles",
            "test_use": "once per seed after all choices are frozen",
        },
        "evaluation": {
            "primary_reporting": "battery-macro MAE and RMSE in cycles",
            "cycle_weighted_metrics": list(RUL_METRICS),
            "per_battery_metrics": list(RUL_METRICS),
            "seed_aggregation": "mean and sample standard deviation across three seeds",
            "ensemble": "equal-weight mean of predictions aligned by battery_id and cycle",
            "negative_prediction_policy": (
                "do not clip for primary metrics; report negative prediction count"
            ),
            "alignment_key": ["battery_id", "cycle"],
            "comparison_requirement": (
                "all models must use identical keys, targets, splits, seeds, and metrics"
            ),
        },
        "prediction_schema": list(RUL_PREDICTION_COLUMNS),
        "test_cohort_sha256": audit["reconstruction"]["split_cohort_sha256"]["test"],
        "artifact_layout": {
            model: f"artifacts/{model}/rul/final/seed_<seed>"
            for model in ("data_driven", "verhulst", "deephpm")
        },
        "per_seed_required_files": [
            "best_checkpoint.pt",
            "scalers.npz",
            "resolved_config.json",
            "history.csv",
            "training_summary.json",
            "test_metrics.json",
            "test_predictions.csv",
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_cycle_labels(
    path: Path,
    labels: ReconstructedRUL,
    splits: Mapping[str, list[int]],
) -> None:
    lookup = _split_lookup(splits)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "battery_id",
                "split",
                "cycle",
                "eol_cycle",
                "target_rul_cycles",
                "endpoint_recorded",
            )
        )
        for battery, cycle, endpoint, target, recorded in zip(
            labels.battery_id,
            labels.cycle,
            labels.eol_cycle,
            labels.target_rul,
            labels.endpoint_recorded,
            strict=True,
        ):
            writer.writerow(
                (
                    int(battery),
                    lookup[int(battery)],
                    int(cycle),
                    int(endpoint),
                    int(target),
                    bool(recorded),
                )
            )


def _write_test_reference(
    path: Path,
    labels: ReconstructedRUL,
    test_batteries: list[int],
) -> None:
    mask = np.isin(labels.battery_id, np.asarray(test_batteries))
    matrix = np.column_stack(
        (
            labels.battery_id[mask],
            labels.cycle[mask],
            labels.eol_cycle[mask],
            labels.target_rul[mask],
        )
    )
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header="battery_id,cycle,eol_cycle,target_rul_cycles",
        comments="",
        fmt="%d",
    )


def _write_markdown(
    path: Path, audit: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    endpoint = audit["endpoint_observability"]
    lines = [
        "# Frozen RUL task definition",
        "",
        "## Decision",
        "",
        "The authoritative target is the existing `RUL_Flt` value in physical cycles:",
        "",
        "`RUL(cycle) = legacy cycle_life - current cycle`",
        "",
        "The exact operational endpoint is the existing `cycle_life` metadata used by",
        "`ProcessData.m`. `LoadData.m` separately documents the intended endpoint as",
        "the first raw discharge capacity strictly below 0.88 Ah—equivalent to SoH <",
        "0.80 at 1.1 Ah nominal capacity—but does not assign its calculated `bat_label`",
        "back to `cycle_life`. Therefore `RUL_Flt` remains authoritative and is **not**",
        "redefined from processed SoH or from the final recorded cycle.",
        "",
        "## Audit findings",
        "",
        f"- Records: {audit['records']:,}",
        f"- Batteries: {audit['batteries']}",
        f"- Label range: {audit['reconstruction']['minimum_rul_cycles']}–{audit['reconstruction']['maximum_rul_cycles']} cycles",
        f"- Endpoint row present: {endpoint['endpoint_recorded_batteries']} batteries",
        f"- Endpoint one cycle beyond saved rows: {endpoint['endpoint_one_cycle_beyond_records_batteries']} batteries",
        f"- Processed SoH visibly below 0.80: {endpoint['processed_soh_below_0p8_visible_batteries']} batteries",
        f"- Right-censored labels under this protocol: {audit['censoring']['right_censored_batteries']}",
        "- Every battery has integer, nonnegative RUL that decreases exactly with cycle.",
        "",
        "The processed SoH curve is not used to recreate the endpoint because its",
        "outlier correction and row retention do not reproduce the source metadata",
        "crossing for most batteries. Forcing the last row to RUL 0 would alter",
        f"{endpoint['records_whose_labels_would_change_if_final_row_were_forced_to_zero']:,} labels.",
        "",
        "## Official battery splits",
        "",
    ]
    for name in ("train", "validation", "test"):
        split = protocol["splits"][name]
        lines.append(
            f"- {name}: {split['batteries']} batteries, {split['records']:,} records, "
            f"from `{split['matlab_source']}`"
        )
    lines.extend(
        [
            "",
            "## Final evaluation protocol",
            "",
            "- Fit scalers using training batteries only.",
            "- Select hyperparameters and checkpoints using validation batteries only.",
            "- Primary selection metric: battery-macro validation RMSE in cycles.",
            "- Evaluate the final test batteries once per fixed seed.",
            "- Report battery-macro, cycle-weighted, per-battery, per-seed, and ensemble results.",
            "- Do not clip negative predictions for primary metrics; report their count.",
            "- Align every model by `(battery_id, cycle)` and require identical targets.",
            "",
            "## Source evidence",
            "",
            "- `LoadData.m:80-88` computes the documented 0.88 Ah `bat_label`.",
            "- `ProcessData.m:37-45` truncates rows beyond `cycle_life`.",
            "- `ProcessData.m:70` constructs `RUL_i = cycle_life - Cycles_i`.",
            "- `functions.py:16,25` loads `RUL_Flt` as the second target column.",
            "",
            "The legacy Case A/B/C experiment splits are not adopted because they use",
            "small hand-picked battery sets or random cycle-level splits rather than the",
            "official battery-wise train/validation/test partition.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rul_audit_artifacts(
    *, dataset_path: Path, output_dir: Path
) -> dict[str, Any]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    arrays = loadmat(dataset_path)
    audit, battery_rows, labels = audit_rul_arrays(arrays)
    audit["dataset_path"] = str(dataset_path.resolve())
    audit["dataset_size_bytes"] = dataset_path.stat().st_size
    protocol = rul_evaluation_protocol(audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rul_task_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "rul_evaluation_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(output_dir / "rul_task_definition.md", audit, protocol)
    _write_csv(output_dir / "rul_battery_summary.csv", battery_rows)
    _write_cycle_labels(output_dir / "rul_cycle_labels.csv", labels, audit["splits"])
    _write_test_reference(
        output_dir / "rul_test_reference.csv", labels, audit["splits"]["test"]
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("SeversonBattery.mat"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/rul/task_definition")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_rul_audit_artifacts(
        dataset_path=args.data, output_dir=args.output_dir
    )
    endpoint = report["endpoint_observability"]
    print(
        f"RUL audit: {report['records']:,} records, {report['batteries']} batteries, "
        f"endpoint rows present for {endpoint['endpoint_recorded_batteries']} batteries"
    )
    print(f"Protocol: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
