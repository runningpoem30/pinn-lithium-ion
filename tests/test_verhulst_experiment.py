import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.run_verhulst import (
    REQUIRED_PHYSICS_WEIGHTS,
    VerhulstRunResult,
    _complete_metrics,
    aggregate_verhulst_results,
    load_verhulst_final_config,
    load_verhulst_tuning_config,
)
from src.protocol import FIXED_SEEDS


def test_verhulst_tuning_config_matches_deephpm_candidates_and_complete_splits():
    config = load_verhulst_tuning_config(Path("configs/verhulst_soh_tuning.yaml"))

    assert config.physics_weights == REQUIRED_PHYSICS_WEIGHTS
    assert config.base_experiment.training.seed == 1234
    assert config.base_experiment.training.early_stopping_monitor == "data"
    assert config.base_experiment.training.max_train_batches is None
    assert config.base_experiment.training.max_validation_batches is None
    assert config.base_experiment.model.surrogate_hidden_dims == (64, 64)


def test_final_config_uses_validation_selected_weight_and_frozen_protocol():
    config = load_verhulst_final_config(Path("configs/verhulst_soh_final.yaml"))

    assert config.seeds == FIXED_SEEDS
    assert config.base_experiment.training.physics_weight == 0.01
    assert config.base_experiment.training.epochs == 60
    assert config.base_experiment.training.max_train_batches is None
    assert config.base_experiment.training.max_validation_batches is None


def test_saved_tuning_report_used_validation_only_and_contains_no_test_outputs():
    tuning_dir = Path("artifacts/verhulst/physics_weight_tuning")
    report = json.loads((tuning_dir / "tuning_report.json").read_text())

    assert report["candidate_weights"] == list(REQUIRED_PHYSICS_WEIGHTS)
    assert report["selected_physics_weight"] == 0.01
    assert report["final_test_set_used"] is False
    assert not list(tuning_dir.glob("**/test_predictions.csv"))
    assert not list(tuning_dir.glob("**/test_metrics.json"))


def test_complete_metrics_use_frozen_soh_and_residual_definitions():
    target = np.asarray([1.0, 0.9, 0.8, 0.7])
    prediction = np.asarray([0.99, 0.91, 0.82, 0.68])
    residual = np.asarray([0.001, -0.001, 0.002, -0.002])
    evaluation = SimpleNamespace(
        metrics=SimpleNamespace(records=4, batteries=2),
        columns={
            "target_soh": target,
            "predicted_soh": prediction,
            "physics_residual": residual,
        },
    )

    metrics = _complete_metrics(evaluation)

    assert metrics["soh_rmse"] == metrics["rmse"]
    assert metrics["soh_mae"] == metrics["mae"]
    assert metrics["soh_r2"] == metrics["r2"]
    assert np.isclose(metrics["bias"], 0.0)
    assert np.isclose(metrics["physical_residual_rmse"], np.sqrt(2.5e-6))


def test_verhulst_aggregation_uses_all_fixed_seeds_and_sample_sd():
    results = []
    for seed, rmse in zip(FIXED_SEEDS, (0.01, 0.02, 0.03), strict=True):
        common = {
            "mae": rmse / 2,
            "rmse": rmse,
            "r2": 0.9,
            "bias": 0.001,
            "signed_error_p05": -0.01,
            "signed_error_p50": 0.0,
            "signed_error_p95": 0.01,
            "absolute_error_p50": 0.005,
            "absolute_error_p90": 0.01,
            "absolute_error_p95": 0.02,
            "absolute_error_p99": 0.03,
            "physical_residual_mae": 1e-5,
            "physical_residual_rmse": 2e-5,
        }
        results.append(
            VerhulstRunResult(
                seed=seed,
                best_epoch=1,
                epochs_completed=1,
                stopped_early=False,
                learned_parameters={
                    "rate_r_per_cycle": 0.005,
                    "capacity_K_pcl": 0.5,
                    "offset_C_pcl": 0.01,
                },
                metrics=common,
            )
        )

    aggregate = aggregate_verhulst_results(results)

    assert aggregate["seeds"] == list(FIXED_SEEDS)
    assert np.isclose(aggregate["rmse"]["mean"], 0.02)
    assert np.isclose(aggregate["rmse"]["std"], 0.01)
