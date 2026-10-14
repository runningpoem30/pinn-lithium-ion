from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import BatteryDataset
from src.evaluate import EvaluationMetrics
from src.models import DeepHPMNN
from src.run_final import FinalRunResult, aggregate_results, load_final_config


def result(seed: int, rmse: float) -> FinalRunResult:
    return FinalRunResult(
        seed=seed,
        best_epoch=10,
        epochs_completed=20,
        stopped_early=True,
        metrics=EvaluationMetrics(
            records=100,
            batteries=10,
            soh_mae=rmse / 2,
            soh_rmse=rmse,
            soh_r2=0.9,
            physical_residual_mae=1e-5,
            physical_residual_rmse=2e-5,
        ),
    )


def test_final_configuration_freezes_weight_seeds_and_complete_data():
    config = load_final_config(Path("configs/deephpm_soh_final.yaml"))
    assert config.seeds == (1234, 2345, 3456)
    assert config.base_experiment.training.physics_weight == 0.001
    assert config.base_experiment.training.max_train_batches is None
    assert config.base_experiment.training.max_validation_batches is None


def test_aggregate_reports_mean_and_sample_standard_deviation():
    aggregate = aggregate_results(
        [result(1234, 0.01), result(2345, 0.02), result(3456, 0.03)]
    )
    assert aggregate["num_seeds"] == 3
    assert aggregate["soh_rmse"]["mean"] == 0.02
    assert aggregate["soh_rmse"]["std"] == 0.01


def test_evaluation_preserves_battery_ids_and_exports_finite_physics_values():
    sample_count = 6
    features = np.linspace(-1.0, 1.0, sample_count * 8).reshape(sample_count, 8)
    cycles = np.arange(1, sample_count + 1, dtype=np.float64)[:, None]
    targets = np.linspace(0.01, 0.06, sample_count)[:, None]
    battery_ids = np.asarray([[85], [85], [86], [86], [87], [87]])
    dataset = BatteryDataset(
        features_raw=features,
        features_normalized=features,
        cycle_raw=cycles,
        cycle_normalized=cycles,
        target_pcl=targets,
        target_pcl_normalized=targets,
        battery_id=battery_ids,
    )
    model = DeepHPMNN(
        feature_dim=8,
        cycle_mean=0.0,
        cycle_scale=1.0,
        target_mean=0.0,
        target_scale=1.0,
        surrogate_hidden_dims=(8,),
        dynamics_hidden_dims=(4,),
    )

    from src.evaluate import evaluate_model

    evaluation = evaluate_model(
        model=model,
        loader=DataLoader(dataset, batch_size=2, shuffle=False),
        device=torch.device("cpu"),
    )

    assert evaluation.metrics.records == sample_count
    assert evaluation.metrics.batteries == 3
    assert evaluation.columns["battery_id"].tolist() == battery_ids.reshape(-1).tolist()
    for values in evaluation.columns.values():
        assert np.all(np.isfinite(values))
