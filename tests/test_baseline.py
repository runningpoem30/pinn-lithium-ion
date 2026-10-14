from pathlib import Path

import numpy as np
import torch

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.data import build_data_pipeline_from_arrays, create_data_loaders
from src.models import DataDrivenNN
from src.protocol import CORE_PREDICTION_COLUMNS, FIXED_SEEDS
from src.run_baseline import (
    BaselineRunResult,
    aggregate_baseline_results,
    evaluate_baseline,
    fit_baseline,
    load_baseline_config,
)
from src.train import load_checkpoint, set_random_seed


def _arrays():
    counts = np.asarray([[6], [5], [4]], dtype=float)
    battery = np.repeat([1, 2, 3], counts.reshape(-1).astype(int))
    cycle = np.concatenate([np.arange(1, count + 1) for count in (6, 5, 4)])
    features = np.column_stack(
        [
            battery * (column + 1) + cycle * (0.1 + column * 0.01)
            for column in range(8)
        ]
    )
    pcl = (0.01 * battery + 0.002 * cycle)[:, None]
    rul = np.concatenate(
        [np.arange(5, -1, -1), np.arange(4, -1, -1), np.arange(3, -1, -1)]
    )
    return {
        "Features_mov_Flt": features,
        "RUL_Flt": rul[:, None].astype(float),
        "PCL_Flt": pcl,
        "Cycles_Flt": cycle[:, None].astype(float),
        "Num_Cycles_Flt": counts,
        "train_ind": np.asarray([[1]], dtype=float),
        "test_ind": np.asarray([[2]], dtype=float),
        "secondary_test_ind": np.asarray([[3]], dtype=float),
    }


def _config(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(batch_size=3),
        model=ModelConfig(
            surrogate_hidden_dims=(8, 8), dynamics_hidden_dims=(4, 4)
        ),
        training=TrainingConfig(
            seed=7,
            epochs=3,
            learning_rate=0.001,
            physics_weight=0.0,
            gradient_clip_norm=1.0,
            early_stopping_patience=2,
            max_train_batches=1,
            max_validation_batches=1,
            device="cpu",
        ),
        output_dir=str(output_dir),
    )


def test_final_baseline_config_matches_frozen_deephpm_ablation():
    config = load_baseline_config(Path("configs/data_driven_soh_final.yaml"))
    assert config.seeds == FIXED_SEEDS
    assert config.base_experiment.model.surrogate_hidden_dims == (64, 64)
    assert config.base_experiment.data.batch_size == 2048
    assert config.base_experiment.training.physics_weight == 0.0
    assert config.base_experiment.training.max_train_batches is None
    assert config.base_experiment.training.max_validation_batches is None


def test_data_driven_model_is_surrogate_only_and_backpropagates():
    bundle = build_data_pipeline_from_arrays(_arrays())
    model = DataDrivenNN.from_scalers(
        bundle.scalers, feature_dim=8, surrogate_hidden_dims=(8, 8)
    )
    assert not hasattr(model, "dynamics_network")
    batch = next(iter(create_data_loaders(bundle, batch_size=3).train))
    output = model(batch["features_normalized"], batch["cycle_raw"])
    loss = torch.mean(
        torch.square(output.pcl_normalized - batch["target_pcl_normalized"])
    )
    loss.backward()
    assert output.pcl.shape == (3, 1)
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters())


def test_baseline_smoke_training_updates_model_and_logs_zero_physics(tmp_path: Path):
    set_random_seed(7)
    bundle = build_data_pipeline_from_arrays(_arrays())
    loaders = create_data_loaders(bundle, batch_size=3, seed=7)
    config = _config(tmp_path / "run")
    model = DataDrivenNN.from_scalers(
        bundle.scalers,
        feature_dim=8,
        surrogate_hidden_dims=config.model.surrogate_hidden_dims,
    )
    initial = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    result = fit_baseline(
        model=model,
        loaders=loaders,
        scalers=bundle.scalers,
        experiment_config=config,
        device=torch.device("cpu"),
    )
    assert result.checkpoint_path.is_file()
    assert all(record.train_physics == 0.0 for record in result.history)
    assert all(record.validation_physics == 0.0 for record in result.history)
    assert all(np.isfinite(record.train_data) for record in result.history)
    assert any(
        not torch.equal(initial[name], current)
        for name, current in model.state_dict().items()
    )

    restored = DataDrivenNN.from_scalers(
        bundle.scalers, feature_dim=8, surrogate_hidden_dims=(8, 8)
    )
    checkpoint = load_checkpoint(result.checkpoint_path, model=restored)
    assert checkpoint["epoch"] == result.best_epoch
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_baseline_evaluation_uses_common_schema_without_physics():
    bundle = build_data_pipeline_from_arrays(_arrays())
    loaders = create_data_loaders(bundle, batch_size=2, seed=7)
    model = DataDrivenNN.from_scalers(
        bundle.scalers, feature_dim=8, surrogate_hidden_dims=(8, 8)
    )
    metrics, columns = evaluate_baseline(
        model=model, loader=loaders.test, device=torch.device("cpu")
    )
    assert tuple(columns) == CORE_PREDICTION_COLUMNS
    assert metrics["records"] == len(bundle.test)
    assert metrics["batteries"] == 1
    assert all(np.all(np.isfinite(values)) for values in columns.values())


def test_baseline_aggregation_uses_sample_standard_deviation():
    results = []
    for seed, rmse in zip(FIXED_SEEDS, (0.01, 0.02, 0.03), strict=True):
        metrics = {
            "records": 10,
            "batteries": 2,
            "mae": rmse / 2,
            "rmse": rmse,
            "r2": 0.9,
            "bias": 0.0,
            "signed_error_p05": -0.1,
            "signed_error_p50": 0.0,
            "signed_error_p95": 0.1,
            "absolute_error_p50": 0.01,
            "absolute_error_p90": 0.02,
            "absolute_error_p95": 0.03,
            "absolute_error_p99": 0.04,
        }
        results.append(BaselineRunResult(seed, 2, 3, True, metrics))
    aggregate = aggregate_baseline_results(results)
    assert aggregate["rmse"]["mean"] == 0.02
    assert aggregate["rmse"]["std"] == 0.01
