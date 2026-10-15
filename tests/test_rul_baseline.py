from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.data import build_data_pipeline_from_arrays, create_data_loaders
from src.models import DataDrivenRULNN
from src.protocol import FIXED_SEEDS
from src.run_rul_baseline import (
    calculate_rul_metrics,
    evaluate_rul_baseline,
    fit_rul_baseline,
    load_rul_baseline_config,
    validate_rul_predictions,
)
from src.train import set_random_seed


def _arrays() -> dict[str, np.ndarray]:
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
        model=ModelConfig(surrogate_hidden_dims=(8, 8), dynamics_hidden_dims=(4,)),
        training=TrainingConfig(
            seed=7,
            epochs=3,
            learning_rate=0.001,
            physics_weight=0.0,
            gradient_clip_norm=1.0,
            early_stopping_patience=2,
            early_stopping_min_delta=0.0,
            max_train_batches=1,
            max_validation_batches=1,
            device="cpu",
        ),
        output_dir=str(output_dir),
    )


def test_final_rul_config_uses_fixed_seeds_and_complete_splits():
    config = load_rul_baseline_config(Path("configs/data_driven_rul_final.yaml"))
    assert config.seeds == FIXED_SEEDS
    assert config.base_experiment.model.surrogate_hidden_dims == (64, 64)
    assert config.base_experiment.training.physics_weight == 0.0
    assert config.base_experiment.training.max_train_batches is None
    assert config.base_experiment.training.max_validation_batches is None


def test_rul_model_uses_training_scaler_and_backpropagates():
    bundle = build_data_pipeline_from_arrays(_arrays())
    model = DataDrivenRULNN.from_scalers(
        bundle.scalers, feature_dim=8, surrogate_hidden_dims=(8, 8)
    )
    batch = next(iter(create_data_loaders(bundle, batch_size=3).train))
    output = model(batch["features_normalized"], batch["cycle_raw"])
    loss = torch.mean(
        torch.square(output.rul_normalized - batch["target_rul_normalized"])
    )
    loss.backward()
    assert output.rul.shape == (3, 1)
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_rul_smoke_training_selects_validation_checkpoint(tmp_path: Path):
    set_random_seed(7)
    bundle = build_data_pipeline_from_arrays(_arrays())
    loaders = create_data_loaders(bundle, batch_size=3, seed=7)
    model = DataDrivenRULNN.from_scalers(
        bundle.scalers, feature_dim=8, surrogate_hidden_dims=(8, 8)
    )
    result = fit_rul_baseline(
        model=model,
        loaders=loaders,
        scalers=bundle.scalers,
        experiment_config=_config(tmp_path / "run"),
        device=torch.device("cpu"),
    )
    assert result.checkpoint_path.is_file()
    assert result.best_epoch >= 1
    assert np.isfinite(result.best_validation_loss)
    assert (tmp_path / "run" / "scalers.npz").is_file()
    assert (tmp_path / "run" / "history.csv").is_file()


def test_rul_evaluation_preserves_targets_ids_cycles_and_units():
    bundle = build_data_pipeline_from_arrays(_arrays())
    loader = create_data_loaders(bundle, batch_size=2).test
    model = DataDrivenRULNN.from_scalers(
        bundle.scalers, feature_dim=8, surrogate_hidden_dims=(8,)
    )
    metrics, columns, per_battery, normalized_mse = evaluate_rul_baseline(
        model=model, loader=loader, device=torch.device("cpu")
    )
    assert columns["battery_id"].tolist() == [3, 3, 3, 3]
    assert columns["cycle"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert columns["eol_cycle"].tolist() == [4.0, 4.0, 4.0, 4.0]
    assert columns["target_rul_cycles"].tolist() == [3.0, 2.0, 1.0, 0.0]
    assert np.allclose(
        columns["rul_error_cycles"],
        columns["predicted_rul_cycles"] - columns["target_rul_cycles"],
    )
    assert metrics["records"] == 4
    assert len(per_battery) == 1
    assert np.isfinite(normalized_mse)


def test_battery_macro_rmse_weights_batteries_equally():
    columns = {
        "battery_id": np.asarray([1, 1, 1, 2, 2]),
        "target_rul_cycles": np.asarray([3.0, 2.0, 1.0, 2.0, 1.0]),
        "predicted_rul_cycles": np.asarray([4.0, 3.0, 2.0, 5.0, 4.0]),
    }
    metrics, rows = calculate_rul_metrics(columns)
    assert np.isclose(metrics["battery_macro_rmse"], 2.0)
    assert np.isclose(metrics["battery_macro_mae"], 2.0)
    assert len(rows) == 2


def test_prediction_validation_rejects_changed_target(tmp_path: Path):
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "battery_id,cycle,eol_cycle,target_rul_cycles\n"
        "3,1,4,3\n3,2,4,2\n",
        encoding="utf-8",
    )
    columns = {
        "battery_id": np.asarray([3, 3]),
        "cycle": np.asarray([1.0, 2.0]),
        "eol_cycle": np.asarray([4.0, 4.0]),
        "target_rul_cycles": np.asarray([3.0, 1.0]),
        "predicted_rul_cycles": np.asarray([2.5, 1.5]),
        "rul_error_cycles": np.asarray([-0.5, 0.5]),
    }
    with pytest.raises(ValueError, match="target_rul_cycles"):
        validate_rul_predictions(columns, reference)
