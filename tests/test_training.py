from pathlib import Path

import numpy as np
import torch

from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig, load_config
from src.data import build_data_pipeline_from_arrays, create_data_loaders
from src.models import DeepHPMNN
from src.train import EarlyStopping, fit, load_checkpoint, set_random_seed


def training_arrays():
    counts = np.array([[6], [5], [4]], dtype=float)
    battery_ids = np.repeat([1, 2, 3], counts.reshape(-1).astype(int))
    cycles = np.concatenate([np.arange(1, count + 1) for count in (6, 5, 4)])
    features = np.column_stack(
        [
            battery_ids * (column + 1) + cycles * (0.1 + column * 0.01)
            for column in range(8)
        ]
    )
    pcl = (0.01 * battery_ids + 0.002 * cycles)[:, None]
    rul = np.concatenate(
        [np.arange(5, -1, -1), np.arange(4, -1, -1), np.arange(3, -1, -1)]
    )
    return {
        "Features_mov_Flt": features,
        "RUL_Flt": rul[:, None].astype(float),
        "PCL_Flt": pcl,
        "Cycles_Flt": cycles[:, None].astype(float),
        "Num_Cycles_Flt": counts,
        "train_ind": np.array([[1]], dtype=float),
        "test_ind": np.array([[2]], dtype=float),
        "secondary_test_ind": np.array([[3]], dtype=float),
    }


def tiny_config(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(batch_size=3),
        model=ModelConfig(
            surrogate_hidden_dims=(8, 8), dynamics_hidden_dims=(4, 4)
        ),
        training=TrainingConfig(
            seed=123,
            epochs=3,
            learning_rate=1e-3,
            physics_weight=0.1,
            gradient_clip_norm=1.0,
            early_stopping_patience=2,
            max_train_batches=1,
            max_validation_batches=1,
            device="cpu",
        ),
        output_dir=str(output_dir),
    )


def test_yaml_configuration_loads_and_validates():
    config = load_config(Path("configs/deephpm_soh_smoke.yaml"))
    assert config.model.surrogate_hidden_dims == (16, 16)
    assert config.training.max_train_batches == 2
    assert config.training.device == "cpu"


def test_early_stopping_tracks_improvement_and_patience():
    stopping = EarlyStopping(patience=2, min_delta=0.01)
    assert stopping.update(1.0) == (True, False)
    assert stopping.update(0.995) == (False, False)
    assert stopping.update(0.994) == (False, True)
    assert stopping.update(0.8) == (True, False)


def test_training_updates_both_networks_and_checkpoint_reloads(tmp_path: Path):
    set_random_seed(123)
    bundle = build_data_pipeline_from_arrays(training_arrays())
    loaders = create_data_loaders(bundle, batch_size=3, seed=123)
    config = tiny_config(tmp_path / "run")
    model = DeepHPMNN.from_scalers(
        bundle.scalers,
        feature_dim=8,
        surrogate_hidden_dims=config.model.surrogate_hidden_dims,
        dynamics_hidden_dims=config.model.dynamics_hidden_dims,
    )
    initial_surrogate = {
        name: value.detach().clone() for name, value in model.surrogate.state_dict().items()
    }
    initial_dynamics = {
        name: value.detach().clone()
        for name, value in model.dynamics_network.state_dict().items()
    }

    result = fit(
        model=model,
        loaders=loaders,
        scalers=bundle.scalers,
        experiment_config=config,
        device=torch.device("cpu"),
    )

    assert result.checkpoint_path.is_file()
    assert (Path(config.output_dir) / "history.csv").is_file()
    assert (Path(config.output_dir) / "scalers.npz").is_file()
    assert all(
        np.isfinite(value)
        for record in result.history
        for value in (
            record.train_total,
            record.train_data,
            record.train_physics,
            record.validation_total,
            record.validation_data,
            record.validation_physics,
            record.gradient_norm,
        )
    )
    assert any(
        not torch.equal(initial_surrogate[name], value)
        for name, value in model.surrogate.state_dict().items()
    )
    assert any(
        not torch.equal(initial_dynamics[name], value)
        for name, value in model.dynamics_network.state_dict().items()
    )

    restored = DeepHPMNN.from_scalers(
        bundle.scalers,
        feature_dim=8,
        surrogate_hidden_dims=config.model.surrogate_hidden_dims,
        dynamics_hidden_dims=config.model.dynamics_hidden_dims,
    )
    checkpoint = load_checkpoint(result.checkpoint_path, model=restored)
    assert checkpoint["epoch"] == result.best_epoch
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
