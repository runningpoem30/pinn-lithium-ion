from pathlib import Path

from src.tune_physics import ValidationMetrics, load_tuning_config, select_best_result


def metric(weight: float, rmse: float, residual_rmse: float) -> ValidationMetrics:
    return ValidationMetrics(
        physics_weight=weight,
        best_epoch=1,
        epochs_completed=1,
        stopped_early=False,
        records=10,
        soh_rmse=rmse,
        soh_r2=0.5,
        physical_residual_mae=residual_rmse / 2,
        physical_residual_rmse=residual_rmse,
        normalized_data_loss=1.0,
        normalized_physics_loss=1.0,
        weighted_physics_loss=weight,
        normalized_total_loss=1.0 + weight,
    )


def test_tuning_configuration_uses_required_weights_and_validation_monitor():
    config = load_tuning_config(Path("configs/deephpm_soh_tuning.yaml"))
    assert config.physics_weights == (0.001, 0.01, 0.1, 1.0)
    assert config.base_experiment.training.early_stopping_monitor == "data"
    assert config.base_experiment.training.max_train_batches is None
    assert config.base_experiment.training.max_validation_batches is None


def test_selection_uses_rmse_then_physical_residual_as_tie_breaker():
    results = [
        metric(0.001, 0.02, 0.003),
        metric(0.01, 0.01, 0.004),
        metric(0.1, 0.01, 0.002),
    ]
    assert select_best_result(results).physics_weight == 0.1
