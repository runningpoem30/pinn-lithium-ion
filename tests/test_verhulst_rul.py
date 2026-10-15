from __future__ import annotations

import numpy as np
import torch

from src.data import DatasetScalers, StandardizationStats
from src.models import VerhulstPINN, verhulst_time_to_threshold
from src.run_verhulst_rul import CrossingConfig, attach_rul_scaler, calculate_losses


def _scalers() -> DatasetScalers:
    return DatasetScalers(
        features=StandardizationStats(np.zeros(8), np.ones(8)),
        cycle=StandardizationStats(np.array([100.0]), np.array([50.0])),
        target_pcl=StandardizationStats(np.array([0.10]), np.array([0.05])),
        target_rul=StandardizationStats(np.array([400.0]), np.array([300.0])),
    )


def test_analytic_crossing_matches_logistic_solution() -> None:
    u = torch.tensor([[0.10]], requires_grad=True)
    r, k, c = torch.tensor([0.005]), torch.tensor([0.50]), torch.tensor([0.01])
    result = verhulst_time_to_threshold(u, r, k, c, threshold=0.20)
    y = (0.10 - 0.01) / (0.50 - 0.01)
    y_eol = (0.20 - 0.01) / (0.50 - 0.01)
    expected = (np.log(y_eol / (1 - y_eol)) - np.log(y / (1 - y))) / 0.005
    assert result.crossing_possible.item()
    assert np.isclose(result.rul.item(), expected, rtol=1e-5)
    result.rul.sum().backward()
    assert torch.isfinite(u.grad).all()


def test_crossing_edge_cases_are_explicit() -> None:
    u = torch.tensor([[0.21], [0.005]])
    result = verhulst_time_to_threshold(
        u, torch.tensor([0.005]), torch.tensor([0.5]), torch.tensor([0.01]),
        non_crossing_horizon=4321.0,
    )
    assert result.threshold_reached[:, 0].tolist() == [True, False]
    assert result.non_crossing[:, 0].tolist() == [False, True]
    assert result.rul[:, 0].tolist() == [0.0, 4321.0]


def test_combined_loss_has_finite_gradients_for_model_and_parameters() -> None:
    scalers = _scalers()
    model = VerhulstPINN.from_scalers(
        scalers, feature_dim=8, surrogate_hidden_dims=(8, 8),
        capacity_bounds=(0.200001, 1.0),
    )
    attach_rul_scaler(model, scalers)
    batch = {
        "features_normalized": torch.randn(6, 8),
        "cycle_raw": torch.linspace(1, 6, 6)[:, None],
        "target_rul_normalized": torch.randn(6, 1),
        "target_pcl_normalized": torch.randn(6, 1),
    }
    losses, output, crossing = calculate_losses(
        model, batch, CrossingConfig(), 0.01, torch.device("cpu"), create_graph=True
    )
    assert output.pcl.shape == crossing.rul.shape == (6, 1)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert model.raw_rate.grad is not None and torch.isfinite(model.raw_rate.grad)
    assert model.raw_capacity.grad is not None and torch.isfinite(model.raw_capacity.grad)
    assert any(parameter.grad is not None for parameter in model.surrogate.parameters())


def test_parameterization_guarantees_threshold_reachable_asymptote() -> None:
    scalers = _scalers()
    model = VerhulstPINN.from_scalers(
        scalers, feature_dim=8, surrogate_hidden_dims=(4,),
        capacity_bounds=(0.200001, 1.0), offset_bounds=(0.0, 0.1),
    )
    for raw in (-100.0, 0.0, 100.0):
        with torch.no_grad():
            model.raw_capacity.fill_(raw)
        assert model.carrying_capacity.item() > 0.2
