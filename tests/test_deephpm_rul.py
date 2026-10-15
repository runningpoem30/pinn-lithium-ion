from __future__ import annotations

import numpy as np
import torch

from src.data import DatasetScalers, StandardizationStats
from src.models import DeepHPMRULNN
from src.run_deephpm_rul import _losses, evaluate
from torch.utils.data import DataLoader
from src.data import BatteryDataset


def scalers() -> DatasetScalers:
    return DatasetScalers(
        features=StandardizationStats(np.zeros(8), np.ones(8)),
        cycle=StandardizationStats(np.array([100.0]), np.array([50.0])),
        target_pcl=StandardizationStats(np.array([0.1]), np.array([0.05])),
        target_rul=StandardizationStats(np.array([400.0]), np.array([300.0])),
    )


def model() -> DeepHPMRULNN:
    return DeepHPMRULNN.from_scalers(
        scalers(), feature_dim=8, surrogate_hidden_dims=(12, 12),
        dynamics_hidden_dims=(8, 8),
    )


def test_shapes_units_and_finite_values() -> None:
    output = model()(torch.randn(7, 8), torch.linspace(1, 700, 7)[:, None])
    for value in vars(output).values():
        assert value.shape == (7, 1)
        assert torch.isfinite(value).all()
    assert torch.allclose(
        output.residual, output.time_derivative - output.dynamics
    )


def test_backpropagation_updates_surrogate_and_dynamics() -> None:
    net = model()
    output = net(torch.randn(11, 8), torch.linspace(1, 500, 11)[:, None])
    total, data, physics = _losses(output, torch.randn(11, 1), 0.1)
    assert torch.isfinite(total) and torch.isfinite(data) and torch.isfinite(physics)
    total.backward()
    for network in (net.surrogate, net.dynamics_network):
        gradients = [parameter.grad for parameter in network.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_raw_cycle_autograd_uses_physical_cycle_units() -> None:
    net = model()
    cycles = torch.tensor([[10.0], [20.0]], requires_grad=True)
    output = net(torch.zeros(2, 8), cycles)
    independently_computed = torch.autograd.grad(
        output.rul, cycles, torch.ones_like(output.rul), create_graph=True
    )[0]
    assert torch.allclose(output.time_derivative, independently_computed)
    assert torch.allclose(
        output.residual_normalized,
        output.residual / (net.target_scale / net.cycle_scale),
    )


def test_evaluation_preserves_ids_rul_and_physics_columns() -> None:
    count = 6
    features = np.zeros((count, 8)); cycles = np.arange(1, count + 1)[:, None]
    rul = np.arange(10, 10 - count, -1)[:, None].astype(float)
    pcl = np.linspace(0.01, 0.06, count)[:, None]
    dataset = BatteryDataset(
        features_raw=features, features_normalized=features,
        cycle_raw=cycles, cycle_normalized=cycles,
        target_pcl=pcl, target_pcl_normalized=pcl,
        target_rul=rul, target_rul_normalized=(rul - 400) / 300,
        eol_cycle=cycles + rul, battery_id=np.array([[1], [1], [2], [2], [3], [3]]),
    )
    metrics, columns, _, losses = evaluate(
        model(), DataLoader(dataset, batch_size=2), torch.device("cpu"), 0.01
    )
    assert metrics["records"] == count
    assert columns["battery_id"].tolist() == [1, 1, 2, 2, 3, 3]
    assert set(("time_derivative_rul_per_cycle", "learned_dynamics_rul_per_cycle", "physics_residual_rul_per_cycle")) <= set(columns)
    assert all(np.isfinite(value).all() for value in columns.values())
    assert all(np.isfinite(value) for value in losses.values())
