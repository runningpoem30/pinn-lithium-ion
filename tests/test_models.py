import numpy as np
import pytest
import torch
from torch import nn

from src.data import DatasetScalers, StandardizationStats
from src.losses import DeepHPMLoss
from src.models import DeepHPMNN, time_derivative


def make_scalers() -> DatasetScalers:
    return DatasetScalers(
        features=StandardizationStats(
            mean=np.zeros(8), scale=np.ones(8)
        ),
        cycle=StandardizationStats(
            mean=np.array([100.0]), scale=np.array([50.0])
        ),
        target_pcl=StandardizationStats(
            mean=np.array([0.05]), scale=np.array([0.02])
        ),
    )


def make_model() -> DeepHPMNN:
    torch.manual_seed(1234)
    return DeepHPMNN.from_scalers(
        make_scalers(),
        feature_dim=8,
        surrogate_hidden_dims=(16, 16),
        dynamics_hidden_dims=(8, 8),
    )


def test_deephpm_output_shapes_and_values_are_finite():
    model = make_model()
    features = torch.randn(7, 8)
    cycles = torch.linspace(1.0, 700.0, 7).reshape(-1, 1)
    output = model(features, cycles)

    for name, value in vars(output).items():
        assert value.shape == (7, 1), name
        assert torch.isfinite(value).all(), name
    assert output.cycle_raw.requires_grad


def test_physical_cycle_derivative_matches_analytic_result():
    cycle = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    values = 3.0 * cycle**2 + 2.0 * cycle
    derivative = time_derivative(values, cycle, create_graph=True)
    assert torch.allclose(derivative, 6.0 * cycle + 2.0)


def test_loss_uses_means_and_configured_physics_weight():
    model = make_model()
    output = model(torch.randn(5, 8), torch.arange(1.0, 6.0)[:, None])
    target = torch.zeros(5, 1)
    criterion = DeepHPMLoss(physics_weight=0.25)
    losses = criterion(output, target)
    expected_data = torch.mean(output.pcl_normalized**2)
    expected_physics = torch.mean(output.residual_normalized**2)
    assert torch.allclose(losses.data, expected_data)
    assert torch.allclose(losses.physics, expected_physics)
    assert torch.allclose(losses.total, expected_data + 0.25 * expected_physics)


def test_backpropagation_reaches_both_networks_with_finite_gradients():
    model = make_model()
    features = torch.randn(11, 8)
    cycles = torch.linspace(10.0, 500.0, 11).reshape(-1, 1)
    targets = torch.randn(11, 1)
    output = model(features, cycles)
    losses = DeepHPMLoss(physics_weight=0.1)(output, targets)
    losses.total.backward()

    for network in (model.surrogate, model.dynamics_network):
        gradients = [
            parameter.grad
            for parameter in network.parameters()
            if parameter.requires_grad
        ]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_model_has_no_dropout_layers():
    model = make_model()
    assert not any(isinstance(module, nn.Dropout) for module in model.modules())


def test_model_can_compute_derivatives_during_no_grad_evaluation():
    model = make_model().eval()
    with torch.no_grad():
        output = model(
            torch.randn(3, 8),
            torch.tensor([[1.0], [2.0], [3.0]]),
            create_graph=False,
        )
    assert output.time_derivative.shape == (3, 1)
    assert torch.isfinite(output.residual).all()


def test_invalid_shapes_and_loss_weight_are_rejected():
    model = make_model()
    with pytest.raises(ValueError, match="Expected 8 features"):
        model(torch.randn(4, 7), torch.randn(4, 1))
    with pytest.raises(ValueError, match="cycle_raw"):
        model(torch.randn(4, 8), torch.randn(4))
    with pytest.raises(ValueError, match="non-negative"):
        DeepHPMLoss(physics_weight=-1.0)
