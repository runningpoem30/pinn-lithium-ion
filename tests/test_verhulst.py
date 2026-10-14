import numpy as np
import pytest
import torch
from torch import nn

from src.data import DatasetScalers, StandardizationStats
from src.losses import DeepHPMLoss
from src.models import DeepHPMNN, VerhulstPINN, verhulst_rhs


def _scalers() -> DatasetScalers:
    return DatasetScalers(
        features=StandardizationStats(mean=np.zeros(8), scale=np.ones(8)),
        cycle=StandardizationStats(
            mean=np.array([100.0]), scale=np.array([50.0])
        ),
        target_pcl=StandardizationStats(
            mean=np.array([0.05]), scale=np.array([0.02])
        ),
    )


def _model() -> VerhulstPINN:
    torch.manual_seed(1234)
    return VerhulstPINN.from_scalers(
        _scalers(),
        feature_dim=8,
        surrogate_hidden_dims=(16, 16),
        initial_rate=0.005,
        initial_capacity=0.5,
        initial_offset=0.01,
    )


def test_verhulst_output_shapes_and_finite_values():
    output = _model()(
        torch.randn(7, 8), torch.linspace(1.0, 700.0, 7).reshape(-1, 1)
    )

    for name, value in vars(output).items():
        assert value.shape == (7, 1), name
        assert torch.isfinite(value).all(), name
    assert output.cycle_raw.requires_grad


def test_surrogate_matches_deephpm_input_output_convention():
    torch.manual_seed(99)
    verhulst = VerhulstPINN.from_scalers(
        _scalers(), feature_dim=8, surrogate_hidden_dims=(8, 8)
    )
    torch.manual_seed(99)
    deephpm = DeepHPMNN.from_scalers(
        _scalers(),
        feature_dim=8,
        surrogate_hidden_dims=(8, 8),
        dynamics_hidden_dims=(4,),
    )
    for name, value in verhulst.surrogate.state_dict().items():
        assert torch.equal(value, deephpm.surrogate.state_dict()[name])

    features = torch.randn(5, 8)
    cycles = torch.linspace(1.0, 100.0, 5).reshape(-1, 1)
    assert torch.allclose(verhulst(features, cycles).pcl, deephpm(features, cycles).pcl)


def test_verhulst_residual_matches_documented_physical_equation():
    model = _model()
    output = model(
        torch.randn(9, 8), torch.linspace(10.0, 500.0, 9).reshape(-1, 1)
    )
    expected_dynamics = verhulst_rhs(
        output.pcl, model.rate, model.carrying_capacity, model.offset
    )

    assert torch.allclose(output.dynamics, expected_dynamics)
    assert torch.allclose(
        output.residual, output.time_derivative - expected_dynamics
    )
    assert torch.allclose(
        output.residual_normalized,
        output.residual / (model.target_scale / model.cycle_scale),
    )


def test_verhulst_rhs_has_positive_pcl_growth_between_offset_and_capacity():
    rate = torch.tensor(0.2)
    offset = torch.tensor(0.1)
    capacity = torch.tensor(0.9)
    states = torch.stack((offset, (offset + capacity) / 2, capacity))
    values = verhulst_rhs(states, rate, capacity, offset)

    assert torch.allclose(values[[0, 2]], torch.zeros(2))
    assert torch.allclose(values[1], rate * (capacity - offset) / 4)
    assert values[1] > 0  # PCL rises, so SoH = 1 - PCL falls.


def test_parameter_transforms_always_obey_physical_constraints():
    model = _model()
    with torch.no_grad():
        for raw_value in (-100.0, 100.0):
            model.raw_rate.fill_(raw_value)
            model.raw_offset.fill_(raw_value)
            model.raw_capacity.fill_(-raw_value)

            assert model.rate.item() > 0.0
            assert model.offset_lower.item() <= model.offset.item()
            assert model.offset.item() <= model.offset_upper.item()
            assert model.capacity_lower.item() <= model.carrying_capacity.item()
            assert model.carrying_capacity.item() <= model.capacity_upper.item()
            assert model.carrying_capacity.item() > model.offset.item()


def test_backpropagation_reaches_surrogate_and_verhulst_parameters():
    model = _model()
    output = model(
        torch.randn(11, 8), torch.linspace(10.0, 500.0, 11).reshape(-1, 1)
    )
    losses = DeepHPMLoss(physics_weight=0.1)(output, torch.randn(11, 1))
    losses.total.backward()

    assert torch.isfinite(losses.total)
    surrogate_gradients = [parameter.grad for parameter in model.surrogate.parameters()]
    physics_gradients = [
        model.raw_rate.grad,
        model.raw_capacity.grad,
        model.raw_offset.grad,
    ]
    for gradient in surrogate_gradients + physics_gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
    assert sum(gradient.abs().sum().item() for gradient in physics_gradients) > 0


def test_verhulst_model_has_no_dropout_and_supports_derivative_evaluation():
    model = _model().eval()
    assert not any(isinstance(module, nn.Dropout) for module in model.modules())
    with torch.no_grad():
        output = model(
            torch.randn(3, 8),
            torch.tensor([[1.0], [2.0], [3.0]]),
            create_graph=False,
        )
    assert torch.isfinite(output.time_derivative).all()
    assert torch.isfinite(output.residual).all()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"initial_rate": 0.0}, "initial_rate"),
        ({"initial_offset": 0.2}, "initial_offset"),
        ({"initial_capacity": 1.0}, "initial_capacity"),
        ({"capacity_bounds": (1.0, 0.2)}, "strictly increasing"),
        ({"offset_bounds": (0.1, 0.0)}, "strictly increasing"),
        ({"offset_bounds": (-0.1, 0.1)}, "non-negative"),
        (
            {"offset_bounds": (0.0, 0.3), "capacity_bounds": (0.2, 1.0)},
            "below capacity",
        ),
    ],
)
def test_invalid_verhulst_parameterization_is_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        VerhulstPINN.from_scalers(_scalers(), feature_dim=8, **options)


def test_invalid_input_shapes_are_rejected():
    model = _model()
    with pytest.raises(ValueError, match="Expected 8 features"):
        model(torch.randn(4, 7), torch.randn(4, 1))
    with pytest.raises(ValueError, match="cycle_raw"):
        model(torch.randn(4, 8), torch.randn(4))
