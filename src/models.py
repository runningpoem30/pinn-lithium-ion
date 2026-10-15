"""Clean data-driven, Verhulst PINN, and DeepHPM model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.data import DatasetScalers


@dataclass(frozen=True)
class DeepHPMOutput:
    """Predictions and learned dynamics returned by :class:`DeepHPMNN`."""

    pcl: torch.Tensor
    pcl_normalized: torch.Tensor
    cycle_raw: torch.Tensor
    cycle_normalized: torch.Tensor
    time_derivative: torch.Tensor
    dynamics: torch.Tensor
    residual: torch.Tensor
    residual_normalized: torch.Tensor


@dataclass(frozen=True)
class VerhulstOutput:
    """Surrogate prediction and shifted-Verhulst physics quantities.

    Every cycle-level field is in physical units except the two fields whose
    names end in ``_normalized``. ``pcl`` is the dimensionless fraction
    ``U = 1 - SoH``; derivatives, dynamics, and residuals therefore have units
    of PCL per physical cycle.
    """

    pcl: torch.Tensor
    pcl_normalized: torch.Tensor
    cycle_raw: torch.Tensor
    cycle_normalized: torch.Tensor
    time_derivative: torch.Tensor
    dynamics: torch.Tensor
    residual: torch.Tensor
    residual_normalized: torch.Tensor


@dataclass(frozen=True)
class DataDrivenOutput:
    """SoH baseline output with no learned physics quantities."""

    pcl: torch.Tensor
    pcl_normalized: torch.Tensor
    cycle_raw: torch.Tensor
    cycle_normalized: torch.Tensor


@dataclass(frozen=True)
class RULDataDrivenOutput:
    """DataDrivenNN output for remaining useful life in physical cycles."""

    rul: torch.Tensor
    rul_normalized: torch.Tensor
    cycle_raw: torch.Tensor
    cycle_normalized: torch.Tensor


class MLP(nn.Module):
    """A deterministic fully connected network with smooth activations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for width in hidden_dims:
            linear = nn.Linear(previous_dim, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.Tanh()))
            previous_dim = width

        output = nn.Linear(previous_dim, output_dim)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DataDrivenNN(nn.Module):
    """Data-only ablation of the DeepHPM surrogate network."""

    def __init__(
        self,
        *,
        feature_dim: int,
        cycle_mean: float,
        cycle_scale: float,
        target_mean: float,
        target_scale: float,
        surrogate_hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if cycle_scale <= 0 or target_scale <= 0:
            raise ValueError("cycle_scale and target_scale must be positive")
        self.feature_dim = feature_dim
        self.surrogate = MLP(
            input_dim=feature_dim + 1,
            hidden_dims=surrogate_hidden_dims,
            output_dim=1,
        )
        self.register_buffer(
            "cycle_mean", torch.tensor([cycle_mean], dtype=torch.float32)
        )
        self.register_buffer(
            "cycle_scale", torch.tensor([cycle_scale], dtype=torch.float32)
        )
        self.register_buffer(
            "target_mean", torch.tensor([target_mean], dtype=torch.float32)
        )
        self.register_buffer(
            "target_scale", torch.tensor([target_scale], dtype=torch.float32)
        )

    @classmethod
    def from_scalers(
        cls,
        scalers: DatasetScalers,
        *,
        feature_dim: int,
        surrogate_hidden_dims: Sequence[int] = (128, 128),
    ) -> "DataDrivenNN":
        return cls(
            feature_dim=feature_dim,
            cycle_mean=float(scalers.cycle.mean[0]),
            cycle_scale=float(scalers.cycle.scale[0]),
            target_mean=float(scalers.target_pcl.mean[0]),
            target_scale=float(scalers.target_pcl.scale[0]),
            surrogate_hidden_dims=surrogate_hidden_dims,
        )

    def forward(
        self, features_normalized: torch.Tensor, cycle_raw: torch.Tensor
    ) -> DataDrivenOutput:
        if features_normalized.ndim != 2:
            raise ValueError("features_normalized must have shape [batch, features]")
        if cycle_raw.ndim != 2 or cycle_raw.shape[1] != 1:
            raise ValueError("cycle_raw must have shape [batch, 1]")
        if features_normalized.shape != (cycle_raw.shape[0], self.feature_dim):
            raise ValueError("feature shape does not match cycle batch or feature_dim")
        cycle_normalized = (cycle_raw - self.cycle_mean) / self.cycle_scale
        pcl_normalized = self.surrogate(
            torch.cat((features_normalized, cycle_normalized), dim=-1)
        )
        pcl = pcl_normalized * self.target_scale + self.target_mean
        return DataDrivenOutput(
            pcl=pcl,
            pcl_normalized=pcl_normalized,
            cycle_raw=cycle_raw,
            cycle_normalized=cycle_normalized,
        )


class DataDrivenRULNN(nn.Module):
    """Data-only RUL baseline using the same surrogate convention as SoH."""

    def __init__(
        self,
        *,
        feature_dim: int,
        cycle_mean: float,
        cycle_scale: float,
        target_mean: float,
        target_scale: float,
        surrogate_hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if cycle_scale <= 0 or target_scale <= 0:
            raise ValueError("cycle_scale and target_scale must be positive")
        self.feature_dim = feature_dim
        self.surrogate = MLP(
            input_dim=feature_dim + 1,
            hidden_dims=surrogate_hidden_dims,
            output_dim=1,
        )
        self.register_buffer("cycle_mean", torch.tensor([cycle_mean], dtype=torch.float32))
        self.register_buffer("cycle_scale", torch.tensor([cycle_scale], dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor([target_mean], dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor([target_scale], dtype=torch.float32))

    @classmethod
    def from_scalers(
        cls,
        scalers: DatasetScalers,
        *,
        feature_dim: int,
        surrogate_hidden_dims: Sequence[int] = (128, 128),
    ) -> "DataDrivenRULNN":
        if scalers.target_rul is None:
            raise ValueError("DatasetScalers does not contain a training-fitted RUL scaler")
        return cls(
            feature_dim=feature_dim,
            cycle_mean=float(scalers.cycle.mean[0]),
            cycle_scale=float(scalers.cycle.scale[0]),
            target_mean=float(scalers.target_rul.mean[0]),
            target_scale=float(scalers.target_rul.scale[0]),
            surrogate_hidden_dims=surrogate_hidden_dims,
        )

    def forward(
        self, features_normalized: torch.Tensor, cycle_raw: torch.Tensor
    ) -> RULDataDrivenOutput:
        if features_normalized.ndim != 2:
            raise ValueError("features_normalized must have shape [batch, features]")
        if cycle_raw.ndim != 2 or cycle_raw.shape[1] != 1:
            raise ValueError("cycle_raw must have shape [batch, 1]")
        if features_normalized.shape != (cycle_raw.shape[0], self.feature_dim):
            raise ValueError("feature shape does not match cycle batch or feature_dim")
        cycle_normalized = (cycle_raw - self.cycle_mean) / self.cycle_scale
        rul_normalized = self.surrogate(
            torch.cat((features_normalized, cycle_normalized), dim=-1)
        )
        rul = rul_normalized * self.target_scale + self.target_mean
        return RULDataDrivenOutput(
            rul=rul,
            rul_normalized=rul_normalized,
            cycle_raw=cycle_raw,
            cycle_normalized=cycle_normalized,
        )


def time_derivative(
    values: torch.Tensor,
    cycle_raw: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Differentiate pointwise predictions with respect to physical cycle."""

    derivative = torch.autograd.grad(
        outputs=values,
        inputs=cycle_raw,
        grad_outputs=torch.ones_like(values),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    if derivative is None:
        raise RuntimeError("Prediction is not connected to raw cycle values")
    return derivative


def verhulst_rhs(
    pcl: torch.Tensor,
    rate: torch.Tensor,
    carrying_capacity: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the shifted Verhulst PCL growth law.

    The equation is

    ``dU/dcycle = r (U - C) [1 - (U - C)/(K - C)]``,

    where ``U`` is PCL, ``r`` has units cycle\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}, and
    ``C`` and ``K`` are dimensionless PCL fractions. Valid model parameters
    obey ``r > 0`` and ``0 <= C < K``.
    """

    return rate * (pcl - offset) * (
        1.0 - (pcl - offset) / (carrying_capacity - offset)
    )


def _inverse_softplus(value: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(value, dtype=torch.float32)))


def _inverse_logit(fraction: float) -> torch.Tensor:
    return torch.logit(torch.tensor(fraction, dtype=torch.float32))


class VerhulstPINN(nn.Module):
    """Physics-informed surrogate with explicit Verhulst degradation dynamics.

    As in :class:`DeepHPMNN`, the surrogate receives eight training-normalized
    features plus cycle normalized *inside* the autograd graph. It predicts
    normalized PCL, which is converted to physical ``U = PCL = 1 - SoH``.

    With increasing cycle count, degradation corresponds to increasing PCL
    (and decreasing SoH). The model therefore constrains the Verhulst rate
    ``r`` to be positive and uses the residual

    ``F = dU/dcycle - r(U-C)(1-(U-C)/(K-C))``.

    ``raw_rate`` is transformed with softplus, while ``raw_offset`` and
    ``raw_capacity`` pass through sigmoids into fixed PCL bounds. The default
    disjoint bounds ``C in (0, 0.1)`` and ``K in (0.2, 1.0)`` guarantee
    ``K > C`` without clipping trainable parameters. The physical residual is
    PCL per cycle; ``residual_normalized`` divides by
    ``target_scale / cycle_scale`` for use in the shared physics loss.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        cycle_mean: float,
        cycle_scale: float,
        target_mean: float,
        target_scale: float,
        surrogate_hidden_dims: Sequence[int] = (128, 128),
        initial_rate: float = 0.005,
        initial_capacity: float = 0.5,
        initial_offset: float = 0.01,
        capacity_bounds: tuple[float, float] = (0.2, 1.0),
        offset_bounds: tuple[float, float] = (0.0, 0.1),
        rate_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if cycle_scale <= 0 or target_scale <= 0:
            raise ValueError("cycle_scale and target_scale must be positive")
        if rate_epsilon <= 0 or initial_rate <= rate_epsilon:
            raise ValueError("initial_rate must be greater than rate_epsilon > 0")
        capacity_lower, capacity_upper = capacity_bounds
        offset_lower, offset_upper = offset_bounds
        if capacity_lower >= capacity_upper:
            raise ValueError("capacity_bounds must be strictly increasing")
        if offset_lower >= offset_upper:
            raise ValueError("offset_bounds must be strictly increasing")
        if not capacity_lower < initial_capacity < capacity_upper:
            raise ValueError("initial_capacity must be strictly inside capacity_bounds")
        if not offset_lower < initial_offset < offset_upper:
            raise ValueError("initial_offset must be strictly inside offset_bounds")
        if offset_lower < 0:
            raise ValueError("offset_bounds must be non-negative PCL fractions")
        if offset_upper >= capacity_lower:
            raise ValueError("offset upper bound must be below capacity lower bound")

        self.feature_dim = feature_dim
        self.rate_epsilon = float(rate_epsilon)
        self.surrogate = MLP(
            input_dim=feature_dim + 1,
            hidden_dims=surrogate_hidden_dims,
            output_dim=1,
        )
        self.raw_rate = nn.Parameter(
            _inverse_softplus(initial_rate - self.rate_epsilon)
        )
        self.raw_capacity = nn.Parameter(
            _inverse_logit(
                (initial_capacity - capacity_lower)
                / (capacity_upper - capacity_lower)
            )
        )
        self.raw_offset = nn.Parameter(
            _inverse_logit(
                (initial_offset - offset_lower) / (offset_upper - offset_lower)
            )
        )

        for name, value in (
            ("cycle_mean", cycle_mean),
            ("cycle_scale", cycle_scale),
            ("target_mean", target_mean),
            ("target_scale", target_scale),
            ("capacity_lower", capacity_lower),
            ("capacity_upper", capacity_upper),
            ("offset_lower", offset_lower),
            ("offset_upper", offset_upper),
        ):
            self.register_buffer(name, torch.tensor([value], dtype=torch.float32))

    @classmethod
    def from_scalers(
        cls,
        scalers: DatasetScalers,
        *,
        feature_dim: int,
        surrogate_hidden_dims: Sequence[int] = (128, 128),
        **parameter_options: float | tuple[float, float],
    ) -> "VerhulstPINN":
        return cls(
            feature_dim=feature_dim,
            cycle_mean=float(scalers.cycle.mean[0]),
            cycle_scale=float(scalers.cycle.scale[0]),
            target_mean=float(scalers.target_pcl.mean[0]),
            target_scale=float(scalers.target_pcl.scale[0]),
            surrogate_hidden_dims=surrogate_hidden_dims,
            **parameter_options,
        )

    @property
    def rate(self) -> torch.Tensor:
        """Positive Verhulst rate ``r`` in inverse cycles."""

        return F.softplus(self.raw_rate) + self.rate_epsilon

    @property
    def carrying_capacity(self) -> torch.Tensor:
        """Verhulst asymptote ``K`` as a dimensionless PCL fraction."""

        return self.capacity_lower + (
            self.capacity_upper - self.capacity_lower
        ) * torch.sigmoid(self.raw_capacity)

    @property
    def offset(self) -> torch.Tensor:
        """Verhulst shift ``C`` as a dimensionless PCL fraction."""

        return self.offset_lower + (
            self.offset_upper - self.offset_lower
        ) * torch.sigmoid(self.raw_offset)

    @property
    def physical_rate_scale(self) -> torch.Tensor:
        """Conversion from normalized PCL/cycle to physical PCL/cycle."""

        return self.target_scale / self.cycle_scale

    def _validate_inputs(
        self, features_normalized: torch.Tensor, cycle_raw: torch.Tensor
    ) -> None:
        if features_normalized.ndim != 2:
            raise ValueError("features_normalized must have shape [batch, features]")
        if cycle_raw.ndim != 2 or cycle_raw.shape[1] != 1:
            raise ValueError("cycle_raw must have shape [batch, 1]")
        if features_normalized.shape != (cycle_raw.shape[0], self.feature_dim):
            raise ValueError(
                f"Expected {self.feature_dim} features and matching batch size"
            )
        if not features_normalized.is_floating_point() or not cycle_raw.is_floating_point():
            raise TypeError("features and cycle must be floating-point tensors")

    def forward(
        self,
        features_normalized: torch.Tensor,
        cycle_raw: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> VerhulstOutput:
        self._validate_inputs(features_normalized, cycle_raw)
        if not cycle_raw.requires_grad:
            cycle_for_grad = cycle_raw.detach().clone().requires_grad_(True)
        else:
            cycle_for_grad = cycle_raw

        with torch.enable_grad():
            cycle_normalized = (
                cycle_for_grad - self.cycle_mean
            ) / self.cycle_scale
            pcl_normalized = self.surrogate(
                torch.cat((features_normalized, cycle_normalized), dim=-1)
            )
            pcl = pcl_normalized * self.target_scale + self.target_mean
            d_pcl_d_cycle = time_derivative(
                pcl, cycle_for_grad, create_graph=create_graph
            )
            dynamics = verhulst_rhs(
                pcl,
                self.rate,
                self.carrying_capacity,
                self.offset,
            )
            residual = d_pcl_d_cycle - dynamics
            residual_normalized = residual / self.physical_rate_scale

        return VerhulstOutput(
            pcl=pcl,
            pcl_normalized=pcl_normalized,
            cycle_raw=cycle_for_grad,
            cycle_normalized=cycle_normalized,
            time_derivative=d_pcl_d_cycle,
            dynamics=dynamics,
            residual=residual,
            residual_normalized=residual_normalized,
        )


class DeepHPMNN(nn.Module):
    """Deep hidden-physics model for cycle-level capacity loss.

    The surrogate learns ``U(features, cycle)``. The dynamics network learns
    ``G(cycle, U)``. Their physical residual is ``dU/dcycle - G``.

    ``features_normalized`` must contain the eight features standardized with
    training-only statistics. ``cycle_raw`` is normalized inside the graph so
    the returned derivative is with respect to physical cycle number.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        cycle_mean: float,
        cycle_scale: float,
        target_mean: float,
        target_scale: float,
        surrogate_hidden_dims: Sequence[int] = (128, 128),
        dynamics_hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if cycle_scale <= 0 or target_scale <= 0:
            raise ValueError("cycle_scale and target_scale must be positive")

        self.feature_dim = feature_dim
        self.surrogate = MLP(
            input_dim=feature_dim + 1,
            hidden_dims=surrogate_hidden_dims,
            output_dim=1,
        )
        self.dynamics_network = MLP(
            input_dim=2,
            hidden_dims=dynamics_hidden_dims,
            output_dim=1,
        )

        self.register_buffer(
            "cycle_mean", torch.tensor([cycle_mean], dtype=torch.float32)
        )
        self.register_buffer(
            "cycle_scale", torch.tensor([cycle_scale], dtype=torch.float32)
        )
        self.register_buffer(
            "target_mean", torch.tensor([target_mean], dtype=torch.float32)
        )
        self.register_buffer(
            "target_scale", torch.tensor([target_scale], dtype=torch.float32)
        )

    @classmethod
    def from_scalers(
        cls,
        scalers: DatasetScalers,
        *,
        feature_dim: int,
        surrogate_hidden_dims: Sequence[int] = (128, 128),
        dynamics_hidden_dims: Sequence[int] = (64, 64),
    ) -> "DeepHPMNN":
        return cls(
            feature_dim=feature_dim,
            cycle_mean=float(scalers.cycle.mean[0]),
            cycle_scale=float(scalers.cycle.scale[0]),
            target_mean=float(scalers.target_pcl.mean[0]),
            target_scale=float(scalers.target_pcl.scale[0]),
            surrogate_hidden_dims=surrogate_hidden_dims,
            dynamics_hidden_dims=dynamics_hidden_dims,
        )

    @property
    def physical_rate_scale(self) -> torch.Tensor:
        """Conversion from normalized PCL/cycle to physical PCL/cycle."""

        return self.target_scale / self.cycle_scale

    def _validate_inputs(
        self, features_normalized: torch.Tensor, cycle_raw: torch.Tensor
    ) -> None:
        if features_normalized.ndim != 2:
            raise ValueError(
                "features_normalized must have shape [batch, features]"
            )
        if cycle_raw.ndim != 2 or cycle_raw.shape[1] != 1:
            raise ValueError("cycle_raw must have shape [batch, 1]")
        if features_normalized.shape[0] != cycle_raw.shape[0]:
            raise ValueError("features and cycle must have the same batch size")
        if features_normalized.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} features, "
                f"received {features_normalized.shape[1]}"
            )
        if not features_normalized.is_floating_point() or not cycle_raw.is_floating_point():
            raise TypeError("features and cycle must be floating-point tensors")

    def forward(
        self,
        features_normalized: torch.Tensor,
        cycle_raw: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> DeepHPMOutput:
        self._validate_inputs(features_normalized, cycle_raw)

        # DataLoader tensors do not require gradients. Create a dedicated leaf
        # without mutating the caller's batch. If a differentiable cycle tensor
        # is intentionally supplied, preserve its graph.
        if not cycle_raw.requires_grad:
            cycle_for_grad = cycle_raw.detach().clone().requires_grad_(True)
        else:
            cycle_for_grad = cycle_raw

        # Derivatives are required even when callers perform model evaluation
        # inside torch.no_grad().
        with torch.enable_grad():
            cycle_normalized = (
                cycle_for_grad - self.cycle_mean
            ) / self.cycle_scale
            surrogate_inputs = torch.cat(
                (features_normalized, cycle_normalized), dim=-1
            )
            pcl_normalized = self.surrogate(surrogate_inputs)
            pcl = pcl_normalized * self.target_scale + self.target_mean
            d_pcl_d_cycle = time_derivative(
                pcl, cycle_for_grad, create_graph=create_graph
            )

            dynamics_inputs = torch.cat(
                (cycle_normalized, pcl_normalized), dim=-1
            )
            dynamics_normalized = self.dynamics_network(dynamics_inputs)
            dynamics = dynamics_normalized * self.physical_rate_scale
            residual = d_pcl_d_cycle - dynamics
            residual_normalized = residual / self.physical_rate_scale

        return DeepHPMOutput(
            pcl=pcl,
            pcl_normalized=pcl_normalized,
            cycle_raw=cycle_for_grad,
            cycle_normalized=cycle_normalized,
            time_derivative=d_pcl_d_cycle,
            dynamics=dynamics,
            residual=residual,
            residual_normalized=residual_normalized,
        )
