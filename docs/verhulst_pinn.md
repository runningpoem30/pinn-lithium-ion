# Clean VerhulstPINN formulation

## State and sign convention

The surrogate uses the same convention as the completed DeepHPM model:

```text
U(features, cycle) = PCL = 1 - SoH
```

`U` is therefore a dimensionless capacity-loss fraction. Battery degradation
normally makes PCL increase with cycle count, so `dU/dcycle` is positive while
`dSoH/dcycle = -dU/dcycle` is negative. Model predictions are converted back to
SoH only for reporting with `predicted_soh = 1 - predicted_pcl`.

The surrogate input is the eight features standardized using training-battery
statistics plus raw cycle standardized inside the autograd graph. It produces
normalized PCL, which is inverse-transformed before the physical equation is
evaluated.

## Verhulst equation and residual

The migrated formulation preserves the shifted Verhulst law used by the legacy
project:

```text
dU/dcycle = r (U - C) [1 - (U - C)/(K - C)]
F = dU/dcycle - r (U - C) [1 - (U - C)/(K - C)]
```

Here `r` is the degradation rate, `C` is the lower PCL offset, and `K` is the
PCL carrying capacity. Between `C` and `K`, the right-hand side is positive;
therefore the law describes increasing PCL and decreasing SoH.

The parameter units are:

- `U`, `C`, and `K`: dimensionless PCL fractions.
- `r`: inverse cycles.
- `dU/dcycle`, Verhulst dynamics, and `F`: PCL per physical cycle.

For loss calculation, `F` is divided by `target_pcl_scale / cycle_scale`. This
produces the normalized residual used by the same data-plus-physics loss as
DeepHPM while retaining the physical residual for reporting.

## Trainable parameter transformations

The optimizer updates unconstrained raw scalars. Differentiable transforms map
them into the physical domain:

```text
r = softplus(raw_rate) + epsilon             r > 0
C = C_min + (C_max - C_min) sigmoid(raw_C)  C in [0.0, 0.1]
K = K_min + (K_max - K_min) sigmoid(raw_K)  K in [0.2, 1.0]
```

The default `C` and `K` intervals are disjoint, which guarantees `K > C` and
prevents division by zero in the shifted logistic term. These bounds preserve
the physical ranges from the legacy implementation. They are configuration
arguments on `VerhulstPINN`, not values inferred from validation or test data.

## Step 10 scope

Step 10 implements the model and its verification tests only. It does not tune
the physics-loss weight, train the full three-seed experiment, inspect final
test performance, or produce model-comparison claims.
