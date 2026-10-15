# Verhulst RUL method

## Scientific definition

Step 13 established the cycle-level label `RUL_Flt = cycle_life - cycle`, in
physical cycles. The source pipeline intended end of life to mean discharge
capacity below 0.88 Ah, equivalent to SoH below 0.80 for the documented
1.1 Ah nominal capacity. The operational labels use the supplied
`cycle_life` metadata because the legacy MATLAB code did not persist its
threshold assignment.

The Verhulst model therefore does **not** fit an arbitrary logistic curve to
RUL. It predicts the degradation state

`U = PCL = 1 - SoH`

and applies the shifted Verhulst law

`dU/dcycle = r (U-C) [1 - (U-C)/(K-C)]`,

where `r > 0`, `0 <= C < 0.1`, and `0.200001 < K < 1`. The physics residual
has units PCL/cycle. The end-of-life state is `U_eol = 0.20`.

For `C < U < U_eol < K`, let `y=(U-C)/(K-C)`. The remaining time is the
analytic logistic crossing time

`RUL_hat = [logit(y_eol) - logit(y)] / r` cycles.

If `U >= U_eol`, predicted RUL is zero. If `U <= C` or `K <= U_eol`, the
learned trajectory has no finite future crossing. Such records are explicitly
flagged as `non_crossing`; a declared 5000-cycle reporting horizon is stored
instead of silently emitting NaN or infinity. Their count and fraction are
reported in every evaluation. The capacity parameter lower bound makes the
`K <= U_eol` case structurally impossible, while a training penalty and PCL
data loss discourage `U <= C`.

## Model fitting and selection

The total training objective is normalized RUL MSE plus PCL data MSE, weighted
Verhulst residual MSE, and a non-crossing-state penalty. All scalers are fitted
on training batteries only. Physics weights 0.001, 0.01, 0.1, and 1.0 are
tuned with seed 1234. Selection uses the lowest validation battery-macro RUL
RMSE; validation residual RMSE breaks ties. Test batteries are untouched until
the selected configuration is trained with seeds 1234, 2345, and 3456.

Checkpoints are selected using validation battery-macro RUL RMSE in cycles.
Every test prediction retains battery ID, cycle, EOL cycle, target and predicted
RUL, PCL state, derivative, Verhulst RHS, physics residual, and crossing masks.
