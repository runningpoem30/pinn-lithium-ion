# DeepHPM RUL formulation

## Target and architecture

The Step 13 audit defines the supervised target as

`R(cycle) = cycle_life - cycle`,

measured in physical cycles. DeepHPM therefore predicts RUL directly rather
than reusing the PCL/SoH output convention or the Verhulst threshold-crossing
calculation.

The surrogate is

`R_hat = U(eight normalized features, raw cycle)`.

Raw cycle is standardized inside the graph. Autograd calculates
`partial R_hat / partial cycle` with respect to physical raw cycle while the
eight features in the current row are held fixed. This distinction matters:
the features themselves change between measured cycles, but the derivative is
the partial derivative of the neural surrogate's explicit cycle coordinate.

The learned dynamics network receives normalized cycle and normalized
predicted RUL:

`G = G(cycle, R_hat)`.

Its normalized output is converted to physical **cycles per cycle**. The
DeepHPM residual is

`F = partial R_hat / partial cycle - G(cycle, R_hat)`,

also in cycles per cycle. The physics loss is mean squared normalized `F`.
The supervised loss is training-scaler-normalized RUL MSE.

## Scientific interpretation

For the audited label, an ideal complete battery trajectory satisfies
`dR/dcycle = -1`. DeepHPM does not hard-code `-1`; it learns `G` so this remains
a learned hidden-physics experiment. Test artifacts report both the learned
derivative and `G`, their residual, and each quantity's deviation from `-1`.

This architecture is scientifically weaker than enforcing the known `-1`
identity as a fixed conservation law, because `dR-G=0` alone can be satisfied
by many mutually consistent functions. The RUL data loss anchors the state,
and validation determines whether the learned-dynamics regularizer improves
generalization. Results must therefore be judged by held-out RUL error, not by
a small residual alone.

## Experimental protocol

The eight features, battery-wise official splits, training-only scalers,
network widths, optimizer, batch size, and seeds match the other RUL models.
Physics weights `0.001`, `0.01`, `0.1`, and `1.0` are selected using validation
battery-macro RUL RMSE, with residual RMSE only as a tie-breaker. Test batteries
are evaluated once per seed after restoring the validation-selected checkpoint.
