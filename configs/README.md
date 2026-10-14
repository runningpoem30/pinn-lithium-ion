# Experiment configurations

`deephpm_soh_smoke.yaml` is the bounded Step 4 CPU smoke configuration. Its
batch limits ensure it cannot be mistaken for a full experiment.

`deephpm_soh_tuning.yaml` defines the Step 5 validation-only physics-weight
sweep. It uses every official training and validation record but never evaluates
the final test batteries.

`deephpm_soh_final.yaml` defines the Step 6 final three-seed CPU experiment. It
freezes the validation-selected physics weight at `0.001`, uses every training
and validation record, selects checkpoints only by validation data loss, and
evaluates the untouched secondary-test batteries once per seed after training.

`data_driven_soh_final.yaml` defines the Step 9 data-only SoH ablation. It uses
the same surrogate architecture, normalized inputs, raw cycle input, optimizer,
batching, early stopping, complete splits, and fixed seeds as the final DeepHPM
experiment, but sets the physics weight to zero and instantiates no dynamics
network.

`verhulst_soh_tuning.yaml` defines the Step 11 validation-only VerhulstPINN
physics-weight sweep. It uses the same candidates and selection rule as the
DeepHPM sweep. The final Verhulst configuration is frozen only after this sweep
has selected its weight.

`verhulst_soh_final.yaml` freezes the validation-selected Verhulst physics
weight at `0.01` and runs the protocol seeds with the complete training and
validation splits. Its loader verifies the weight against the saved tuning
report before final training is allowed.
