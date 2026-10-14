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
