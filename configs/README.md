# Experiment configurations

`deephpm_soh_smoke.yaml` is the bounded Step 4 CPU smoke configuration. Its
batch limits ensure it cannot be mistaken for a full experiment.

Full training and tuning configurations will be added in later steps.

`deephpm_soh_tuning.yaml` defines the Step 5 validation-only physics-weight
sweep. It uses every official training and validation record but never evaluates
the final test batteries.
