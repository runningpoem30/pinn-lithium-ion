# Reproducibility commands

These commands regenerate reports from saved predictions only; they do not train or select models.

```bash
PYTHONPATH=. .venv/bin/python -m src.compare_soh
PYTHONPATH=. .venv/bin/python -m src.compare_rul
PYTHONPATH=. .venv/bin/python -m src.package_results
PYTHONPATH=. .venv/bin/python -m pytest -q
```

Frozen experimental controls: official battery-wise splits, eight features,
training-only scalers, seeds 1234/2345/3456, and validation-only checkpoint and
physics-weight selection. Test predictions are consumed only for final descriptive evaluation.
