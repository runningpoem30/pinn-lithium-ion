from __future__ import annotations

import csv
import json
from pathlib import Path

from src.combined_residuals import generate_combined_residual_analysis


def test_combined_residual_graph_uses_aligned_saved_predictions(tmp_path: Path) -> None:
    png = tmp_path / "combined.png"
    table = tmp_path / "bins.csv"
    report = tmp_path / "summary.json"
    markdown = tmp_path / "analysis.md"
    summary = generate_combined_residual_analysis(
        Path("artifacts/comparisons/soh/aligned_ensemble_predictions.csv"),
        Path("artifacts/comparisons/rul/aligned_ensemble_predictions.csv"),
        png, table, report, markdown,
    )
    assert summary["records"] == 41240
    assert summary["batteries"] == 40
    assert len(summary["global_metrics"]) == 4
    assert png.is_file() and png.stat().st_size > 0
    assert png.with_suffix(".svg").is_file()
    assert png.with_suffix(".pdf").is_file()
    with table.open() as stream:
        assert len(list(csv.DictReader(stream))) == 100  # four series x 25 bins
    assert json.loads(report.read_text())["x_axis"].startswith("elapsed life fraction")
    text = markdown.read_text()
    assert "cannot be directly compared" in text
    assert "physics residual and prediction error" in text
