from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.package_results import build_results_package


def test_final_results_package_is_complete_and_non_training(tmp_path: Path) -> None:
    output = tmp_path / "package"
    payload = build_results_package(
        Path("artifacts/comparisons/soh"),
        Path("artifacts/comparisons/rul"),
        output,
    )
    assert payload["retraining_performed"] is False
    assert payload["test_results_used_for_model_selection"] is False
    assert payload["cohort"] == {
        "records": 41240,
        "batteries": 40,
        "seeds": [1234, 2345, 3456],
        "keys_identical_across_tasks": True,
    }
    required = {
        "README.md", "compact_results_table.csv", "rul_life_region_summary.csv",
        "key_findings.csv", "key_findings.md", "limitations.md",
        "reproducibility.md", "figure_captions.csv", "slide_plan.md",
        "results_package.json", "package_manifest.json",
        "combined_physics_residual_analysis.md",
        "combined_physics_residual_bins.csv",
        "combined_physics_residual_summary.json",
    }
    assert required <= {path.name for path in output.iterdir() if path.is_file()}
    with (output / "compact_results_table.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 6
    with (output / "rul_life_region_summary.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 9
    with (output / "figure_captions.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 9
    figures = sorted((output / "figures").glob("*.png"))
    assert len(figures) == 9 and all(path.stat().st_size > 0 for path in figures)


def test_manifest_checksums_every_packaged_file(tmp_path: Path) -> None:
    output = tmp_path / "package"
    build_results_package(
        Path("artifacts/comparisons/soh"),
        Path("artifacts/comparisons/rul"),
        output,
    )
    manifest = json.loads((output / "package_manifest.json").read_text())
    packaged = manifest["packaged_files"]
    assert packaged
    for item in packaged:
        path = output / item["path"]
        assert path.is_file()
        assert item["bytes"] == path.stat().st_size
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_slide_language_contains_interpretation_guardrails(tmp_path: Path) -> None:
    output = tmp_path / "package"
    build_results_package(
        Path("artifacts/comparisons/soh"),
        Path("artifacts/comparisons/rul"),
        output,
    )
    slides = (output / "slide_plan.md").read_text()
    limits = (output / "limitations.md").read_text()
    commands = (output / "reproducibility.md").read_text()
    assert "not used to retune or select" in slides
    assert "cannot be compared directly" in limits
    assert "do not train or select models" in commands
    assert "src.package_results" in commands
