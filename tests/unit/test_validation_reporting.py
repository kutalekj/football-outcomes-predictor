from pathlib import Path

import pytest

from football_outcomes.validation.reporting import (
    combine_validation_reports,
    render_validation_markdown,
    sha256_file,
)


def component(
    *,
    ok: bool = True,
    critical: int = 0,
    warnings: int = 0,
    metrics: dict | None = None,
    findings: list | None = None,
) -> dict:
    return {
        "ok": ok,
        "critical_issue_count": critical,
        "warning_count": warnings,
        "metrics": metrics or {},
        "findings": findings or [],
    }


def complete_components() -> dict:
    return {
        "domain": component(warnings=3),
        "selection": component(
            metrics={
                "selected_matches": 10,
                "selected_competitions": 2,
                ("selected_competition_" "seasons"): 3,
                "constructed_rounds": 4,
            }
        ),
        "readiness": component(
            warnings=1,
            metrics={
                ("processed_array_" "matches"): 9,
                ("binary_under_25_" "count"): 4,
                ("binary_over_25_" "count"): 5,
            },
        ),
        "coverage": component(
            warnings=8,
            metrics={
                ("selected_team_" "mapping_rate"): 1.0,
                ("unique_player_cache_" "match_rate"): 0.9,
                ("lineup_reference_" "cache_match_rate"): 0.95,
                ("strength_cell_" "observation_rate"): 0.98,
            },
        ),
    }


def test_combination_sums_component_results() -> None:
    report = combine_validation_reports(
        snapshot_filename="test.pkl",
        snapshot_sha256="ABC",
        snapshot_size_bytes=100,
        reports=complete_components(),
    )

    assert report["overall_ok"]
    assert report["critical_issue_count"] == 0
    assert report["warning_count"] == 12


def test_failed_component_fails_report() -> None:
    reports = complete_components()
    reports["selection"] = component(
        ok=False,
        critical=2,
    )

    report = combine_validation_reports(
        snapshot_filename="test.pkl",
        snapshot_sha256="ABC",
        snapshot_size_bytes=100,
        reports=reports,
    )

    assert not report["overall_ok"]
    assert report["critical_issue_count"] == 2


def test_missing_component_is_rejected() -> None:
    reports = complete_components()
    reports.pop("coverage")

    with pytest.raises(
        ValueError,
        match="Missing validation components",
    ):
        combine_validation_reports(
            snapshot_filename="test.pkl",
            snapshot_sha256="ABC",
            snapshot_size_bytes=100,
            reports=reports,
        )


def test_markdown_contains_acceptance_summary() -> None:
    report = combine_validation_reports(
        snapshot_filename="test.pkl",
        snapshot_sha256="ABC",
        snapshot_size_bytes=100,
        reports=complete_components(),
    )

    rendered = render_validation_markdown(report)

    assert "# Step 6 full-dataset " "validation report" in rendered
    assert "**PASS**" in rendered
    assert "Selected matches: `10`" in rendered
    assert "Selected-team mapping: " "`100.00%`" in rendered


def test_sha256_file_is_deterministic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"test")

    first = sha256_file(path)
    second = sha256_file(path)

    assert first == second
    assert first == ("9F86D081884C7D659A2FEAA0" "C55AD015A3BF4F1B2B0B822C" "D15D6C15B0F00A08")
