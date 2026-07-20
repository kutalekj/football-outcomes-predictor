from __future__ import annotations

from pathlib import Path

from football_outcomes.validation.coverage import (
    validate_coverage_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def perfect_summary() -> dict:
    return {
        "selected_matches": 1,
        "selected_match_sides": 2,
        "feature_ready_matches": 1,
        "complete_lineups": 2,
        "empty_lineups": 0,
        "partial_lineups": 0,
        "oversized_lineups": 0,
        "lineup_player_references": 22,
        ("lineup_cache_" "matched_references"): 22,
        ("lineup_cache_" "failed_references"): 0,
        "unique_lineup_players": 22,
        ("unique_lineup_players_" "cache_matched"): 22,
        ("unique_lineup_players_" "cache_failed"): 0,
        "unique_selected_teams": 2,
        "mapped_selected_teams": 2,
        "strength_matrices": 2,
        "valid_strength_matrices": 2,
        "missing_strength_matrices": 0,
        "invalid_strength_matrices": 0,
        "strength_cells": 748,
        "observed_strength_cells": 748,
        "missing_strength_cells": 0,
        "selected_team_mapping_rate": 1.0,
        ("unique_player_cache_" "match_rate"): 1.0,
        ("lineup_reference_cache_" "match_rate"): 1.0,
        ("strength_cell_" "observation_rate"): 1.0,
    }


def perfect_rows() -> list[dict]:
    return [
        {
            "competition": "Test League",
            "season": 2024,
            "selected_matches": 1,
            "lineup_cache_match_rate": 1.0,
            ("strength_cell_" "observation_rate"): 1.0,
        }
    ]


def test_coverage_module_is_offline_and_global_free() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "validation" / "coverage.py"
    source = source_path.read_text(encoding="utf-8")

    assert "requests" not in source
    assert "fs_globals" not in source
    assert "Global" not in source
    assert "fs_settings" not in source


def test_perfect_coverage_passes_without_findings() -> None:
    report = validate_coverage_summary(
        perfect_summary(),
        perfect_rows(),
    )

    assert report.ok
    assert report.critical_issue_count == 0
    assert report.warning_count == 0


def test_quality_gaps_are_warnings() -> None:
    summary = perfect_summary()
    summary.update(
        {
            "complete_lineups": 1,
            "partial_lineups": 1,
            ("lineup_cache_" "matched_references"): 20,
            ("lineup_cache_" "failed_references"): 2,
            ("unique_lineup_players_" "cache_matched"): 20,
            ("unique_lineup_players_" "cache_failed"): 2,
            "observed_strength_cells": 700,
            "missing_strength_cells": 48,
            ("fully_missing_" "strength_rows"): 1,
        }
    )

    report = validate_coverage_summary(
        summary,
        perfect_rows(),
    )

    assert report.ok
    assert report.count_for("noncomplete_lineups") == 1
    assert report.count_for(("unmatched_lineup_player_" "references")) == 2
    assert report.count_for("missing_strength_cells") == 48


def test_invalid_strength_matrix_is_critical() -> None:
    summary = perfect_summary()
    summary.update(
        {
            "valid_strength_matrices": 1,
            "invalid_strength_matrices": 1,
        }
    )

    report = validate_coverage_summary(
        summary,
        perfect_rows(),
    )

    assert report.count_for("invalid_strength_matrices") == 1
    assert not report.ok


def test_lineup_conservation_failure_is_critical() -> None:
    summary = perfect_summary()
    summary["complete_lineups"] = 1

    report = validate_coverage_summary(
        summary,
        perfect_rows(),
    )

    assert report.count_for("lineup_side_count_mismatch") == 1
    assert not report.ok


def test_incomplete_team_mapping_is_critical() -> None:
    summary = perfect_summary()
    summary.update(
        {
            "mapped_selected_teams": 1,
            ("missing_selected_" "team_mappings"): 1,
        }
    )

    report = validate_coverage_summary(
        summary,
        perfect_rows(),
    )

    assert report.count_for(("incomplete_selected_" "team_mapping")) == 1
    assert not report.ok


def test_scope_findings_are_deterministic_and_capped() -> None:
    rows = [
        {
            "competition": "League B",
            "season": 2024,
            "selected_matches": 0,
            "lineup_cache_match_rate": 0.9,
            ("strength_cell_" "observation_rate"): 0.8,
        },
        {
            "competition": "League A",
            "season": 2024,
            "selected_matches": 1,
            "lineup_cache_match_rate": 0.8,
            ("strength_cell_" "observation_rate"): 0.9,
        },
    ]

    first = validate_coverage_summary(
        perfect_summary(),
        rows,
        max_examples_per_finding=1,
    )
    second = validate_coverage_summary(
        perfect_summary(),
        rows,
        max_examples_per_finding=1,
    )

    assert first.to_dict() == second.to_dict()
    assert first.count_for(("competition_season_" "player_match_gap")) == 2
    assert len(first.findings[("competition_season_" "player_match_gap")].examples) == 1
