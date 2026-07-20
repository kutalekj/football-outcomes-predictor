from __future__ import annotations

from pathlib import Path

import pytest

from football_outcomes.validation.imputation import (
    build_step7_validation_report,
    choose_audit_fold_indices,
    render_step7_validation_markdown,
    safe_ratio,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def valid_arguments() -> dict:
    return {
        "snapshot": {
            "filename": "snapshot.pkl",
            "sha256": "ABC",
            "size_bytes": 100,
        },
        "config": {
            "window_rounds": 25,
            "audit_fold_count": 5,
            "minimum_group_support": 20,
            "neutral_value": 50.0,
            "player_count": 11,
            "skill_count": 34,
            "max_age_days": 365,
            "max_snapshots": 4,
        },
        "scope": {
            "selected_matches": 2,
            "array_ready_matches": 2,
            "round_count": 30,
            "competition_seasons": 1,
            "total_strength_cells": 1496,
        },
        "legacy": {
            "observed_cells": 1400,
            "missing_cells": 96,
            "observation_rate": 1400 / 1496,
        },
        "past_only": {
            "nearest_past_cells": 1300,
            "older_past_cells": 100,
            "observed_cells": 1400,
            "unresolved_cells": 96,
            "observation_rate": 1400 / 1496,
            "maximum_source_age_days": 100,
            "future_source_cells": 0,
            "source_cells_beyond_max_age": 0,
            "invalid_observed_values": 0,
            "invalid_matrices": 0,
        },
        "comparison": {
            "both_observed_equal_cells": 1390,
            "both_observed_changed_cells": 5,
            "legacy_only_observed_cells": 5,
            "past_only_only_observed_cells": 5,
            "both_missing_cells": 91,
        },
        "rolling_audit": {
            "audit_fold_count": 1,
            "audit_round_indices": [30],
            "validation_matches": 1,
            "validation_cells": 748,
            "observed_provenance_cells": 700,
            "imputed_provenance_cells": 48,
            "unresolved_provenance_cells": 0,
            "neutral_fallback_cells": 0,
            "observed_mask_mismatches": 0,
            "nonstructured_array_mismatches": 0,
            "invalid_strength_values": 0,
            "invalid_masks": 0,
            "provenance_conservation_failures": 0,
            "validation_provenance_counts": {
                "NEAREST_PAST_SOFIFA": 700,
                "GLOBAL_SKILL_MEDIAN": 48,
            },
        },
    }


def test_validation_module_is_pure_and_offline() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "validation" / "imputation.py"
    source = source_path.read_text(encoding="utf-8")

    assert "requests" not in source
    assert "fs_globals" not in source
    assert "Global" not in source
    assert "fs_settings" not in source
    assert "tensorflow" not in source


def test_audit_indices_include_first_and_final_folds() -> None:
    indices = choose_audit_fold_indices(
        round_count=320,
        window_rounds=25,
        audit_fold_count=5,
    )

    assert indices[0] == 25
    assert indices[-1] == 319
    assert len(indices) == 5
    assert len(indices) == len(set(indices))
    assert indices == choose_audit_fold_indices(
        round_count=320,
        window_rounds=25,
        audit_fold_count=5,
    )


def test_no_eligible_fold_returns_empty_result() -> None:
    assert (
        choose_audit_fold_indices(
            round_count=25,
            window_rounds=25,
            audit_fold_count=5,
        )
        == ()
    )


def test_valid_report_passes_with_quality_warnings() -> None:
    report = build_step7_validation_report(**valid_arguments())

    assert report["overall_ok"]
    assert report["critical_issue_count"] == 0
    assert report["warning_count"] > 0


def test_future_source_cells_fail_report() -> None:
    arguments = valid_arguments()
    arguments["past_only"] = {
        **arguments["past_only"],
        "future_source_cells": 2,
    }

    report = build_step7_validation_report(**arguments)

    assert not report["overall_ok"]
    assert report["critical_issue_count"] == 2
    assert any(finding["code"] == "future_source_cells" for finding in report["critical_findings"])


def test_unresolved_audited_cells_fail_report() -> None:
    arguments = valid_arguments()
    arguments["rolling_audit"] = {
        **arguments["rolling_audit"],
        "imputed_provenance_cells": 45,
        "unresolved_provenance_cells": 3,
    }

    report = build_step7_validation_report(**arguments)

    assert not report["overall_ok"]
    assert report["critical_issue_count"] == 3


def test_markdown_contains_acceptance_evidence() -> None:
    report = build_step7_validation_report(**valid_arguments())
    rendered = render_step7_validation_markdown(report)

    assert "# Step 7 leakage-safe SoFIFA " "imputation report" in rendered
    assert "**PASS**" in rendered
    assert "Future-source cells: `0`" in rendered
    assert "Cells left unresolved: `0`" in rendered


def test_safe_ratio_handles_zero_denominator() -> None:
    assert safe_ratio(1, 0) == 0.0
    assert safe_ratio(1, 4) == 0.25


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        (
            "window_rounds",
            0,
            "window_rounds must be",
        ),
        (
            "audit_fold_count",
            0,
            "audit_fold_count must be",
        ),
        (
            "round_count",
            -1,
            "round_count must be",
        ),
    ],
)
def test_invalid_fold_configuration_is_rejected(
    argument,
    value,
    message,
) -> None:
    values = {
        "round_count": 30,
        "window_rounds": 25,
        "audit_fold_count": 5,
    }
    values[argument] = value

    with pytest.raises(
        ValueError,
        match=message,
    ):
        choose_audit_fold_indices(**values)
