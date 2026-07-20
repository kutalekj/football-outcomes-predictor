from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Any

from football_outcomes.validation.domain import (
    DomainValidationReport,
)

_REQUIRED_SUMMARY_KEYS = (
    "selected_matches",
    "selected_match_sides",
    "feature_ready_matches",
    "complete_lineups",
    "empty_lineups",
    "partial_lineups",
    "oversized_lineups",
    "lineup_player_references",
    "lineup_cache_matched_references",
    "lineup_cache_failed_references",
    "unique_lineup_players",
    "unique_lineup_players_cache_matched",
    "unique_lineup_players_cache_failed",
    "unique_selected_teams",
    "mapped_selected_teams",
    "strength_matrices",
    "valid_strength_matrices",
    "missing_strength_matrices",
    "invalid_strength_matrices",
    "strength_cells",
    "observed_strength_cells",
    "missing_strength_cells",
)


def _integer_metric(
    summary: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int | None:
    value = summary.get(
        key,
        default,
    )

    if type(value) is not int:
        return None

    return value


def _optional_count(
    summary: Mapping[str, Any],
    key: str,
) -> int:
    value = _integer_metric(
        summary,
        key,
        default=0,
    )

    if value is None:
        return 0

    return value


def _add_count_warning(
    report: DomainValidationReport,
    *,
    code: str,
    count: int,
    message: str,
) -> None:
    if count <= 0:
        return

    report.add(
        code,
        entity_type="coverage",
        entity_id="selected-scope",
        message=message,
        severity="warning",
        count=count,
    )


def _add_conservation_failure(
    report: DomainValidationReport,
    *,
    code: str,
    expected: int,
    actual: int,
    description: str,
) -> None:
    if actual == expected:
        return

    report.add(
        code,
        entity_type="coverage",
        entity_id="selected-scope",
        message=(f"{description}: expected " f"{expected}, found {actual}."),
    )


def _validate_rates(
    report: DomainValidationReport,
    summary: Mapping[str, Any],
) -> None:
    for key in (
        "selected_team_mapping_rate",
        "unique_player_cache_match_rate",
        "lineup_reference_cache_match_rate",
        "strength_cell_observation_rate",
    ):
        if key not in summary:
            continue

        value = summary[key]

        if (
            not isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 1.0
        ):
            report.add(
                "invalid_coverage_rate",
                entity_type="metric",
                entity_id=key,
                message=("Coverage rate must be " f"in [0, 1]; found " f"{value!r}."),
            )


def _validate_scope_rows(
    report: DomainValidationReport,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    selected_match_total = 0

    for row in rows:
        competition = str(
            row.get(
                "competition",
                "unknown",
            )
        )
        season = row.get(
            "season",
            "unknown",
        )
        scope_id = (
            competition,
            season,
        )

        selected_matches = row.get(
            "selected_matches",
            0,
        )

        if type(selected_matches) is int:
            selected_match_total += selected_matches

        cache_rate = row.get("lineup_cache_match_rate")

        if (
            isinstance(
                cache_rate,
                (
                    int,
                    float,
                ),
            )
            and not isinstance(
                cache_rate,
                bool,
            )
            and float(cache_rate) < 1.0
        ):
            report.add(
                ("competition_season_" "player_match_gap"),
                entity_type=("competition-season"),
                entity_id=scope_id,
                message=("Lineup-reference player " "match rate is " f"{float(cache_rate):.6f}."),
                severity="warning",
            )

        strength_rate = row.get(("strength_cell_" "observation_rate"))

        if (
            isinstance(
                strength_rate,
                (
                    int,
                    float,
                ),
            )
            and not isinstance(
                strength_rate,
                bool,
            )
            and float(strength_rate) < 1.0
        ):
            report.add(
                ("competition_season_" "strength_gap"),
                entity_type=("competition-season"),
                entity_id=scope_id,
                message=("Strength-cell observation " "rate is " f"{float(strength_rate):.6f}."),
                severity="warning",
            )

    expected = report.metrics.get("selected_matches")

    if type(expected) is int and selected_match_total != expected:
        report.add(
            ("competition_season_" "match_total_mismatch"),
            entity_type="coverage",
            entity_id="competition-seasons",
            message=(
                "Competition-season rows "
                f"contain {selected_match_total} "
                "matches, while the summary "
                f"contains {expected}."
            ),
        )


def validate_coverage_summary(
    summary: Mapping[str, Any],
    competition_seasons: Sequence[Mapping[str, Any]],
    *,
    max_examples_per_finding: int = 5,
) -> DomainValidationReport:
    if max_examples_per_finding < 0:
        raise ValueError("max_examples_per_finding " "must be non-negative.")

    report = DomainValidationReport(max_examples_per_finding=(max_examples_per_finding))

    missing_keys = [key for key in _REQUIRED_SUMMARY_KEYS if key not in summary]

    for key in missing_keys:
        report.add(
            "missing_coverage_metric",
            entity_type="metric",
            entity_id=key,
            message=("Required coverage metric " "is absent."),
        )

    if missing_keys:
        return report

    for key, value in summary.items():
        if isinstance(
            value,
            (
                int,
                float,
            ),
        ) and not isinstance(
            value,
            bool,
        ):
            report.metrics[key] = value

            if value < 0:
                report.add(
                    "negative_coverage_metric",
                    entity_type="metric",
                    entity_id=key,
                    message=("Coverage metrics must " "not be negative."),
                )

    report.metrics["competition_seasons"] = len(competition_seasons)

    selected_sides = _integer_metric(
        summary,
        "selected_match_sides",
    )
    lineup_references = _integer_metric(
        summary,
        "lineup_player_references",
    )
    unique_players = _integer_metric(
        summary,
        "unique_lineup_players",
    )
    unique_teams = _integer_metric(
        summary,
        "unique_selected_teams",
    )
    strength_matrices = _integer_metric(
        summary,
        "strength_matrices",
    )
    strength_cells = _integer_metric(
        summary,
        "strength_cells",
    )

    assert selected_sides is not None
    assert lineup_references is not None
    assert unique_players is not None
    assert unique_teams is not None
    assert strength_matrices is not None
    assert strength_cells is not None

    classified_lineups = sum(
        _optional_count(
            summary,
            key,
        )
        for key in (
            "complete_lineups",
            "partial_lineups",
            "empty_lineups",
            "oversized_lineups",
            "invalid_lineups",
        )
    )

    _add_conservation_failure(
        report,
        code=("lineup_side_count_mismatch"),
        expected=selected_sides,
        actual=classified_lineups,
        description=("Classified lineup sides"),
    )

    classified_references = sum(
        _optional_count(
            summary,
            key,
        )
        for key in (
            ("lineup_cache_" "matched_references"),
            ("lineup_cache_" "failed_references"),
            ("lineup_cache_" "missing_references"),
            ("lineup_cache_" "invalid_references"),
            ("invalid_lineup_" "player_references"),
        )
    )

    _add_conservation_failure(
        report,
        code=("lineup_reference_" "count_mismatch"),
        expected=lineup_references,
        actual=classified_references,
        description=("Classified lineup player " "references"),
    )

    classified_unique_players = sum(
        _optional_count(
            summary,
            key,
        )
        for key in (
            ("unique_lineup_players_" "cache_matched"),
            ("unique_lineup_players_" "cache_failed"),
            ("unique_lineup_players_" "cache_missing"),
            ("unique_lineup_players_" "cache_invalid"),
        )
    )

    _add_conservation_failure(
        report,
        code=("unique_player_status_" "count_mismatch"),
        expected=unique_players,
        actual=classified_unique_players,
        description=("Classified unique lineup " "players"),
    )

    classified_teams = sum(
        _optional_count(
            summary,
            key,
        )
        for key in (
            "mapped_selected_teams",
            ("missing_selected_" "team_mappings"),
            ("explicitly_unmapped_" "selected_teams"),
        )
    )

    _add_conservation_failure(
        report,
        code=("selected_team_mapping_" "count_mismatch"),
        expected=unique_teams,
        actual=classified_teams,
        description=("Classified selected teams"),
    )

    classified_matrices = sum(
        _optional_count(
            summary,
            key,
        )
        for key in (
            "valid_strength_matrices",
            "missing_strength_matrices",
            "invalid_strength_matrices",
        )
    )

    _add_conservation_failure(
        report,
        code=("strength_matrix_" "count_mismatch"),
        expected=strength_matrices,
        actual=classified_matrices,
        description=("Classified strength matrices"),
    )

    _add_conservation_failure(
        report,
        code=("strength_side_" "count_mismatch"),
        expected=selected_sides,
        actual=strength_matrices,
        description=("Strength matrices"),
    )

    classified_cells = _optional_count(
        summary,
        "observed_strength_cells",
    ) + _optional_count(
        summary,
        "missing_strength_cells",
    )

    _add_conservation_failure(
        report,
        code=("strength_cell_" "count_mismatch"),
        expected=strength_cells,
        actual=classified_cells,
        description=("Classified strength cells"),
    )

    invalid_lineups = _optional_count(
        summary,
        "invalid_lineups",
    )
    invalid_player_references = _optional_count(
        summary,
        ("invalid_lineup_" "player_references"),
    )
    invalid_matrices = _optional_count(
        summary,
        "invalid_strength_matrices",
    )
    missing_team_mappings = _optional_count(
        summary,
        ("missing_selected_" "team_mappings"),
    ) + _optional_count(
        summary,
        ("explicitly_unmapped_" "selected_teams"),
    )

    for (
        code,
        count,
        message,
    ) in (
        (
            "invalid_selected_lineups",
            invalid_lineups,
            ("Selected match sides have " "non-list lineups."),
        ),
        (
            ("invalid_selected_lineup_" "player_references"),
            invalid_player_references,
            ("Selected lineups contain " "invalid player references."),
        ),
        (
            "invalid_strength_matrices",
            invalid_matrices,
            ("Selected strength matrices " "have invalid shapes or " "values."),
        ),
        (
            ("incomplete_selected_" "team_mapping"),
            missing_team_mappings,
            ("Selected teams are not all " "mapped to SoFIFA teams."),
        ),
    ):
        if count > 0:
            report.add(
                code,
                entity_type="coverage",
                entity_id="selected-scope",
                message=message,
                count=count,
            )

    noncomplete_lineups = sum(
        _optional_count(
            summary,
            key,
        )
        for key in (
            "empty_lineups",
            "partial_lineups",
            "oversized_lineups",
        )
    )

    _add_count_warning(
        report,
        code="noncomplete_lineups",
        count=noncomplete_lineups,
        message=("Selected match sides do not " "contain exactly 11 raw lineup " "entries."),
    )
    _add_count_warning(
        report,
        code=("lineups_without_explicit_" "goalkeeper"),
        count=_optional_count(
            summary,
            ("lineups_without_" "explicit_goalkeeper"),
        ),
        message=("Raw lineups contain no player " "labelled as goalkeeper."),
    )

    unmatched_references = sum(
        _optional_count(
            summary,
            key,
        )
        for key in (
            ("lineup_cache_" "failed_references"),
            ("lineup_cache_" "missing_references"),
            ("lineup_cache_" "invalid_references"),
        )
    )

    _add_count_warning(
        report,
        code=("unmatched_lineup_player_" "references"),
        count=unmatched_references,
        message=("Lineup player references do " "not have successful SoFIFA " "matches."),
    )

    unmatched_unique_players = sum(
        _optional_count(
            summary,
            key,
        )
        for key in (
            ("unique_lineup_players_" "cache_failed"),
            ("unique_lineup_players_" "cache_missing"),
            ("unique_lineup_players_" "cache_invalid"),
        )
    )

    _add_count_warning(
        report,
        code=("unmatched_unique_lineup_" "players"),
        count=unmatched_unique_players,
        message=("Unique lineup players do not " "have successful SoFIFA matches."),
    )

    for code, key, message in (
        (
            "missing_strength_matrices",
            "missing_strength_matrices",
            ("Strength matrices are " "absent."),
        ),
        (
            "fully_missing_strength_matrices",
            ("fully_missing_" "strength_matrices"),
            ("Validly shaped strength " "matrices contain no " "observed cells."),
        ),
        (
            "missing_strength_cells",
            "missing_strength_cells",
            ("Strength skill cells are " "missing."),
        ),
        (
            "fully_missing_strength_rows",
            ("fully_missing_" "strength_rows"),
            ("Player-strength rows contain " "no observed skill cells."),
        ),
        (
            ("partially_missing_" "strength_rows"),
            ("partially_missing_" "strength_rows"),
            ("Player-strength rows contain " "some missing skill cells."),
        ),
    ):
        _add_count_warning(
            report,
            code=code,
            count=_optional_count(
                summary,
                key,
            ),
            message=message,
        )

    _validate_rates(
        report,
        summary,
    )
    _validate_scope_rows(
        report,
        competition_seasons,
    )

    return report
