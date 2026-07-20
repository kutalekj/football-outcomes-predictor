from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import datetime

from football_outcomes.data.fs_models import (
    FSDataBundle,
    FSMatch,
)
from football_outcomes.datasets.rounds import (
    distribute_matches_into_rounds,
)
from football_outcomes.utils.fs_feature_utils import (
    match_sort_key,
)
from football_outcomes.validation.domain import (
    DomainValidationReport,
)

CompetitionSeasonKey = tuple[str, int]
ValidRoundMap = Mapping[
    CompetitionSeasonKey,
    set[int] | frozenset[int],
]


@dataclass(frozen=True)
class SelectionValidationConfig:
    competitions: tuple[str, ...]
    first_season: int
    last_season_exclusive: int
    excluded_competition_seasons: frozenset[CompetitionSeasonKey]
    valid_round_ids_by_season: ValidRoundMap


def _season_is_in_range(
    season: object,
    config: SelectionValidationConfig,
) -> bool:
    return type(season) is int and config.first_season <= season < config.last_season_exclusive


def matches_before_round_filter(
    matches: Sequence[FSMatch],
    config: SelectionValidationConfig,
) -> list[FSMatch]:
    result = []

    for match in matches:
        competition = getattr(
            match,
            "comp_name",
            None,
        )
        season = getattr(
            match,
            "season",
            None,
        )

        if competition not in config.competitions:
            continue

        if not _season_is_in_range(
            season,
            config,
        ):
            continue

        key = (
            competition,
            season,
        )

        if key in config.excluded_competition_seasons:
            continue

        result.append(match)

    return result


def select_validation_matches(
    matches: Sequence[FSMatch],
    config: SelectionValidationConfig,
) -> list[FSMatch]:
    selected = []

    for match in matches_before_round_filter(
        matches,
        config,
    ):
        key = (
            match.comp_name,
            match.season,
        )

        valid_round_ids = config.valid_round_ids_by_season.get(key)

        if valid_round_ids is None:
            continue

        if (
            getattr(
                match,
                "round_id",
                None,
            )
            not in valid_round_ids
        ):
            continue

        selected.append(match)

    return selected


def _validate_round_team_uniqueness(
    report: DomainValidationReport,
    rounds: Sequence[Sequence[FSMatch]],
) -> None:
    for round_index, round_matches in enumerate(rounds):
        seen_team_ids: set[int] = set()

        for match in round_matches:
            for side in (
                "home",
                "away",
            ):
                team = getattr(
                    match,
                    f"{side}_team",
                    None,
                )
                team_id = getattr(
                    team,
                    "id",
                    None,
                )

                if type(team_id) is not int:
                    report.add(
                        "round_match_missing_team",
                        entity_type="round",
                        entity_id=round_index,
                        message=(f"Match " f"{getattr(match, 'id', None)} " f"has no valid {side} team."),
                    )
                    continue

                if team_id in seen_team_ids:
                    report.add(
                        "team_repeated_within_round",
                        entity_type="round",
                        entity_id=round_index,
                        message=(f"Team {team_id} appears " "more than once."),
                    )

                seen_team_ids.add(team_id)


def _validate_selected_match_core(
    report: DomainValidationReport,
    matches: Sequence[FSMatch],
) -> None:
    for match in matches:
        for side in (
            "home",
            "away",
        ):
            team = getattr(
                match,
                f"{side}_team",
                None,
            )

            if team is None:
                report.add(
                    (f"selected_match_" f"missing_{side}_team"),
                    entity_type="match",
                    entity_id=match.id,
                    message=(f"Selected match has " f"no {side} team."),
                )

        if match.home_team is not None and match.away_team is not None and match.home_team.id == match.away_team.id:
            report.add(
                ("selected_match_same_" "home_and_away_team"),
                entity_type="match",
                entity_id=match.id,
                message=("Selected match uses the " "same team on both sides."),
            )

        if not isinstance(
            match.datetime,
            datetime,
        ):
            report.add(
                ("selected_match_invalid_" "datetime"),
                entity_type="match",
                entity_id=match.id,
                message=("Selected match has no " "valid datetime."),
            )

        for side in (
            "home",
            "away",
        ):
            goals = getattr(
                match,
                f"{side}_goals",
                None,
            )

            if type(goals) is not int or goals < 0:
                report.add(
                    (f"selected_match_" f"invalid_{side}_goals"),
                    entity_type="match",
                    entity_id=match.id,
                    message=(f"{side} goals must " "be a non-negative " "integer."),
                )


def validate_bundle_selection(
    bundle: FSDataBundle,
    config: SelectionValidationConfig,
    *,
    max_examples_per_finding: int = 5,
) -> DomainValidationReport:
    if max_examples_per_finding < 0:
        raise ValueError("max_examples_per_finding " "must be non-negative.")

    report = DomainValidationReport(max_examples_per_finding=(max_examples_per_finding))

    report.metrics["total_snapshot_matches"] = len(bundle.matches)

    in_scope = matches_before_round_filter(
        bundle.matches,
        config,
    )

    report.metrics["matches_before_round_filter"] = len(in_scope)

    present_competitions = {match.comp_name for match in in_scope}

    for competition in sorted(set(config.competitions) - present_competitions):
        report.add(
            "configured_competition_missing",
            entity_type="competition",
            entity_id=competition,
            message=("No in-range, non-excluded " "matches were found."),
        )

    present_keys = {
        (
            match.comp_name,
            match.season,
        )
        for match in in_scope
    }

    for key in sorted(present_keys):
        if key not in config.valid_round_ids_by_season:
            report.add(
                "missing_round_whitelist",
                entity_type=("competition-season"),
                entity_id=key,
                message=("No valid round-id set " "is configured."),
            )

    selected = select_validation_matches(
        bundle.matches,
        config,
    )

    report.metrics["selected_matches"] = len(selected)
    report.metrics["matches_filtered_by_round"] = len(in_scope) - len(selected)
    report.metrics["selected_competitions"] = len({match.comp_name for match in selected})
    report.metrics["selected_competition_seasons"] = len(
        {
            (
                match.comp_name,
                match.season,
            )
            for match in selected
        }
    )

    if not selected:
        report.add(
            "no_selected_matches",
            entity_type="selection",
            entity_id="all",
            message=("Selection produced no matches."),
        )
        return report

    _validate_selected_match_core(
        report,
        selected,
    )

    if report.critical_issue_count:
        return report

    ordered = sorted(
        selected,
        key=match_sort_key,
    )

    rounds = distribute_matches_into_rounds(ordered)
    flattened = [match for round_matches in rounds for match in round_matches]

    report.metrics["constructed_rounds"] = len(rounds)

    round_sizes = [len(round_matches) for round_matches in rounds]

    report.metrics["minimum_round_size"] = min(round_sizes)
    report.metrics["maximum_round_size"] = max(round_sizes)
    report.metrics["final_round_size"] = round_sizes[-1]

    if len(flattened) != len(ordered):
        report.add(
            "round_match_count_mismatch",
            entity_type="rounds",
            entity_id="all",
            message=(f"Rounds contain " f"{len(flattened)} matches, " f"but selection contains " f"{len(ordered)}."),
        )

    ordered_ids = [match.id for match in ordered]
    flattened_ids = [match.id for match in flattened]

    if flattened_ids != ordered_ids:
        report.add(
            "round_order_mismatch",
            entity_type="rounds",
            entity_id="all",
            message=("Round construction did not " "preserve chronological order."),
        )

    _validate_round_team_uniqueness(
        report,
        rounds,
    )

    return report
