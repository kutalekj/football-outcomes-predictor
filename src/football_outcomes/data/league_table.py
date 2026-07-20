from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from football_outcomes.config import fs_settings as sett

if TYPE_CHECKING:
    from football_outcomes.data.fs_models import (
        FSCompSeason,
        FSMatch,
    )


def valid_round_ids_for_season(
    competition_season: FSCompSeason,
) -> Optional[set[int]]:
    """
    Return the configured strict round whitelist.

    None means that the competition-season is not
    configured for league-table calculations.
    """

    valid_round_ids = getattr(
        sett,
        "LEAGUE_VALID_ROUND_IDS_BY_SEASON",
        {},
    ).get(
        (
            competition_season.name,
            competition_season.season,
        )
    )

    if valid_round_ids is None:
        return None

    return {int(round_id) for round_id in valid_round_ids}


def is_match_table_eligible(
    competition_season: FSCompSeason,
    match: FSMatch,
) -> bool:
    """Return whether a match belongs in this table."""

    if match is None:
        return False

    if (
        getattr(
            match,
            "datetime",
            None,
        )
        is None
    ):
        return False

    match_season = getattr(
        match,
        "season",
        None,
    )
    if match_season is None:
        return False

    if not (sett.FIRST_SEASON <= int(match_season) < sett.LAST_SEASON):
        return False

    if (
        getattr(
            match,
            "comp_name",
            None,
        )
        != competition_season.name
    ):
        return False

    if (
        getattr(
            match,
            "comp_season_id",
            None,
        )
        != competition_season.id
    ):
        return False

    valid_round_ids = valid_round_ids_for_season(competition_season)
    if valid_round_ids is None:
        return False

    round_id = getattr(
        match,
        "round_id",
        None,
    )
    if round_id is None:
        return False

    return int(round_id) in valid_round_ids


def get_table_matches(
    competition_season: FSCompSeason,
) -> list[FSMatch]:
    """Return retained table-eligible matches."""

    return [
        match
        for match in competition_season.matches
        if is_match_table_eligible(
            competition_season,
            match,
        )
    ]


def match_time_key(
    match: FSMatch,
) -> tuple[Any, int, int]:
    """
    Order matches by UTC date, hour and ID.

    Missing hours retain the legacy -1 fallback.
    """

    match_datetime = getattr(
        match,
        "datetime",
        None,
    )
    hour = getattr(
        match,
        "hour_utc",
        None,
    )
    normalized_hour = int(hour) if isinstance(hour, int) else -1

    return (
        match_datetime,
        normalized_hour,
        match.id,
    )


def init_league_table(
    competition_season: FSCompSeason,
) -> None:
    """Initialize table teams and empty statistics."""

    table_matches = get_table_matches(competition_season)

    teams_by_id = {}

    for match in table_matches:
        if match.home_team is not None:
            teams_by_id[match.home_team.id] = match.home_team

        if match.away_team is not None:
            teams_by_id[match.away_team.id] = match.away_team

    competition_season.teams = sorted(
        teams_by_id.values(),
        key=lambda team: team.id,
    )

    competition_season.team_stats = {
        team.id: {
            "points": 0.0,
            "games_played": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "avg_points_per_game": 0.0,
        }
        for team in competition_season.teams
    }

    competition_season._pre_match_positions = {}
    competition_season._table_initialized = True


def ensure_table(
    competition_season: FSCompSeason,
) -> None:
    if not competition_season._table_initialized:
        init_league_table(competition_season)


def reset_table(
    competition_season: FSCompSeason,
) -> None:
    ensure_table(competition_season)

    for team_id in competition_season.team_stats:
        competition_season.team_stats[team_id].update(
            points=0.0,
            games_played=0.0,
            goals_for=0.0,
            goals_against=0.0,
            avg_points_per_game=0.0,
        )


def apply_match_to_table(
    competition_season: FSCompSeason,
    match: FSMatch,
) -> None:
    ensure_table(competition_season)

    if match.home_team is None or match.away_team is None:
        raise ValueError("League table update failed: " "team not found.")

    if match.home_goals is None or match.away_goals is None:
        raise ValueError("League table update failed: " "goals not found.")

    home_team_id = match.home_team.id
    away_team_id = match.away_team.id

    if home_team_id not in competition_season.team_stats or away_team_id not in competition_season.team_stats:
        raise ValueError(
            "League table update failed: "
            "team id not found in current "
            f"table for match {match.id} "
            f"({competition_season.name} "
            f"{competition_season.season})."
        )

    home_goals = float(match.home_goals)
    away_goals = float(match.away_goals)

    home_stats = competition_season.team_stats[home_team_id]
    away_stats = competition_season.team_stats[away_team_id]

    home_stats["games_played"] += 1.0
    home_stats["goals_for"] += home_goals
    home_stats["goals_against"] += away_goals

    away_stats["games_played"] += 1.0
    away_stats["goals_for"] += away_goals
    away_stats["goals_against"] += home_goals

    if home_goals > away_goals:
        home_stats["points"] += 3.0
    elif home_goals < away_goals:
        away_stats["points"] += 3.0
    else:
        home_stats["points"] += 1.0
        away_stats["points"] += 1.0


def recompute_average_points(
    competition_season: FSCompSeason,
) -> None:
    ensure_table(competition_season)

    for stats in competition_season.team_stats.values():
        games_played = stats["games_played"]
        stats["avg_points_per_game"] = stats["points"] / games_played if games_played > 0 else 0.0


def sorted_team_ids(
    competition_season: FSCompSeason,
) -> list[int]:
    """Return the deterministic table ordering."""

    ensure_table(competition_season)
    recompute_average_points(competition_season)

    def ordering_key(
        team_id: int,
    ) -> tuple[float, float, float, int]:
        stats = competition_season.team_stats[team_id]
        goal_difference = stats["goals_for"] - stats["goals_against"]

        return (
            stats["avg_points_per_game"],
            goal_difference,
            stats["goals_for"],
            -team_id,
        )

    return sorted(
        competition_season.team_stats.keys(),
        key=ordering_key,
        reverse=True,
    )


def rank_to_position01(
    rank_1based: int,
    number_of_teams: int,
) -> float:
    """Map first place to 1 and last place to 0."""

    if number_of_teams <= 1:
        return 1.0

    return float(1.0 - ((rank_1based - 1) / (number_of_teams - 1)))


def build_pre_match_positions_cache(
    competition_season: FSCompSeason,
) -> None:
    """
    Cache positions before every retained match.

    Matches at the same date and hour are treated as
    one batch to avoid within-timeslot leakage.
    """

    ensure_table(competition_season)
    reset_table(competition_season)

    matches_sorted = sorted(
        get_table_matches(competition_season),
        key=match_time_key,
    )

    competition_season._pre_match_positions = {}

    index = 0

    while index < len(matches_sorted):
        (
            batch_datetime,
            batch_hour,
            _,
        ) = match_time_key(matches_sorted[index])

        batch = []

        while index < len(matches_sorted):
            (
                match_datetime,
                match_hour,
                _,
            ) = match_time_key(matches_sorted[index])

            if match_datetime != batch_datetime or match_hour != batch_hour:
                break

            batch.append(matches_sorted[index])
            index += 1

        ordered_team_ids = sorted_team_ids(competition_season)
        number_of_teams = len(ordered_team_ids)

        rank_by_team = {
            team_id: rank
            for rank, team_id in enumerate(
                ordered_team_ids,
                start=1,
            )
        }

        for match in batch:
            if match.home_team is None or match.away_team is None:
                continue

            home_team_id = match.home_team.id
            away_team_id = match.away_team.id

            if home_team_id not in rank_by_team or away_team_id not in rank_by_team:
                raise ValueError(
                    "Team missing from pre-match "
                    "ranking cache build for "
                    f"{competition_season.name} "
                    f"{competition_season.season}, "
                    f"match_id={match.id}, "
                    f"round_id={match.round_id}."
                )

            competition_season._pre_match_positions[match.id] = {
                home_team_id: (
                    rank_to_position01(
                        rank_by_team[home_team_id],
                        number_of_teams,
                    )
                ),
                away_team_id: (
                    rank_to_position01(
                        rank_by_team[away_team_id],
                        number_of_teams,
                    )
                ),
            }

        for match in batch:
            apply_match_to_table(
                competition_season,
                match,
            )

    recompute_average_points(competition_season)


def get_team_position_before_match(
    competition_season: FSCompSeason,
    team_id: int,
    match: FSMatch,
) -> float:
    """Return the cached position or recompute it."""

    ensure_table(competition_season)

    position_map = competition_season._pre_match_positions.get(match.id)

    if position_map is not None and team_id in position_map:
        return position_map[team_id]

    return get_team_position_up_to_match(
        competition_season,
        team_id,
        match,
    )


def get_team_position_up_to_match(
    competition_season: FSCompSeason,
    team_id: int,
    match: FSMatch,
) -> float:
    """
    Recompute the table strictly before a match.
    """

    ensure_table(competition_season)

    if (
        match is None
        or getattr(
            match,
            "datetime",
            None,
        )
        is None
    ):
        return 0.0

    if not is_match_table_eligible(
        competition_season,
        match,
    ):
        return 0.0

    target_key = match_time_key(match)

    reset_table(competition_season)

    matches_sorted = sorted(
        get_table_matches(competition_season),
        key=match_time_key,
    )

    for prior_match in matches_sorted:
        if match_time_key(prior_match) >= target_key:
            break

        apply_match_to_table(
            competition_season,
            prior_match,
        )

    ordered_team_ids = sorted_team_ids(competition_season)
    number_of_teams = len(ordered_team_ids)

    for rank, current_team_id in enumerate(
        ordered_team_ids,
        start=1,
    ):
        if current_team_id == team_id:
            return rank_to_position01(
                rank,
                number_of_teams,
            )

    raise ValueError(
        f"Team [{team_id}] not found in "
        "table for competition season "
        f"[{competition_season.name}, "
        f"{competition_season.season}] "
        f"(id={competition_season.id}), "
        f"match_id={match.id}."
    )
