from datetime import datetime, timezone
from pathlib import Path

import pytest

from football_outcomes.config import fs_settings as settings
from football_outcomes.data import (
    league_table,
)
from football_outcomes.data.fs_models import (
    FSCompSeason,
    FSMatch,
    FSTeam,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_team(
    team_id: int,
) -> FSTeam:
    return FSTeam(
        team_id,
        f"Team {team_id}",
        f"team {team_id}",
        f"Team {team_id}",
        f"Team {team_id}",
        f"T{team_id}",
        "Test Country",
    )


def make_match(
    match_id: int,
    home_team: FSTeam,
    away_team: FSTeam,
    *,
    round_id: int,
    match_datetime: datetime,
    home_goals: int,
    away_goals: int,
) -> FSMatch:
    match = FSMatch(match_id)
    match.home_team = home_team
    match.away_team = away_team
    match.season = 2024
    match.comp_season_id = 10
    match.comp_name = "Test League"
    match.round_id = round_id
    match.datetime = match_datetime
    match.hour_utc = match_datetime.hour
    match.home_goals = home_goals
    match.away_goals = away_goals

    return match


def test_league_table_module_has_no_global_or_network_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "league_table.py"
    source = source_path.read_text(encoding="utf-8")

    assert "football_outcomes.config.fs_globals" not in source
    assert "requests" not in source
    assert "http.client" not in source


def test_same_timeslot_positions_do_not_leak(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "FIRST_SEASON",
        2021,
    )
    monkeypatch.setattr(
        settings,
        "LAST_SEASON",
        2025,
    )
    monkeypatch.setattr(
        settings,
        "LEAGUE_VALID_ROUND_IDS_BY_SEASON",
        {
            (
                "Test League",
                2024,
            ): {1, 2}
        },
    )

    teams = {team_id: make_team(team_id) for team_id in range(1, 5)}

    first_time = datetime(
        2024,
        1,
        1,
        12,
        tzinfo=timezone.utc,
    )
    second_time = datetime(
        2024,
        1,
        2,
        12,
        tzinfo=timezone.utc,
    )

    first_match = make_match(
        101,
        teams[1],
        teams[4],
        round_id=1,
        match_datetime=first_time,
        home_goals=2,
        away_goals=0,
    )
    simultaneous_match = make_match(
        102,
        teams[2],
        teams[3],
        round_id=1,
        match_datetime=first_time,
        home_goals=0,
        away_goals=1,
    )
    later_match = make_match(
        103,
        teams[1],
        teams[3],
        round_id=2,
        match_datetime=second_time,
        home_goals=0,
        away_goals=0,
    )

    competition_season = FSCompSeason(
        10,
        2024,
        "Test Country",
        "Test League",
    )
    competition_season.matches = [
        first_match,
        simultaneous_match,
        later_match,
    ]

    league_table.build_pre_match_positions_cache(competition_season)

    first_positions = competition_season._pre_match_positions[first_match.id]
    simultaneous_positions = competition_season._pre_match_positions[simultaneous_match.id]
    later_positions = competition_season._pre_match_positions[later_match.id]

    assert first_positions[teams[1].id] == pytest.approx(1.0)
    assert first_positions[teams[4].id] == pytest.approx(0.0)

    assert simultaneous_positions[teams[2].id] == pytest.approx(2.0 / 3.0)
    assert simultaneous_positions[teams[3].id] == pytest.approx(1.0 / 3.0)

    assert later_positions[teams[1].id] == pytest.approx(1.0)
    assert later_positions[teams[3].id] == pytest.approx(2.0 / 3.0)


def test_legacy_methods_delegate_to_service(
    monkeypatch,
) -> None:
    competition_season = object.__new__(FSCompSeason)
    match = object()
    calls = []

    def fake_position(
        supplied_season,
        supplied_team_id,
        supplied_match,
    ):
        calls.append(
            (
                supplied_season,
                supplied_team_id,
                supplied_match,
            )
        )
        return 0.75

    monkeypatch.setattr(
        league_table,
        "get_team_position_before_match",
        fake_position,
    )

    result = competition_season.get_team_position_before_match(
        123,
        match,
    )

    assert result == 0.75
    assert calls == [
        (
            competition_season,
            123,
            match,
        )
    ]


def test_rank_normalization_is_preserved() -> None:
    assert (
        league_table.rank_to_position01(
            1,
            20,
        )
        == 1.0
    )
    assert (
        league_table.rank_to_position01(
            20,
            20,
        )
        == 0.0
    )
    assert (
        league_table.rank_to_position01(
            1,
            1,
        )
        == 1.0
    )

    assert FSCompSeason._rank_to_position01(
        10,
        20,
    ) == league_table.rank_to_position01(
        10,
        20,
    )
