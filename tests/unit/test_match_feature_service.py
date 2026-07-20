from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_outcomes.config import fs_settings as settings
from football_outcomes.data import (
    match_features,
)
from football_outcomes.data.fs_models import (
    FSMatch,
    FSMatchFeatures,
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


def test_feature_service_does_not_define_serialized_classes() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "match_features.py"
    source = source_path.read_text(encoding="utf-8")

    assert "class FSMatch:" not in source
    assert "class FSMatchFeatures:" not in source

    assert FSMatch.__module__ == "football_outcomes.data.fs_models"
    assert FSMatchFeatures.__module__ == "football_outcomes.data.fs_models"


def test_models_no_longer_own_feature_dependencies() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "fs_models.py"
    source = source_path.read_text(encoding="utf-8")

    assert "fs_globals" not in source
    assert "fs_feature_utils" not in source
    assert "fs_player_skill_utils" not in source
    assert "calculate_team_strength" not in source


def test_legacy_match_method_delegates_to_service(
    monkeypatch,
) -> None:
    match = FSMatch(100)
    sentinel = object()
    calls = []

    def fake_builder(
        *,
        match,
        team_index_league,
        team_index_all,
    ):
        calls.append(
            (
                match,
                team_index_league,
                team_index_all,
            )
        )
        return sentinel

    monkeypatch.setattr(
        match_features,
        "calculate_match_features",
        fake_builder,
    )

    league_index = object()
    all_index = object()

    result = match.calculate_match_features(
        league_index,
        all_index,
    )

    assert result is sentinel
    assert calls == [
        (
            match,
            league_index,
            all_index,
        )
    ]


def test_feature_service_preserves_core_outputs(
    monkeypatch,
) -> None:
    class FeatureUtilsStub:
        @staticmethod
        def hour_month_cyclic(
            hour,
            month,
        ):
            assert hour == 12
            assert month == 2

            return (
                0.1,
                0.2,
                0.3,
                0.4,
            )

        @staticmethod
        def normalize_season(
            season,
        ):
            assert season == 2024
            return 0.25

        @staticmethod
        def calculate_elo_for_match(
            *,
            team_index_league,
            curr_match,
        ):
            return (
                0.35,
                0.45,
            )

        @staticmethod
        def clip01(value):
            return float(value)

        @staticmethod
        def avg_goals_scored_conceded_role_last_n(
            *args,
            **kwargs,
        ):
            return (
                0.6,
                0.7,
            )

        def __getattr__(
            self,
            name,
        ):
            return lambda *args, **kwargs: 0.5

    home_team = make_team(1)
    away_team = make_team(2)

    first_match_date = datetime(
        2024,
        1,
        1,
    )
    last_match_date = first_match_date + timedelta(days=10)
    current_datetime = first_match_date + timedelta(days=5)

    competition_season = SimpleNamespace(
        first_match_date=(first_match_date),
        last_match_date=(last_match_date),
        get_team_position_before_match=(lambda team_id, match: (0.8 if team_id == home_team.id else 0.4)),
    )
    global_state = SimpleNamespace(all_comp_seasons={10: competition_season})

    monkeypatch.setattr(
        match_features,
        "fu",
        FeatureUtilsStub(),
    )
    monkeypatch.setattr(
        match_features,
        "Global",
        SimpleNamespace(get_instance=(lambda: global_state)),
    )
    monkeypatch.setattr(
        match_features,
        "calculate_team_strength",
        (lambda match, team_id: [[float(team_id)]]),
    )

    match = FSMatch(200)
    match.home_team = home_team
    match.away_team = away_team
    match.season = 2024
    match.comp_season_id = 10
    match.comp_name = settings.COMPS_LEAGUE[0]
    match.datetime = current_datetime
    match.hour_utc = 12
    match.month = 2

    features = match_features.calculate_match_features(
        match=match,
        team_index_league=object(),
        team_index_all=object(),
    )

    assert isinstance(
        features,
        FSMatchFeatures,
    )

    assert features.comp_id == 0
    assert features.season == 0.25

    assert features.hours_sin == 0.1
    assert features.hours_cos == 0.2
    assert features.month_sin == 0.3
    assert features.month_cos == 0.4

    assert features.home_elo == 0.35
    assert features.away_elo == 0.45

    assert features.match_position_in_season == pytest.approx(0.55)

    assert features.home_avg_xg_last_5 == 0.5
    assert features.away_avg_points_last_20 == 0.5

    assert features.home_curr_position == 0.8
    assert features.away_curr_position == 0.4

    assert features.home_avg_goals_scored_home_last_5 == 0.6
    assert features.home_avg_goals_conceded_home_last_5 == 0.7

    assert features.home_team_strength == [[1.0]]
    assert features.away_team_strength == [[2.0]]
