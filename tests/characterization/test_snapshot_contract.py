from __future__ import annotations

import pickle

import pytest

from football_outcomes.config.fs_globals import (
    Global,
)
from football_outcomes.data.fs_models import (
    FSCompSeason,
    FSDataBundle,
    FSMatch,
    FSPlayer,
    FSTeam,
)
from football_outcomes.data.snapshots import (
    SNAPSHOT_VERSION,
    load_snapshot,
    save_snapshot,
)

MODEL_MODULE = "football_outcomes.data.fs_models"


def build_test_bundle() -> FSDataBundle:
    competition = FSCompSeason(
        100,
        2024,
        "England",
        "England Premier League",
    )

    home_team = FSTeam(
        101,
        "Home FC",
        "home fc",
        "Home FC",
        "Home Football Club",
        "HOME",
        "England",
    )
    away_team = FSTeam(
        102,
        "Away FC",
        "away fc",
        "Away FC",
        "Away Football Club",
        "AWAY",
        "England",
    )

    home_player = FSPlayer(
        201,
        "Home Player",
        "Home",
        "Player",
        "H. Player",
        "Home Player",
    )
    away_player = FSPlayer(
        202,
        "Away Player",
        "Away",
        "Player",
        "A. Player",
        "Away Player",
    )

    match = FSMatch(301)
    match.home_team = home_team
    match.away_team = away_team
    match.home_lineup = [home_player]
    match.away_lineup = [away_player]
    match.season = 2024
    match.comp_season_id = competition.id
    match.comp_name = competition.name
    match.country = competition.country
    match.home_goals = 1
    match.away_goals = 0

    competition.matches = [match]

    home_team.comp_seasons[competition.id] = [home_player]
    away_team.comp_seasons[competition.id] = [away_player]

    return FSDataBundle(
        comp_seasons={
            competition.id: competition,
        },
        teams={
            home_team.id: home_team,
            away_team.id: away_team,
        },
        players={
            home_player.id: home_player,
            away_player.id: away_player,
        },
        matches=[match],
        leagues_list=[],
        meta={
            "snapshot_version": (SNAPSHOT_VERSION),
        },
    )


def test_serialized_model_module_paths_are_stable() -> None:
    serialized_classes = (
        FSDataBundle,
        FSCompSeason,
        FSTeam,
        FSPlayer,
        FSMatch,
    )

    assert {cls.__module__ for cls in serialized_classes} == {MODEL_MODULE}


def test_snapshot_round_trip_preserves_types_and_links(
    tmp_path,
) -> None:
    snapshot_path = tmp_path / "snapshot.pkl"

    save_snapshot(
        build_test_bundle(),
        snapshot_path,
    )
    loaded = load_snapshot(snapshot_path)

    assert isinstance(
        loaded,
        FSDataBundle,
    )
    assert isinstance(
        loaded.comp_seasons[100],
        FSCompSeason,
    )
    assert isinstance(
        loaded.teams[101],
        FSTeam,
    )
    assert isinstance(
        loaded.players[201],
        FSPlayer,
    )
    assert isinstance(
        loaded.matches[0],
        FSMatch,
    )

    loaded_match = loaded.matches[0]

    assert loaded_match.home_team is loaded.teams[101]
    assert loaded_match.away_team is loaded.teams[102]
    assert loaded_match.home_lineup[0] is loaded.players[201]
    assert loaded_match.away_lineup[0] is loaded.players[202]

    loaded_competition = loaded.comp_seasons[100]
    assert hasattr(
        loaded_competition,
        "conn",
    )


def test_snapshot_version_is_enforced(
    tmp_path,
) -> None:
    snapshot_path = tmp_path / "incompatible.pkl"
    bundle = build_test_bundle()
    bundle.meta["snapshot_version"] = SNAPSHOT_VERSION + 1

    with snapshot_path.open("wb") as file:
        pickle.dump(
            bundle,
            file,
            protocol=(pickle.HIGHEST_PROTOCOL),
        )

    with pytest.raises(
        RuntimeError,
        match=("Incompatible snapshot " "version"),
    ):
        load_snapshot(snapshot_path)


def test_legacy_bundle_state_backfills_optional_fields() -> None:
    legacy_state = {
        "comp_seasons": {},
        "teams": {},
        "players": {},
        "matches": [],
        "leagues_list": [],
        "meta": {},
    }

    bundle = object.__new__(FSDataBundle)
    bundle.__setstate__(legacy_state)

    assert bundle.sofifa_snapshots == []
    assert bundle.sofifa_player_occurrences == {}
    assert bundle.sofifa_players_by_dob == {}
    assert bundle.fs_to_sofifa_cache == {}
    assert bundle.sofifa_team_meta == {}
    assert bundle.sofifa_players_by_team == {}
    assert bundle.sofifa_teams_by_league == {}
    assert bundle.fs_team_to_sofifa_team == {}


def test_global_singleton_contract_is_characterized() -> None:
    first = Global.get_instance()
    second = Global.get_instance()

    assert first is second

    expected_attributes = {
        "all_matches",
        "all_comp_seasons",
        "all_players",
        "all_teams",
        "leagues_list",
        "sf_avg_team_strength",
        "sofifa_snapshots",
        "sofifa_player_occurrences",
        "sofifa_players_by_dob",
        "fs_to_sofifa_cache",
        "sofifa_team_meta",
        "sofifa_players_by_team",
        "sofifa_teams_by_league",
        "fs_team_to_sofifa_team",
    }

    assert expected_attributes.issubset(vars(first))
