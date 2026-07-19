from datetime import date
from pathlib import Path
from types import SimpleNamespace

from football_outcomes.data import (
    sofifa_team_matching,
)
from football_outcomes.utils import (
    fs_player_skill_utils,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def install_global(
    monkeypatch,
    state,
) -> None:
    monkeypatch.setattr(
        sofifa_team_matching,
        "Global",
        SimpleNamespace(get_instance=lambda: state),
    )


def test_team_module_has_no_legacy_or_matrix_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "sofifa_team_matching.py"
    source = source_path.read_text(encoding="utf-8")

    assert "fs_player_skill_utils" not in source
    assert "team_strength_matrix" not in source
    assert "sofifa_skills" not in source
    assert "lineups" not in source


def test_legacy_team_exports_are_direct_aliases() -> None:
    assert fs_player_skill_utils._norm_country is sofifa_team_matching._norm_country
    assert fs_player_skill_utils._norm_league is sofifa_team_matching._norm_league
    assert fs_player_skill_utils._norm_team is sofifa_team_matching._norm_team
    assert fs_player_skill_utils._extract_sofifa_team_info is sofifa_team_matching._extract_sofifa_team_info
    assert fs_player_skill_utils.build_sofifa_team_indexes is sofifa_team_matching.build_sofifa_team_indexes
    assert fs_player_skill_utils.match_fs_teams_to_sofifa_teams is sofifa_team_matching.match_fs_teams_to_sofifa_teams


def test_team_info_extraction_is_preserved() -> None:
    record = {
        "club_id": "20",
        "club_name": {"name": "Test Club"},
        "club_league_id": "13",
        "club_league_name": {"name": "Test League"},
    }

    assert sofifa_team_matching.extract_sofifa_team_info(record) == (
        20,
        "Test Club",
        13,
        "Test League",
    )

    invalid = {
        "club_id": "invalid",
        "club_name": "",
        "club_league_id": "invalid",
        "club_league_name": None,
    }

    assert sofifa_team_matching.extract_sofifa_team_info(invalid) == (
        None,
        None,
        None,
        None,
    )


def test_team_indexes_are_built_deterministically(
    monkeypatch,
) -> None:
    player_dob = date(
        1995,
        5,
        20,
    )

    state = SimpleNamespace(
        sofifa_snapshots=[
            (
                date(2024, 1, 1),
                {
                    1: {
                        "name": "Player One",
                        "full_name": ("Player One"),
                        "dob": player_dob,
                        "club_id": 20,
                        "club_name": ("Zulu Club"),
                        "club_league_id": 13,
                        "club_league_name": ("Test League"),
                    },
                    2: {
                        "name": "Player Two",
                        "full_name": ("Player Two"),
                        "dob": None,
                        "club_id": 10,
                        "club_name": ("Alpha Club"),
                        "club_league_id": 13,
                        "club_league_name": ("Test League"),
                    },
                },
            ),
            (
                date(2024, 2, 1),
                {
                    1: {
                        "name": "Player One",
                        "full_name": ("Player One"),
                        "dob": player_dob,
                        "club_id": 20,
                        "club_name": ("Zulu Club"),
                        "club_league_id": 13,
                        "club_league_name": ("Test League"),
                    },
                    3: {
                        "name": ("Player Three"),
                        "full_name": ("Player Three"),
                        "dob": None,
                        "club_id": 20,
                        "club_name": ("Zulu Club"),
                        "club_league_id": 13,
                        "club_league_name": ("Test League"),
                    },
                },
            ),
        ],
        sofifa_players_by_team={},
        sofifa_team_meta={},
        sofifa_teams_by_league={},
    )

    install_global(
        monkeypatch,
        state,
    )
    monkeypatch.setattr(
        sofifa_team_matching.sett,
        "DEBUG_TEAM_STRENGTH",
        False,
    )

    sofifa_team_matching.build_sofifa_team_indexes(force=True)

    assert [row[0] for row in state.sofifa_players_by_team[20]] == [1, 3]

    assert [row[0] for row in state.sofifa_players_by_team[10]] == [2]

    assert state.sofifa_team_meta[20]["league_ids"] == {13}

    assert state.sofifa_teams_by_league[13] == [
        (
            10,
            "Alpha Club",
        ),
        (
            20,
            "Zulu Club",
        ),
    ]


def test_manual_and_automatic_team_mapping_are_preserved(
    monkeypatch,
) -> None:
    match = SimpleNamespace(
        home_team=SimpleNamespace(
            id=1,
            name="Excluded Club",
        ),
        away_team=SimpleNamespace(
            id=2,
            name="Manchester Utd",
        ),
        country="England",
        comp_name="Test League",
    )

    state = SimpleNamespace(
        all_matches=[match],
        sofifa_players_by_team={
            200: [
                (
                    500,
                    "Player",
                    "Player",
                    None,
                )
            ]
        },
        sofifa_team_meta={
            200: {
                "name": ("Manchester United"),
                "league_ids": {13},
            }
        },
        sofifa_teams_by_league={
            13: [
                (
                    200,
                    "Manchester United",
                ),
                (
                    201,
                    "Liverpool",
                ),
            ]
        },
        fs_team_to_sofifa_team={},
    )

    install_global(
        monkeypatch,
        state,
    )

    monkeypatch.setattr(
        sofifa_team_matching.sett,
        "COMPS_LEAGUE",
        ["Test League"],
    )
    monkeypatch.setattr(
        sofifa_team_matching.sett,
        "FS_TEAM_ID_TO_SOFIFA_TEAM_ID",
        {1: -1},
    )
    monkeypatch.setattr(
        sofifa_team_matching.sett,
        "FS_LEAGUE_TO_SOFIFA_LEAGUE_ID",
        {"Test League": 13},
    )
    monkeypatch.setattr(
        sofifa_team_matching.sett,
        "DEBUG_TEAM_STRENGTH",
        False,
    )

    sofifa_team_matching.match_fs_teams_to_sofifa_teams(force=True)

    assert state.fs_team_to_sofifa_team[1] == -1
    assert state.fs_team_to_sofifa_team[2] == 200


def test_application_pipeline_uses_extracted_team_service() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "application" / "footystats_pipeline.py"
    source = source_path.read_text(encoding="utf-8")

    assert "football_outcomes.data." "sofifa_team_matching" in source
    assert "match_fs_teams_to_sofifa_teams" in source
    assert "football_outcomes.utils." "fs_player_skill_utils" not in source
