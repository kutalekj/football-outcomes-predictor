from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_outcomes.config import fs_settings as settings
from football_outcomes.data import (
    sofifa_player_matching,
    sofifa_skills,
    team_strength_matrix,
)
from football_outcomes.data.fs_models import (
    FSMatch,
    FSPlayer,
    FSTeam,
)
from football_outcomes.utils import (
    fs_player_skill_utils,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_player(
    player_id: int,
    position: str,
) -> FSPlayer:
    player = FSPlayer(
        player_id,
        f"Player {player_id}",
        "First",
        "Last",
        f"P{player_id}",
        f"Player {player_id}",
    )
    player.position = position
    return player


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
    lineup: list[FSPlayer],
) -> FSMatch:
    match = FSMatch(100)
    match.home_team = make_team(1)
    match.away_team = make_team(2)
    match.home_lineup = lineup
    match.away_lineup = []
    match.datetime = datetime(
        2024,
        1,
        10,
        12,
    )
    match.comp_name = "Test League"
    return match


def test_matrix_module_has_no_legacy_matching_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "team_strength_matrix.py"
    source = source_path.read_text(encoding="utf-8")

    assert "football_outcomes.utils." "fs_player_skill_utils" not in source
    assert "fs_globals" not in source
    assert "rapidfuzz" not in source


def test_legacy_matrix_exports_are_direct_aliases() -> None:
    assert fs_player_skill_utils._gk_role_score is team_strength_matrix.goalkeeper_role_score
    assert fs_player_skill_utils._ensure_one_goalkeeper_row is team_strength_matrix.ensure_one_goalkeeper_row
    assert fs_player_skill_utils.calculate_team_strength is team_strength_matrix.calculate_team_strength


def test_goalkeeper_role_score_ignores_missing_values(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "GK_SKILL_START_INDEX",
        2,
    )
    monkeypatch.setattr(
        settings,
        "GK_SKILL_END_INDEX",
        4,
    )

    score = team_strength_matrix.goalkeeper_role_score(
        [
            10.0,
            -1.0,
            30.0,
            50.0,
        ]
    )

    assert score == pytest.approx(30.0)

    assert (
        team_strength_matrix.goalkeeper_role_score(
            [
                -1.0,
                -1.0,
                30.0,
                50.0,
            ]
        )
        == 0.0
    )


def test_explicit_goalkeeper_is_moved_to_front(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "FORCE_EXACTLY_ONE_GK_ROW",
        True,
    )

    defender = make_player(
        10,
        "Defender",
    )
    goalkeeper = make_player(
        11,
        "Goalkeeper",
    )

    rows = [
        (
            defender,
            [1.0, 2.0],
        ),
        (
            goalkeeper,
            [3.0, 4.0],
        ),
    ]

    result = team_strength_matrix.ensure_one_goalkeeper_row(rows)

    assert result is rows
    assert result[0][0] is goalkeeper
    assert result[1][0] is defender


def test_goalkeeper_signature_or_missing_row_is_used(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "FORCE_EXACTLY_ONE_GK_ROW",
        True,
    )
    monkeypatch.setattr(
        settings,
        "GK_SKILL_START_INDEX",
        2,
    )
    monkeypatch.setattr(
        settings,
        "GK_SKILL_END_INDEX",
        4,
    )
    monkeypatch.setattr(
        settings,
        "GK_ROLE_SCORE_MIN_GAP",
        0.5,
    )
    monkeypatch.setattr(
        settings,
        "PLAYER_SKILLS",
        [
            "outfield_1",
            "outfield_2",
            "gk_1",
            "gk_2",
        ],
    )

    goalkeeper_like = make_player(
        20,
        "Defender",
    )
    outfield_like = make_player(
        21,
        "Forward",
    )

    rows = [
        (
            outfield_like,
            [
                70.0,
                70.0,
                10.0,
                10.0,
            ],
        ),
        (
            goalkeeper_like,
            [
                10.0,
                10.0,
                80.0,
                80.0,
            ],
        ),
    ]

    result = team_strength_matrix.ensure_one_goalkeeper_row(rows)

    assert result[0][0] is (goalkeeper_like)

    missing_rows = [
        (
            outfield_like,
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ],
        )
    ]

    missing_result = team_strength_matrix.ensure_one_goalkeeper_row(missing_rows)

    assert missing_result[0][0].id == -1
    assert missing_result[0][0].position == "Goalkeeper"
    assert missing_result[0][1] == [
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    ]


def test_explicit_core_sorts_matches_and_pads(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "TEAM_STRENGTH_NUM_PLAYERS",
        4,
    )
    monkeypatch.setattr(
        settings,
        "PLAYER_SKILLS",
        [
            "outfield",
            "goalkeeper",
        ],
    )
    monkeypatch.setattr(
        settings,
        "FORCE_EXACTLY_ONE_GK_ROW",
        True,
    )
    monkeypatch.setattr(
        settings,
        "DEBUG_TEAM_STRENGTH",
        False,
    )

    forward = make_player(
        30,
        "Forward",
    )
    goalkeeper = make_player(
        31,
        "Goalkeeper",
    )

    match = make_match(
        [
            forward,
            goalkeeper,
        ]
    )

    matcher_calls = []
    loader_calls = []

    def match_player(
        player,
        *,
        fs_team_id=None,
    ):
        matcher_calls.append(
            (
                player.id,
                fs_team_id,
            )
        )

        return SimpleNamespace(
            sofifa_id=player.id + 1000,
            score_best=90.0,
            score_second=70.0,
            used_dob_gate=True,
            reason="test",
            sofifa_best_name=(f"SoFIFA {player.id}"),
        )

    def merge_skills(
        sofifa_id,
        match_datetime,
    ):
        loader_calls.append(
            (
                sofifa_id,
                match_datetime,
            )
        )

        return (
            [
                float(sofifa_id),
                float(sofifa_id + 1),
            ],
            1,
            0,
        )

    matrix = team_strength_matrix.build_team_strength_matrix(
        match=match,
        team_id=match.home_team.id,
        match_player=match_player,
        merge_skills=merge_skills,
        debug_log=lambda message: None,
        player_display_name=(lambda player: player.known_as),
    )

    assert matcher_calls == [
        (
            goalkeeper.id,
            match.home_team.id,
        ),
        (
            forward.id,
            match.home_team.id,
        ),
    ]

    assert [call[0] for call in loader_calls] == [
        goalkeeper.id + 1000,
        forward.id + 1000,
    ]

    assert matrix == [
        [
            float(goalkeeper.id + 1000),
            float(goalkeeper.id + 1001),
        ],
        [
            float(forward.id + 1000),
            float(forward.id + 1001),
        ],
        [
            -1.0,
            -1.0,
        ],
        [
            -1.0,
            -1.0,
        ],
    ]


def test_compatibility_entry_point_wires_extracted_dependencies(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "TEAM_STRENGTH_NUM_PLAYERS",
        2,
    )
    monkeypatch.setattr(
        settings,
        "PLAYER_SKILLS",
        [
            "skill_1",
            "skill_2",
        ],
    )
    monkeypatch.setattr(
        settings,
        "FORCE_EXACTLY_ONE_GK_ROW",
        False,
    )
    monkeypatch.setattr(
        settings,
        "DEBUG_TEAM_STRENGTH",
        False,
    )

    player = make_player(
        40,
        "Forward",
    )
    match = make_match([player])

    matcher_calls = []
    loader_calls = []

    def fake_matcher(
        supplied_player,
        *,
        fs_team_id=None,
    ):
        matcher_calls.append(
            (
                supplied_player,
                fs_team_id,
            )
        )

        return SimpleNamespace(
            sofifa_id=500,
            score_best=99.0,
            score_second=80.0,
            used_dob_gate=True,
            reason="test",
            sofifa_best_name=("Matched Player"),
        )

    def fake_loader(
        sofifa_id,
        match_datetime,
    ):
        loader_calls.append(
            (
                sofifa_id,
                match_datetime,
            )
        )
        return (
            [
                55.0,
                66.0,
            ],
            1,
            0,
        )

    monkeypatch.setattr(
        sofifa_player_matching,
        "match_fs_to_sofifa",
        fake_matcher,
    )
    monkeypatch.setattr(
        sofifa_skills,
        "merge_skills_from_snapshots",
        fake_loader,
    )

    matrix = team_strength_matrix.calculate_team_strength(
        match,
        match.home_team.id,
    )

    assert matcher_calls == [
        (
            player,
            match.home_team.id,
        )
    ]
    assert loader_calls == [
        (
            500,
            match.datetime,
        )
    ]

    assert matrix == [
        [
            55.0,
            66.0,
        ],
        [
            -1.0,
            -1.0,
        ],
    ]


def test_match_feature_service_uses_matrix_module() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "match_features.py"
    source = source_path.read_text(encoding="utf-8")

    assert "football_outcomes.data." "team_strength_matrix" in source
    assert "football_outcomes.utils." "fs_player_skill_utils import" not in source
