from pathlib import Path

import pytest

from football_outcomes.config import fs_settings as settings
from football_outcomes.data import (
    lineups,
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


def make_match() -> FSMatch:
    match = FSMatch(100)
    match.home_team = make_team(1)
    match.away_team = make_team(2)
    return match


def test_lineup_module_has_no_global_or_matching_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "lineups.py"
    source = source_path.read_text(encoding="utf-8")

    assert "fs_globals" not in source
    assert "rapidfuzz" not in source
    assert "fs_player_skill_utils" not in source
    assert "sofifa" not in source.lower()


def test_legacy_lineup_exports_are_direct_aliases() -> None:
    assert fs_player_skill_utils._FS_POS_ORDER is lineups.FS_POSITION_ORDER
    assert fs_player_skill_utils._pos_rank is lineups.position_rank
    assert fs_player_skill_utils._select_and_sort_lineup is lineups.select_and_sort_lineup
    assert fs_player_skill_utils.calculate_team_position_indices is lineups.calculate_team_position_indices


def test_missing_goalkeeper_and_padding_are_preserved() -> None:
    match = make_match()

    forward = make_player(
        11,
        "Forward",
    )
    defender = make_player(
        12,
        "Defender",
    )
    midfielder = make_player(
        13,
        "Midfielder",
    )

    original_lineup = [
        forward,
        defender,
        midfielder,
    ]
    match.home_lineup = original_lineup

    lineup_sorted, side = lineups.select_and_sort_lineup(
        match,
        match.home_team.id,
    )

    assert side == "home"
    assert match.home_lineup is original_lineup
    assert match.home_lineup == [
        forward,
        defender,
        midfielder,
    ]

    assert len(lineup_sorted) == (settings.TEAM_STRENGTH_NUM_PLAYERS)

    assert [player.position for player in lineup_sorted[:4]] == [
        "Goalkeeper",
        "Defender",
        "Midfielder",
        "Forward",
    ]

    assert all(player.position == "Unknown" for player in lineup_sorted[4:])

    expected_positions = [
        "Goalkeeper",
        "Defender",
        "Midfielder",
        "Forward",
    ] + [
        "Unknown"
    ] * (settings.TEAM_STRENGTH_NUM_PLAYERS - 4)

    expected_indices = [int(settings.FS_PLAYER_POSITION_TO_IDX[position]) for position in expected_positions]

    assert (
        lineups.calculate_team_position_indices(
            match,
            match.home_team.id,
        )
        == expected_indices
    )


def test_existing_goalkeeper_and_away_selection_are_preserved() -> None:
    match = make_match()

    goalkeeper = make_player(
        21,
        "Goalkeeper",
    )
    defender = make_player(
        22,
        "Defender",
    )
    forward = make_player(
        23,
        "Forward",
    )

    match.away_lineup = [
        forward,
        goalkeeper,
        defender,
    ]

    lineup_sorted, side = lineups.select_and_sort_lineup(
        match,
        match.away_team.id,
    )

    assert side == "away"
    assert lineup_sorted[0] is goalkeeper
    assert lineup_sorted[1] is defender
    assert lineup_sorted[2] is forward

    goalkeeper_rows = [player for player in lineup_sorted if player.position == "Goalkeeper"]
    assert goalkeeper_rows == [goalkeeper]


def test_invalid_team_and_oversized_lineup_are_rejected() -> None:
    match = make_match()

    with pytest.raises(
        ValueError,
        match="not in match",
    ):
        lineups.select_and_sort_lineup(
            match,
            999,
        )

    match.home_lineup = [
        make_player(
            player_id,
            "Defender",
        )
        for player_id in range(settings.TEAM_STRENGTH_NUM_PLAYERS + 1)
    ]

    with pytest.raises(
        ValueError,
        match="Lineup has >",
    ):
        lineups.select_and_sort_lineup(
            match,
            match.home_team.id,
        )


def test_active_arrays_use_extracted_lineup_service() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "datasets" / "arrays.py"
    source = source_path.read_text(encoding="utf-8")

    assert "football_outcomes.data.lineups" in source
    assert "football_outcomes.utils." "fs_player_skill_utils" not in source
