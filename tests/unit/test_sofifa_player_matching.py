from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_outcomes.data import (
    sofifa_player_matching,
)
from football_outcomes.data.fs_models import (
    FSPlayer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_player(
    player_id: int,
    name: str,
    *,
    birthday=None,
) -> FSPlayer:
    player = FSPlayer(
        player_id,
        name,
        "",
        "",
        name,
        name,
    )
    player.birthday = birthday
    return player


def install_global(
    monkeypatch,
    state,
) -> None:
    monkeypatch.setattr(
        sofifa_player_matching,
        "Global",
        SimpleNamespace(get_instance=lambda: state),
    )


def test_matching_module_has_no_matrix_or_skill_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "sofifa_player_matching.py"
    source = source_path.read_text(encoding="utf-8")

    assert "fs_player_skill_utils" not in source
    assert "team_strength_matrix" not in source
    assert "sofifa_skills" not in source
    assert "lineups" not in source


def test_name_normalization_and_similarity_are_preserved() -> None:
    assert sofifa_player_matching.normalize_name("  Jean-Luc O'Neil  ") == "jean luc o neil"

    assert sofifa_player_matching.name_key_last_first_initial("kevin de bruyne") == "bruyne|k"

    assert sofifa_player_matching.similarity(
        "john doe",
        "john doe",
        "john",
        "john doe",
    ) == pytest.approx(100.0)


def test_team_and_dob_match_is_cached(
    monkeypatch,
) -> None:
    player_dob = date(
        1995,
        5,
        20,
    )
    player = make_player(
        10,
        "John Doe",
        birthday=player_dob,
    )

    state = SimpleNamespace(
        fs_to_sofifa_cache={},
        fs_team_to_sofifa_team={100: 200},
        sofifa_players_by_team={
            200: [
                (
                    500,
                    "John Doe",
                    "John Doe",
                    player_dob,
                )
            ]
        },
        sofifa_players_by_dob={
            player_dob: [
                (
                    500,
                    "John Doe",
                    "John Doe",
                )
            ]
        },
    )

    install_global(
        monkeypatch,
        state,
    )

    monkeypatch.setattr(
        sofifa_player_matching.sett,
        "USE_FS_TO_SOFIFA_CACHE",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        sofifa_player_matching.sett,
        "SF_MATCH_LOW_THRESHOLD",
        40.0,
        raising=False,
    )

    result = sofifa_player_matching.match_fs_to_sofifa(
        player,
        fs_team_id=100,
    )

    assert result.sofifa_id == 500
    assert result.score_best == 100.0
    assert result.used_dob_gate is True
    assert result.reason == ("team_dob_pass")
    assert result.sofifa_best_name == ("John Doe")

    assert state.fs_to_sofifa_cache[player.id] == (
        500,
        100.0,
        -1.0,
        True,
        "team_dob_pass",
        "John Doe",
    )


def test_legacy_five_field_success_cache_is_supported(
    monkeypatch,
) -> None:
    player = make_player(
        20,
        "Cached Player",
    )

    state = SimpleNamespace(
        fs_to_sofifa_cache={
            player.id: (
                600,
                95.0,
                75.0,
                True,
                "dob_gate_pass",
            )
        }
    )

    install_global(
        monkeypatch,
        state,
    )

    monkeypatch.setattr(
        sofifa_player_matching.sett,
        "USE_FS_TO_SOFIFA_CACHE",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        sofifa_player_matching.sett,
        ("FS_TO_SOFIFA_" "CACHE_RETRY_AMBIGUOUS"),
        False,
        raising=False,
    )
    monkeypatch.setattr(
        sofifa_player_matching.sett,
        ("FS_TO_SOFIFA_" "CACHE_MIN_MARGIN"),
        0.0,
        raising=False,
    )
    monkeypatch.setattr(
        sofifa_player_matching.sett,
        ("FS_TO_SOFIFA_" "CACHE_ONLY_TRUST_REASONS"),
        {"dob_gate_pass"},
        raising=False,
    )

    result = sofifa_player_matching.match_fs_to_sofifa(player)

    assert result.sofifa_id == 600
    assert result.score_best == 95.0
    assert result.score_second == 75.0
    assert result.reason == ("cache:dob_gate_pass")
    assert result.sofifa_best_name is None


def test_failed_cache_can_be_reused_without_retry(
    monkeypatch,
) -> None:
    player = make_player(
        30,
        "Unmatched Player",
    )

    state = SimpleNamespace(
        fs_to_sofifa_cache={
            player.id: (
                None,
                55.0,
                50.0,
                False,
                "name_bucket_fail",
                "Closest Candidate",
            )
        }
    )

    install_global(
        monkeypatch,
        state,
    )

    monkeypatch.setattr(
        sofifa_player_matching.sett,
        "USE_FS_TO_SOFIFA_CACHE",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        sofifa_player_matching.sett,
        ("FS_TO_SOFIFA_" "CACHE_RETRY_FAILED"),
        False,
        raising=False,
    )

    result = sofifa_player_matching.match_fs_to_sofifa(player)

    assert result.sofifa_id is None
    assert result.reason == ("cache:name_bucket_fail")
    assert result.sofifa_best_name == ("Closest Candidate")


def test_name_bucket_is_final_fallback(
    monkeypatch,
) -> None:
    player = make_player(
        40,
        "Jane Smith",
    )

    unrelated_dob = date(
        1990,
        1,
        1,
    )

    state = SimpleNamespace(
        fs_to_sofifa_cache={},
        fs_team_to_sofifa_team={},
        sofifa_players_by_team={},
        sofifa_players_by_dob={
            unrelated_dob: [
                (
                    700,
                    "Jane Smith",
                    "Jane Smith",
                )
            ]
        },
    )

    install_global(
        monkeypatch,
        state,
    )

    monkeypatch.setattr(
        sofifa_player_matching.sett,
        "USE_FS_TO_SOFIFA_CACHE",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        sofifa_player_matching.sett,
        "SF_NAME_BUCKET_MAX",
        200,
        raising=False,
    )
    monkeypatch.setattr(
        sofifa_player_matching.sett,
        "SF_MATCH_HIGH_THRESHOLD",
        70.0,
        raising=False,
    )

    result = sofifa_player_matching.match_fs_to_sofifa(player)

    assert result.sofifa_id == 700
    assert result.score_best == 100.0
    assert result.used_dob_gate is False
    assert result.reason == ("name_bucket_pass")

    assert state.sofifa_players_by_namekey["smith|j"][0][0] == 700
