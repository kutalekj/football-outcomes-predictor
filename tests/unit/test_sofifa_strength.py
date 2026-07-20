from __future__ import annotations

from copy import deepcopy
from datetime import (
    date,
    datetime,
)
from pathlib import Path

import pytest

from football_outcomes.config import fs_settings as settings
from football_outcomes.data.fs_models import (
    FSMatch,
    FSPlayer,
    FSTeam,
)
from football_outcomes.data.lineups import (
    calculate_team_position_indices,
)
from football_outcomes.data.sofifa_strength import (
    UNRESOLVED_SOURCE_AGE_DAYS,
    PastOnlyStrengthConfig,
    cached_sofifa_id,
    reconstruct_past_only_match_strength,
    reconstruct_past_only_team_strength,
)
from football_outcomes.data.sofifa_temporal import (
    SkillProvenance,
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


def make_match() -> FSMatch:
    match = FSMatch(100)
    match.home_team = make_team(1)
    match.away_team = make_team(2)
    match.datetime = datetime(
        2024,
        1,
        10,
        12,
    )
    match.comp_name = "Test League"
    return match


def make_config(
    *,
    player_count: int = 3,
    skill_count: int = 2,
) -> PastOnlyStrengthConfig:
    return PastOnlyStrengthConfig(
        player_count=player_count,
        skill_count=skill_count,
        max_age_days=30,
        max_snapshots=3,
    )


@pytest.fixture
def three_player_rows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "TEAM_STRENGTH_NUM_PLAYERS",
        3,
    )


def test_strength_module_is_offline_and_nonlegacy() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "sofifa_strength.py"
    source = source_path.read_text(encoding="utf-8")

    assert "requests" not in source
    assert "fs_globals" not in source
    assert "Global" not in source
    assert "sofifa_player_matching" not in source
    assert "team_strength_matrix" not in source


@pytest.mark.parametrize(
    (
        "record",
        "expected",
    ),
    [
        (
            (
                500,
                99.0,
                80.0,
                True,
                "pass",
            ),
            500,
        ),
        (
            (
                600,
                99.0,
                80.0,
                True,
                "pass",
                "Player",
            ),
            600,
        ),
        (
            (
                None,
                50.0,
                40.0,
                False,
                "fail",
            ),
            None,
        ),
        (
            (),
            None,
        ),
        (
            "invalid",
            None,
        ),
    ],
)
def test_cached_sofifa_id(
    record,
    expected,
) -> None:
    assert (
        cached_sofifa_id(
            {10: record},
            10,
        )
        == expected
    )


def test_team_rows_align_with_lineup_positions(
    three_player_rows,
) -> None:
    match = make_match()

    goalkeeper = make_player(
        10,
        "Goalkeeper",
    )
    defender = make_player(
        11,
        "Defender",
    )
    forward = make_player(
        12,
        "Forward",
    )

    match.home_lineup = [
        forward,
        goalkeeper,
        defender,
    ]

    snapshots = [
        (
            date(2024, 1, 9),
            {
                100: {
                    "skills": [
                        10.0,
                        11.0,
                    ]
                },
                101: {
                    "skills": [
                        20.0,
                        -1.0,
                    ]
                },
            },
        ),
        (
            date(2024, 1, 8),
            {
                101: {
                    "skills": [
                        21.0,
                        22.0,
                    ]
                }
            },
        ),
    ]

    occurrences = {
        100: [
            (
                0,
                date(2024, 1, 9),
            )
        ],
        101: [
            (
                0,
                date(2024, 1, 9),
            ),
            (
                1,
                date(2024, 1, 8),
            ),
        ],
    }

    result = reconstruct_past_only_team_strength(
        match=match,
        team_id=match.home_team.id,
        snapshots=snapshots,
        player_occurrences=occurrences,
        fs_to_sofifa_cache={
            10: (
                100,
                99.0,
                80.0,
                True,
                "pass",
            ),
            11: (
                101,
                99.0,
                80.0,
                True,
                "pass",
            ),
            12: (
                None,
                50.0,
                40.0,
                False,
                "fail",
            ),
        },
        config=make_config(),
    )

    assert result.side == "home"
    assert result.fs_player_ids == (
        10,
        11,
        12,
    )
    assert result.sofifa_player_ids == (
        100,
        101,
        None,
    )
    assert result.position_indices == tuple(
        calculate_team_position_indices(
            match,
            match.home_team.id,
        )
    )

    assert result.skills == (
        (
            10.0,
            11.0,
        ),
        (
            20.0,
            22.0,
        ),
        (
            -1.0,
            -1.0,
        ),
    )
    assert result.provenance == (
        (
            SkillProvenance.NEAREST_PAST_SOFIFA,
            SkillProvenance.NEAREST_PAST_SOFIFA,
        ),
        (
            SkillProvenance.NEAREST_PAST_SOFIFA,
            SkillProvenance.OLDER_PAST_SOFIFA,
        ),
        (
            SkillProvenance.UNRESOLVED,
            SkillProvenance.UNRESOLVED,
        ),
    )
    assert result.source_age_days == (
        (
            1,
            1,
        ),
        (
            1,
            2,
        ),
        (
            -1,
            -1,
        ),
    )

    assert result.observed_count == 4
    assert result.nearest_past_count == 3
    assert result.older_past_count == 1
    assert result.unresolved_count == 2
    assert result.matched_player_rows == 2
    assert result.unmatched_player_rows == 1


def test_future_snapshot_is_unresolved_at_matrix_boundary(
    three_player_rows,
) -> None:
    match = make_match()

    goalkeeper = make_player(
        20,
        "Goalkeeper",
    )
    defender = make_player(
        21,
        "Defender",
    )
    forward = make_player(
        22,
        "Forward",
    )

    match.home_lineup = [
        goalkeeper,
        defender,
        forward,
    ]

    snapshots = [
        (
            date(2024, 1, 11),
            {
                200: {
                    "skills": [
                        90.0,
                        91.0,
                    ]
                }
            },
        )
    ]

    result = reconstruct_past_only_team_strength(
        match=match,
        team_id=match.home_team.id,
        snapshots=snapshots,
        player_occurrences={
            200: [
                (
                    0,
                    date(2024, 1, 9),
                )
            ]
        },
        fs_to_sofifa_cache={
            20: (
                200,
                99.0,
                80.0,
                True,
                "pass",
            )
        },
        config=make_config(),
    )

    assert result.skills[0] == (
        -1.0,
        -1.0,
    )
    assert result.source_age_days[0] == (
        UNRESOLVED_SOURCE_AGE_DAYS,
        UNRESOLVED_SOURCE_AGE_DAYS,
    )
    assert result.observed_count == 0


def test_synthetic_and_padded_rows_are_unresolved(
    three_player_rows,
) -> None:
    match = make_match()

    forward = make_player(
        30,
        "Forward",
    )
    match.home_lineup = [forward]

    result = reconstruct_past_only_team_strength(
        match=match,
        team_id=match.home_team.id,
        snapshots=[
            (
                date(2024, 1, 9),
                {
                    300: {
                        "skills": [
                            60.0,
                            70.0,
                        ]
                    }
                },
            )
        ],
        player_occurrences={
            300: [
                (
                    0,
                    date(2024, 1, 9),
                )
            ]
        },
        fs_to_sofifa_cache={
            30: (
                300,
                99.0,
                80.0,
                True,
                "pass",
            )
        },
        config=make_config(),
    )

    assert result.fs_player_ids == (
        -1,
        30,
        -1,
    )
    assert result.sofifa_player_ids == (
        None,
        300,
        None,
    )
    assert result.skills == (
        (
            -1.0,
            -1.0,
        ),
        (
            60.0,
            70.0,
        ),
        (
            -1.0,
            -1.0,
        ),
    )
    assert result.unresolved_count == 4


def test_match_reconstruction_does_not_mutate_inputs(
    three_player_rows,
) -> None:
    match = make_match()

    match.home_lineup = [
        make_player(
            40,
            "Goalkeeper",
        ),
        make_player(
            41,
            "Defender",
        ),
        make_player(
            42,
            "Forward",
        ),
    ]
    match.away_lineup = [
        make_player(
            50,
            "Goalkeeper",
        ),
        make_player(
            51,
            "Defender",
        ),
        make_player(
            52,
            "Forward",
        ),
    ]

    snapshots = [
        (
            date(2024, 1, 9),
            {
                400: {
                    "skills": [
                        50.0,
                        51.0,
                    ]
                },
                500: {
                    "skills": [
                        60.0,
                        61.0,
                    ]
                },
            },
        )
    ]
    occurrences = {
        400: [
            (
                0,
                date(2024, 1, 9),
            )
        ],
        500: [
            (
                0,
                date(2024, 1, 9),
            )
        ],
    }
    cache = {
        40: (
            400,
            99.0,
            80.0,
            True,
            "pass",
        ),
        50: (
            500,
            99.0,
            80.0,
            True,
            "pass",
        ),
    }

    home_before = list(match.home_lineup)
    away_before = list(match.away_lineup)
    snapshots_before = deepcopy(snapshots)
    occurrences_before = deepcopy(occurrences)
    cache_before = deepcopy(cache)

    first = reconstruct_past_only_match_strength(
        match=match,
        snapshots=snapshots,
        player_occurrences=occurrences,
        fs_to_sofifa_cache=cache,
        config=make_config(),
    )
    second = reconstruct_past_only_match_strength(
        match=match,
        snapshots=snapshots,
        player_occurrences=occurrences,
        fs_to_sofifa_cache=cache,
        config=make_config(),
    )

    assert first == second
    assert first.match_id == match.id
    assert first.home.side == "home"
    assert first.away.side == "away"

    assert match.home_lineup == home_before
    assert match.away_lineup == away_before
    assert snapshots == snapshots_before
    assert occurrences == occurrences_before
    assert cache == cache_before


def test_missing_datetime_is_rejected(
    three_player_rows,
) -> None:
    match = make_match()
    match.datetime = None

    with pytest.raises(
        ValueError,
        match="has no datetime",
    ):
        reconstruct_past_only_team_strength(
            match=match,
            team_id=match.home_team.id,
            snapshots=[],
            player_occurrences={},
            fs_to_sofifa_cache={},
            config=make_config(),
        )


@pytest.mark.parametrize(
    (
        "keyword",
        "value",
        "message",
    ),
    [
        (
            "player_count",
            0,
            "player_count must be",
        ),
        (
            "skill_count",
            0,
            "skill_count must be",
        ),
        (
            "max_age_days",
            -1,
            "max_age_days must be",
        ),
        (
            "max_snapshots",
            0,
            "max_snapshots must be",
        ),
    ],
)
def test_invalid_configuration_is_rejected(
    keyword,
    value,
    message,
) -> None:
    arguments = {
        "player_count": 3,
        "skill_count": 2,
        "max_age_days": 30,
        "max_snapshots": 3,
    }
    arguments[keyword] = value

    with pytest.raises(
        ValueError,
        match=message,
    ):
        PastOnlyStrengthConfig(**arguments)
