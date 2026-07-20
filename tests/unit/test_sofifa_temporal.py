from __future__ import annotations

from copy import deepcopy
from datetime import (
    date,
    datetime,
)
from pathlib import Path

import pytest

from football_outcomes.data.sofifa_temporal import (
    MISSING_SKILL_VALUE,
    SkillProvenance,
    ordered_past_snapshot_candidates,
    reconstruct_past_only_skills,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_temporal_module_is_pure_and_offline() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "sofifa_temporal.py"
    source = source_path.read_text(encoding="utf-8")

    assert "fs_globals" not in source
    assert "Global" not in source
    assert "fs_settings" not in source
    assert "requests" not in source
    assert "sofifa_player_matching" not in source
    assert "team_strength_matrix" not in source


def test_provenance_codes_match_contract() -> None:
    assert [int(value) for value in SkillProvenance] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ]


def test_candidates_use_actual_snapshot_dates() -> None:
    snapshots = [
        (
            date(2024, 1, 11),
            {},
        ),
        (
            date(2024, 1, 1),
            {},
        ),
        (
            date(2024, 1, 9),
            {},
        ),
        (
            date(2023, 10, 1),
            {},
        ),
    ]

    occurrences = [
        (
            0,
            date(2024, 1, 1),
        ),
        (
            1,
            date(2024, 1, 1),
        ),
        (
            2,
            date(2024, 1, 9),
        ),
        (
            2,
            date(2024, 1, 9),
        ),
        (
            3,
            date(2023, 10, 1),
        ),
    ]

    candidates = ordered_past_snapshot_candidates(
        occurrences,
        snapshots,
        datetime(
            2024,
            1,
            10,
            12,
        ),
        max_age_days=30,
        max_snapshots=3,
    )

    assert [
        (
            candidate.snapshot_index,
            candidate.snapshot_date,
            candidate.age_days,
        )
        for candidate in candidates
    ] == [
        (
            2,
            date(2024, 1, 9),
            1,
        ),
        (
            1,
            date(2024, 1, 1),
            9,
        ),
    ]


def test_older_past_snapshot_completes_missing_skills() -> None:
    sofifa_id = 50

    snapshots = [
        (
            date(2024, 1, 9),
            {
                sofifa_id: {
                    "skills": [
                        10.0,
                        -1.0,
                        None,
                        40.0,
                    ]
                }
            },
        ),
        (
            date(2024, 1, 8),
            {
                sofifa_id: {
                    "skills": [
                        11.0,
                        20.0,
                        30.0,
                        41.0,
                    ]
                }
            },
        ),
        (
            date(2024, 1, 11),
            {
                sofifa_id: {
                    "skills": [
                        99.0,
                        99.0,
                        99.0,
                        99.0,
                    ]
                }
            },
        ),
    ]

    result = reconstruct_past_only_skills(
        sofifa_id=sofifa_id,
        match_datetime=datetime(
            2024,
            1,
            10,
            12,
        ),
        snapshots=snapshots,
        player_occurrences={
            sofifa_id: [
                (
                    0,
                    date(2024, 1, 9),
                ),
                (
                    1,
                    date(2024, 1, 8),
                ),
                (
                    2,
                    date(2024, 1, 11),
                ),
            ]
        },
        skill_count=4,
        max_age_days=30,
        max_snapshots=3,
    )

    assert result.skills == (
        10.0,
        20.0,
        30.0,
        40.0,
    )
    assert result.provenance == (
        SkillProvenance.NEAREST_PAST_SOFIFA,
        SkillProvenance.OLDER_PAST_SOFIFA,
        SkillProvenance.OLDER_PAST_SOFIFA,
        SkillProvenance.NEAREST_PAST_SOFIFA,
    )
    assert result.source_dates == (
        date(2024, 1, 9),
        date(2024, 1, 8),
        date(2024, 1, 8),
        date(2024, 1, 9),
    )
    assert result.snapshots_used == 2
    assert result.nearest_snapshot_delta_days == 1
    assert result.observed_count == 4
    assert result.nearest_past_count == 2
    assert result.older_past_count == 2
    assert result.unresolved_count == 0


def test_future_snapshot_is_never_used() -> None:
    sofifa_id = 60

    snapshots = [
        (
            date(2024, 1, 11),
            {
                sofifa_id: {
                    "skills": [
                        80.0,
                        81.0,
                    ]
                }
            },
        )
    ]

    result = reconstruct_past_only_skills(
        sofifa_id=sofifa_id,
        match_datetime=datetime(
            2024,
            1,
            10,
        ),
        snapshots=snapshots,
        player_occurrences={
            sofifa_id: [
                (
                    0,
                    date(2024, 1, 9),
                )
            ]
        },
        skill_count=2,
        max_age_days=30,
        max_snapshots=2,
    )

    assert result.skills == (
        MISSING_SKILL_VALUE,
        MISSING_SKILL_VALUE,
    )
    assert result.provenance == (
        SkillProvenance.UNRESOLVED,
        SkillProvenance.UNRESOLVED,
    )
    assert result.source_dates == (
        None,
        None,
    )
    assert result.snapshots_used == 0
    assert result.nearest_snapshot_delta_days is None


def test_invalid_skill_values_remain_unresolved() -> None:
    sofifa_id = 70

    snapshots = [
        (
            date(2024, 1, 9),
            {
                sofifa_id: {
                    "skills": [
                        None,
                        float("nan"),
                        float("inf"),
                        -1.0,
                        True,
                        "80",
                        75,
                    ]
                }
            },
        )
    ]

    result = reconstruct_past_only_skills(
        sofifa_id=sofifa_id,
        match_datetime=date(
            2024,
            1,
            10,
        ),
        snapshots=snapshots,
        player_occurrences={
            sofifa_id: [
                (
                    0,
                    date(2024, 1, 9),
                )
            ]
        },
        skill_count=7,
        max_age_days=30,
        max_snapshots=2,
    )

    assert result.skills == (
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        75.0,
    )
    assert result.observed_count == 1
    assert result.unresolved_count == 6
    assert result.source_dates[-1] == (date(2024, 1, 9))


def test_age_and_snapshot_limits_are_enforced() -> None:
    sofifa_id = 80

    snapshots = [
        (
            date(2024, 1, 9),
            {
                sofifa_id: {
                    "skills": [
                        10.0,
                        -1.0,
                    ]
                }
            },
        ),
        (
            date(2024, 1, 8),
            {
                sofifa_id: {
                    "skills": [
                        11.0,
                        20.0,
                    ]
                }
            },
        ),
        (
            date(2023, 1, 1),
            {
                sofifa_id: {
                    "skills": [
                        30.0,
                        40.0,
                    ]
                }
            },
        ),
    ]

    occurrences = {
        sofifa_id: [
            (
                0,
                date(2024, 1, 9),
            ),
            (
                1,
                date(2024, 1, 8),
            ),
            (
                2,
                date(2023, 1, 1),
            ),
        ]
    }

    limited = reconstruct_past_only_skills(
        sofifa_id=sofifa_id,
        match_datetime=date(
            2024,
            1,
            10,
        ),
        snapshots=snapshots,
        player_occurrences=occurrences,
        skill_count=2,
        max_age_days=30,
        max_snapshots=1,
    )

    assert limited.skills == (
        10.0,
        -1.0,
    )
    assert limited.snapshots_used == 1

    complete = reconstruct_past_only_skills(
        sofifa_id=sofifa_id,
        match_datetime=date(
            2024,
            1,
            10,
        ),
        snapshots=snapshots,
        player_occurrences=occurrences,
        skill_count=2,
        max_age_days=30,
        max_snapshots=2,
    )

    assert complete.skills == (
        10.0,
        20.0,
    )
    assert complete.snapshots_used == 2


def test_result_is_deterministic_without_input_mutation() -> None:
    sofifa_id = 90

    snapshots = [
        (
            date(2024, 1, 5),
            {
                sofifa_id: {
                    "skills": [
                        60.0,
                        70.0,
                    ]
                }
            },
        )
    ]
    occurrences = {
        sofifa_id: [
            (
                0,
                date(2024, 1, 5),
            )
        ]
    }

    snapshots_before = deepcopy(snapshots)
    occurrences_before = deepcopy(occurrences)

    first = reconstruct_past_only_skills(
        sofifa_id=sofifa_id,
        match_datetime=date(
            2024,
            1,
            6,
        ),
        snapshots=snapshots,
        player_occurrences=occurrences,
        skill_count=2,
        max_age_days=30,
        max_snapshots=2,
    )
    second = reconstruct_past_only_skills(
        sofifa_id=sofifa_id,
        match_datetime=date(
            2024,
            1,
            6,
        ),
        snapshots=snapshots,
        player_occurrences=occurrences,
        skill_count=2,
        max_age_days=30,
        max_snapshots=2,
    )

    assert first == second
    assert snapshots == snapshots_before
    assert occurrences == occurrences_before


@pytest.mark.parametrize(
    (
        "keyword",
        "value",
        "message",
    ),
    [
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
        "sofifa_id": 1,
        "match_datetime": date(
            2024,
            1,
            1,
        ),
        "snapshots": [],
        "player_occurrences": {},
        "skill_count": 2,
        "max_age_days": 30,
        "max_snapshots": 2,
    }
    arguments[keyword] = value

    with pytest.raises(
        ValueError,
        match=message,
    ):
        reconstruct_past_only_skills(**arguments)
