from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

from football_outcomes.data import (
    sofifa_skills,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_skill_module_has_no_matching_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "sofifa_skills.py"
    source = source_path.read_text(encoding="utf-8")

    assert "rapidfuzz" not in source
    assert "MatchCandidate" not in source
    assert "MatchResult" not in source
    assert "FSPlayer" not in source
    assert "fs_player_skill_utils" not in source


def test_snapshot_order_is_past_first_and_bounded() -> None:
    match_date = date(
        2024,
        1,
        10,
    )

    occurrences = [
        (
            0,
            date(2024, 1, 11),
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
            3,
            date(2024, 1, 12),
        ),
        (
            4,
            date(2023, 10, 1),
        ),
    ]

    ordered = sofifa_skills.ordered_snapshot_candidates(
        occurrences,
        match_date,
        max_days=30,
        max_snapshots=3,
    )

    assert ordered == [
        (
            2,
            date(2024, 1, 9),
        ),
        (
            1,
            date(2024, 1, 1),
        ),
        (
            0,
            date(2024, 1, 11),
        ),
    ]


def test_missing_skills_are_filled_from_later_snapshots() -> None:
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
                        12.0,
                        22.0,
                        32.0,
                        42.0,
                    ]
                }
            },
        ),
    ]

    result = sofifa_skills.merge_skills_from_snapshot_data(
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
        max_days=30,
        max_snapshots=3,
    )

    assert result == (
        [
            10.0,
            20.0,
            30.0,
            40.0,
        ],
        2,
        1,
    )


def test_future_snapshot_is_used_after_invalid_past_record() -> None:
    sofifa_id = 60

    snapshots = [
        (
            date(2024, 1, 9),
            {sofifa_id: {"skills": [1.0]}},
        ),
        (
            date(2024, 1, 11),
            {
                sofifa_id: {
                    "skills": [
                        5.0,
                        6.0,
                    ]
                }
            },
        ),
    ]

    result = sofifa_skills.merge_skills_from_snapshot_data(
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
                ),
                (
                    1,
                    date(2024, 1, 11),
                ),
            ]
        },
        skill_count=2,
        max_days=30,
        max_snapshots=2,
    )

    assert result == (
        [5.0, 6.0],
        1,
        -1,
    )


def test_missing_occurrences_return_missing_vector() -> None:
    result = sofifa_skills.merge_skills_from_snapshot_data(
        sofifa_id=999,
        match_datetime=datetime(
            2024,
            1,
            10,
        ),
        snapshots=[],
        player_occurrences={},
        skill_count=3,
        max_days=30,
        max_snapshots=4,
    )

    assert result == (
        [-1.0, -1.0, -1.0],
        0,
        0,
    )


def test_legacy_wrapper_reads_current_global_state(
    monkeypatch,
) -> None:
    sofifa_id = 70
    snapshot_date = date(
        2024,
        1,
        5,
    )

    state = SimpleNamespace(
        sofifa_snapshots=[
            (
                snapshot_date,
                {sofifa_id: {"skills": [8.0]}},
            )
        ],
        sofifa_player_occurrences={
            sofifa_id: [
                (
                    0,
                    snapshot_date,
                )
            ]
        },
    )

    monkeypatch.setattr(
        sofifa_skills,
        "Global",
        SimpleNamespace(get_instance=(lambda: state)),
    )
    monkeypatch.setattr(
        sofifa_skills.sett,
        "PLAYER_SKILLS",
        ["skill"],
    )
    monkeypatch.setattr(
        sofifa_skills.sett,
        "SF_MAX_TIMEDELTA_DAYS",
        30,
    )
    monkeypatch.setattr(
        sofifa_skills.sett,
        "SF_MAX_SNAPSHOTS_TO_SCAN",
        4,
    )

    result = sofifa_skills.merge_skills_from_snapshots(
        sofifa_id,
        datetime(
            2024,
            1,
            6,
        ),
    )

    assert result == (
        [8.0],
        1,
        1,
    )
