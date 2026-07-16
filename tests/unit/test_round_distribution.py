from __future__ import annotations

from types import SimpleNamespace

from football_outcomes.training.fs_training_utils import (
    distribute_matches_into_rounds,
)


def make_match(home_id: int, away_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        home_team=SimpleNamespace(id=home_id),
        away_team=SimpleNamespace(id=away_id),
    )


def test_round_starts_when_a_team_reappears() -> None:
    matches = [
        make_match(1, 2),
        make_match(3, 4),
        make_match(1, 5),
        make_match(6, 7),
    ]

    rounds = distribute_matches_into_rounds(matches)

    assert [len(round_) for round_ in rounds] == [2, 2]


def test_no_team_occurs_twice_within_a_round() -> None:
    matches = [
        make_match(1, 2),
        make_match(3, 4),
        make_match(1, 5),
        make_match(6, 7),
    ]

    rounds = distribute_matches_into_rounds(matches)

    for round_ in rounds:
        team_ids = [
            team_id
            for match in round_
            for team_id in (
                match.home_team.id,
                match.away_team.id,
            )
        ]

        assert len(team_ids) == len(set(team_ids))
