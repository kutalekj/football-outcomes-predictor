from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from football_outcomes.data.fs_models import FSMatch


def distribute_matches_into_rounds(
    sorted_matches: Sequence[FSMatch],
) -> list[list[FSMatch]]:
    """Create chronological rounds with no repeated team."""

    rounds: list[list[FSMatch]] = []
    current_round: list[FSMatch] = []
    teams_in_round: set[int] = set()

    for match in sorted_matches:
        home_id = match.home_team.id
        away_id = match.away_team.id

        if home_id in teams_in_round or away_id in teams_in_round:
            rounds.append(current_round)
            current_round = []
            teams_in_round = set()

        current_round.append(match)
        teams_in_round.add(home_id)
        teams_in_round.add(away_id)

    if current_round:
        rounds.append(current_round)

    return rounds


def summarize_rounds(
    rounds: Sequence[Sequence[FSMatch]],
) -> dict[str, int | float]:
    """Return basic round-size statistics."""

    sizes = np.asarray(
        [len(round_) for round_ in rounds],
        dtype=np.int32,
    )

    return {
        "num_rounds": int(len(rounds)),
        "min_round_size": (int(sizes.min()) if sizes.size else 0),
        "max_round_size": (int(sizes.max()) if sizes.size else 0),
        "mean_round_size": (float(sizes.mean()) if sizes.size else 0.0),
        "median_round_size": (float(np.median(sizes)) if sizes.size else 0.0),
    }
