from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from football_outcomes.data.fs_models import FSMatch


def target_for_match(
    match: FSMatch,
    mode: str,
    max_goals_class: int = 10,
) -> float | int:
    """Construct one prediction target from a match result."""

    total_goals = (match.home_goals or 0) + (match.away_goals or 0)

    if mode == "binary_u25":
        return 1.0 if total_goals <= 2 else 0.0

    if mode == "goals_dist":
        return int(
            min(
                total_goals,
                max_goals_class,
            )
        )

    if mode == "goals_reg":
        return float(total_goals)

    raise ValueError(f"Unknown mode: {mode}")


def target_dtype(
    mode: str,
) -> type[np.float32] | type[np.int32]:
    """Return the legacy NumPy dtype for a target mode."""

    if mode in (
        "binary_u25",
        "goals_reg",
    ):
        return np.float32

    return np.int32


def build_targets_for_matches(
    matches: Sequence[FSMatch],
    mode: str,
    max_goals_class: int = 10,
) -> np.ndarray:
    """Build the target array for a sequence of matches."""

    values = [
        target_for_match(
            match,
            mode,
            max_goals_class,
        )
        for match in matches
    ]

    return np.asarray(
        values,
        dtype=target_dtype(mode),
    )
