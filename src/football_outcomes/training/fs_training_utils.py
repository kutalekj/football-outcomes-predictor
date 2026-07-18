from __future__ import annotations

from typing import List

import numpy as np

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import FSMatch
from football_outcomes.datasets import arrays as _arrays
from football_outcomes.datasets import mappings as _mappings
from football_outcomes.datasets import rounds as _rounds
from football_outcomes.datasets.targets import (
    target_dtype,
    target_for_match,
)

# Compatibility exports for callers using the legacy module path.
CatMaps = _mappings.CatMaps
extract_numerical_features = _arrays.extract_numerical_features
_strength_to_value_and_mask = _arrays.strength_to_value_and_mask
distribute_matches_into_rounds = _rounds.distribute_matches_into_rounds
summarize_rounds = _rounds.summarize_rounds


# Categorical mappings


def build_categorical_maps(
    league_matches_sorted: List[FSMatch],
) -> CatMaps:
    """Build mappings using the legacy competition ordering."""

    return _mappings.build_categorical_maps(
        league_matches_sorted,
        sett.COMPS_LEAGUE,
    )


# Feature extraction


def build_arrays_for_matches(
    matches: List[FSMatch],
    cat_maps: CatMaps,
    mode: str,
    max_goals_class: int = 10,
):
    """Build arrays using the legacy competition ordering."""

    return _arrays.build_arrays_for_matches(
        matches=matches,
        cat_maps=cat_maps,
        competition_names=sett.COMPS_LEAGUE,
        mode=mode,
        max_goals_class=max_goals_class,
    )


def build_flat_tabular_arrays_for_matches(
    matches: List[FSMatch],
    cat_maps: CatMaps,
    mode: str,
    max_goals_class: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build flat arrays using legacy settings."""

    return _arrays.build_flat_tabular_arrays_for_matches(
        matches=matches,
        cat_maps=cat_maps,
        competition_names=(sett.COMPS_LEAGUE),
        mode=mode,
        max_goals_class=(max_goals_class),
    )


def build_aux_targets_for_matches(
    matches: List[FSMatch],
    aux_mode: str,
    max_goals_class: int = 10,
) -> np.ndarray:
    """
    Build auxiliary targets from raw match outcomes.

    Supported:
      - "binary_u25" : 1 if total goals <= 2 else 0
      - "goals_reg"  : total goals as float
      - "goals_dist" : clipped total-goals class
    """
    vals = []

    for match in matches:
        try:
            value = target_for_match(
                match,
                aux_mode,
                max_goals_class,
            )
        except ValueError:
            raise ValueError(f"Unknown aux_mode: {aux_mode}") from None

        vals.append(value)

    return np.asarray(
        vals,
        dtype=target_dtype(aux_mode),
    )


def build_strength_only_arrays_for_matches(
    matches: List[FSMatch],
    mode: str,
    max_goals_class: int = 10,
):
    """Build structured arrays through the new module."""

    return _arrays.build_strength_only_arrays_for_matches(
        matches=matches,
        mode=mode,
        max_goals_class=(max_goals_class),
    )
