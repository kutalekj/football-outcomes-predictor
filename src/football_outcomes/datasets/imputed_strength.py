from __future__ import annotations

from collections import Counter
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import date

import numpy as np

from football_outcomes.data.fs_models import (
    FSMatch,
)
from football_outcomes.data.sofifa_imputation import (
    FittedStrengthImputer,
    ImputedTeamStrengthResult,
    StrengthImputationConfig,
    StrengthImputationSample,
    apply_strength_imputer,
    fit_strength_imputer,
)
from football_outcomes.data.sofifa_strength import (
    PastOnlyMatchStrengthResult,
    PastOnlyStrengthConfig,
    reconstruct_past_only_match_strength,
)
from football_outcomes.data.sofifa_temporal import (
    Snapshot,
)
from football_outcomes.datasets.arrays import (
    build_arrays_for_matches,
)
from football_outcomes.datasets.mappings import (
    CatMaps,
)

ArrayBundle = tuple[
    np.ndarray,
    ...,
]


@dataclass(frozen=True)
class StrengthImputationContext:
    snapshots: Sequence[Snapshot]
    player_occurrences: Mapping[
        int,
        Sequence[tuple[int, date]],
    ]
    fs_to_sofifa_cache: Mapping[
        int,
        object,
    ]
    reconstruction_config: PastOnlyStrengthConfig


@dataclass(frozen=True)
class FoldImputationDiagnostics:
    training_team_count: int
    training_observed_cells: int
    training_provenance_counts: tuple[
        tuple[str, int],
        ...,
    ]
    validation_provenance_counts: tuple[
        tuple[str, int],
        ...,
    ]

    def to_dict(self) -> dict:
        return {
            "training_team_count": (self.training_team_count),
            "training_observed_cells": (self.training_observed_cells),
            ("training_provenance_counts"): dict(self.training_provenance_counts),
            ("validation_provenance_counts"): dict(self.validation_provenance_counts),
        }


def _build_base_arrays(
    matches: Sequence[FSMatch],
    *,
    cat_maps: CatMaps,
    competition_names: Sequence[str],
    mode: str,
    max_goals_class: int,
) -> ArrayBundle:
    arrays = build_arrays_for_matches(
        matches=matches,
        cat_maps=cat_maps,
        competition_names=(competition_names),
        mode=mode,
        max_goals_class=(max_goals_class),
    )

    if len(arrays) != 8:
        raise RuntimeError("Expected the active array " "builder to return eight arrays; " f"found {len(arrays)}.")

    return tuple(arrays)


def _reconstruct_matches(
    matches: Sequence[FSMatch],
    context: StrengthImputationContext,
) -> tuple[
    PastOnlyMatchStrengthResult,
    ...,
]:
    return tuple(
        reconstruct_past_only_match_strength(
            match=match,
            snapshots=context.snapshots,
            player_occurrences=(context.player_occurrences),
            fs_to_sofifa_cache=(context.fs_to_sofifa_cache),
            config=(context.reconstruction_config),
        )
        for match in matches
    )


def _training_samples(
    matches: Sequence[FSMatch],
    reconstructed: Sequence[PastOnlyMatchStrengthResult],
) -> list[StrengthImputationSample]:
    if len(matches) != len(reconstructed):
        raise RuntimeError("Match and reconstruction " "counts differ.")

    samples = []

    for match, result in zip(
        matches,
        reconstructed,
    ):
        samples.extend(
            [
                StrengthImputationSample(
                    competition_name=(match.comp_name),
                    strength=result.home,
                ),
                StrengthImputationSample(
                    competition_name=(match.comp_name),
                    strength=result.away,
                ),
            ]
        )

    return samples


def _normalised_values(
    result: ImputedTeamStrengthResult,
) -> np.ndarray:
    values = np.asarray(
        result.skills,
        dtype=np.float32,
    )

    if not np.isfinite(values).all():
        raise RuntimeError("Imputed strength values " "must be finite.")

    if np.any(values < 0.0) or np.any(values > 100.0):
        raise RuntimeError("Imputed strength values " "must remain in [0, 100].")

    return values / np.float32(100.0)


def _observed_mask(
    result: ImputedTeamStrengthResult,
) -> np.ndarray:
    mask = np.asarray(
        result.observed_mask,
        dtype=np.float32,
    )

    if not np.isin(
        mask,
        (
            0.0,
            1.0,
        ),
    ).all():
        raise RuntimeError("Observed masks must be binary.")

    return mask


def _count_provenance(
    result: ImputedTeamStrengthResult,
    counts: Counter,
) -> None:
    for row in result.provenance:
        for provenance in row:
            counts[provenance.name] += 1


def _build_completed_inputs(
    matches: Sequence[FSMatch],
    reconstructed: Sequence[PastOnlyMatchStrengthResult],
    imputer: FittedStrengthImputer,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[tuple[str, int], ...],
]:
    if len(matches) != len(reconstructed):
        raise RuntimeError("Match and reconstruction " "counts differ.")

    strength_rows = []
    home_position_rows = []
    away_position_rows = []
    counts: Counter = Counter()

    for match, raw in zip(
        matches,
        reconstructed,
    ):
        home = apply_strength_imputer(
            StrengthImputationSample(
                competition_name=(match.comp_name),
                strength=raw.home,
            ),
            imputer,
        )
        away = apply_strength_imputer(
            StrengthImputationSample(
                competition_name=(match.comp_name),
                strength=raw.away,
            ),
            imputer,
        )

        home_values = _normalised_values(home)
        away_values = _normalised_values(away)
        home_mask = _observed_mask(home)
        away_mask = _observed_mask(away)

        strength_rows.append(
            np.stack(
                [
                    home_values,
                    home_mask,
                    away_values,
                    away_mask,
                ],
                axis=0,
            )
        )

        home_position_rows.append(
            np.asarray(
                home.position_indices,
                dtype=np.int32,
            )
        )
        away_position_rows.append(
            np.asarray(
                away.position_indices,
                dtype=np.int32,
            )
        )

        _count_provenance(
            home,
            counts,
        )
        _count_provenance(
            away,
            counts,
        )

    return (
        np.stack(
            strength_rows,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
        np.stack(
            home_position_rows,
            axis=0,
        ).astype(
            np.int32,
            copy=False,
        ),
        np.stack(
            away_position_rows,
            axis=0,
        ).astype(
            np.int32,
            copy=False,
        ),
        tuple(sorted(counts.items())),
    )


def _replace_structured_inputs(
    base: ArrayBundle,
    *,
    strength: np.ndarray,
    home_positions: np.ndarray,
    away_positions: np.ndarray,
) -> ArrayBundle:
    if len(base) != 8:
        raise RuntimeError("Expected eight base arrays.")

    result = list(base)
    result[4] = strength
    result[5] = home_positions
    result[6] = away_positions

    return tuple(result)


def build_fold_imputed_arrays(
    *,
    training_matches: Sequence[FSMatch],
    validation_matches: Sequence[FSMatch],
    cat_maps: CatMaps,
    competition_names: Sequence[str],
    mode: str,
    max_goals_class: int,
    context: StrengthImputationContext,
    imputation_config: StrengthImputationConfig,
) -> tuple[
    ArrayBundle,
    ArrayBundle,
    FoldImputationDiagnostics,
]:
    if not training_matches:
        raise ValueError("training_matches must not " "be empty.")

    if not validation_matches:
        raise ValueError("validation_matches must not " "be empty.")

    if context.reconstruction_config.skill_count != imputation_config.skill_count:
        raise ValueError("Reconstruction and imputation " "skill counts differ.")

    training_base = _build_base_arrays(
        training_matches,
        cat_maps=cat_maps,
        competition_names=(competition_names),
        mode=mode,
        max_goals_class=(max_goals_class),
    )
    validation_base = _build_base_arrays(
        validation_matches,
        cat_maps=cat_maps,
        competition_names=(competition_names),
        mode=mode,
        max_goals_class=(max_goals_class),
    )

    training_raw = _reconstruct_matches(
        training_matches,
        context,
    )
    validation_raw = _reconstruct_matches(
        validation_matches,
        context,
    )

    imputer = fit_strength_imputer(
        _training_samples(
            training_matches,
            training_raw,
        ),
        imputation_config,
    )

    (
        training_strength,
        training_home_positions,
        training_away_positions,
        training_counts,
    ) = _build_completed_inputs(
        training_matches,
        training_raw,
        imputer,
    )

    (
        validation_strength,
        validation_home_positions,
        validation_away_positions,
        validation_counts,
    ) = _build_completed_inputs(
        validation_matches,
        validation_raw,
        imputer,
    )

    training_arrays = _replace_structured_inputs(
        training_base,
        strength=(training_strength),
        home_positions=(training_home_positions),
        away_positions=(training_away_positions),
    )
    validation_arrays = _replace_structured_inputs(
        validation_base,
        strength=(validation_strength),
        home_positions=(validation_home_positions),
        away_positions=(validation_away_positions),
    )

    diagnostics = FoldImputationDiagnostics(
        training_team_count=(imputer.training_team_count),
        training_observed_cells=(imputer.training_observed_cells),
        training_provenance_counts=(training_counts),
        validation_provenance_counts=(validation_counts),
    )

    return (
        training_arrays,
        validation_arrays,
        diagnostics,
    )
