from __future__ import annotations

import math
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from numbers import Real
from statistics import median
from types import MappingProxyType

from football_outcomes.data.sofifa_strength import (
    UNRESOLVED_SOURCE_AGE_DAYS,
    PastOnlyTeamStrengthResult,
)
from football_outcomes.data.sofifa_temporal import (
    SkillProvenance,
)

_OBSERVED_PROVENANCE = frozenset(
    {
        SkillProvenance.NEAREST_PAST_SOFIFA,
        SkillProvenance.OLDER_PAST_SOFIFA,
    }
)

_PAST_ONLY_PROVENANCE = frozenset(
    {
        SkillProvenance.UNRESOLVED,
        *_OBSERVED_PROVENANCE,
    }
)

_IMPUTED_PROVENANCE = frozenset(
    {
        SkillProvenance.COMPETITION_POSITION_MEDIAN,
        SkillProvenance.POSITION_MEDIAN,
        SkillProvenance.GLOBAL_SKILL_MEDIAN,
        SkillProvenance.NEUTRAL_FALLBACK,
    }
)


@dataclass(frozen=True)
class StrengthImputationConfig:
    skill_count: int
    minimum_group_support: int = 20
    neutral_value: float = 50.0

    def __post_init__(self) -> None:
        if type(self.skill_count) is not int or self.skill_count <= 0:
            raise ValueError("skill_count must be a " "positive integer.")

        if type(self.minimum_group_support) is not int or self.minimum_group_support <= 0:
            raise ValueError("minimum_group_support must " "be a positive integer.")

        if isinstance(
            self.neutral_value,
            bool,
        ) or not isinstance(
            self.neutral_value,
            Real,
        ):
            raise ValueError("neutral_value must be a " "finite number in [0, 100].")

        neutral = float(self.neutral_value)

        if not math.isfinite(neutral) or neutral < 0.0 or neutral > 100.0:
            raise ValueError("neutral_value must be a " "finite number in [0, 100].")


@dataclass(frozen=True)
class StrengthImputationSample:
    competition_name: str
    strength: PastOnlyTeamStrengthResult

    def __post_init__(self) -> None:
        if not self.competition_name:
            raise ValueError("competition_name must not " "be empty.")


@dataclass(frozen=True)
class SkillMedianVector:
    values: tuple[
        float | None,
        ...,
    ]
    support: tuple[int, ...]


@dataclass(frozen=True)
class FittedStrengthImputer:
    config: StrengthImputationConfig
    competition_position_medians: Mapping[
        tuple[str, int],
        SkillMedianVector,
    ]
    position_medians: Mapping[
        int,
        SkillMedianVector,
    ]
    global_medians: SkillMedianVector
    training_team_count: int
    training_observed_cells: int


@dataclass(frozen=True)
class ImputedTeamStrengthResult:
    side: str
    team_id: int
    skills: tuple[
        tuple[float, ...],
        ...,
    ]
    provenance: tuple[
        tuple[SkillProvenance, ...],
        ...,
    ]
    source_age_days: tuple[
        tuple[int, ...],
        ...,
    ]
    observed_mask: tuple[
        tuple[int, ...],
        ...,
    ]
    fs_player_ids: tuple[int, ...]
    sofifa_player_ids: tuple[
        int | None,
        ...,
    ]
    position_indices: tuple[int, ...]

    def provenance_count(
        self,
        value: SkillProvenance,
    ) -> int:
        return sum(provenance is value for row in self.provenance for provenance in row)

    @property
    def observed_count(self) -> int:
        return sum(provenance in _OBSERVED_PROVENANCE for row in self.provenance for provenance in row)

    @property
    def imputed_count(self) -> int:
        return sum(provenance in _IMPUTED_PROVENANCE for row in self.provenance for provenance in row)

    @property
    def unresolved_count(self) -> int:
        return self.provenance_count(SkillProvenance.UNRESOLVED)


def _validate_matrix_shapes(
    strength: PastOnlyTeamStrengthResult,
    *,
    skill_count: int,
) -> None:
    row_count = len(strength.skills)

    for name, value in (
        (
            "provenance",
            strength.provenance,
        ),
        (
            "source_age_days",
            strength.source_age_days,
        ),
    ):
        if len(value) != row_count:
            raise ValueError(f"{name} row count does not " "match the skill matrix.")

    for name, value in (
        (
            "fs_player_ids",
            strength.fs_player_ids,
        ),
        (
            "sofifa_player_ids",
            strength.sofifa_player_ids,
        ),
        (
            "position_indices",
            strength.position_indices,
        ),
    ):
        if len(value) != row_count:
            raise ValueError(f"{name} length does not " "match the skill matrix.")

    for row_index in range(row_count):
        for name, matrix in (
            (
                "skills",
                strength.skills,
            ),
            (
                "provenance",
                strength.provenance,
            ),
            (
                "source_age_days",
                strength.source_age_days,
            ),
        ):
            if len(matrix[row_index]) != skill_count:
                raise ValueError(f"{name} row " f"{row_index} has " "an invalid skill width.")

        position = strength.position_indices[row_index]

        if type(position) is not int or position < 0:
            raise ValueError("Position indices must be " "non-negative integers.")


def _coerce_observed_value(
    value: object,
) -> float:
    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        Real,
    ):
        raise ValueError("Observed skill values must be " "finite non-negative numbers.")

    numeric = float(value)

    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError("Observed skill values must be " "finite non-negative numbers.")

    return numeric


def _normalise_provenance(
    value: object,
) -> SkillProvenance:
    try:
        return SkillProvenance(value)
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError("Invalid skill provenance code.") from error


def _empty_buckets(
    skill_count: int,
) -> list[list[float]]:
    return [[] for _ in range(skill_count)]


def _median_vector(
    buckets: Sequence[Sequence[float]],
    *,
    minimum_support: int,
) -> SkillMedianVector:
    values: list[float | None] = []
    support: list[int] = []

    for skill_values in buckets:
        count = len(skill_values)
        support.append(count)

        if count < minimum_support:
            values.append(None)
            continue

        values.append(float(median(skill_values)))

    return SkillMedianVector(
        values=tuple(values),
        support=tuple(support),
    )


def _freeze_mapping(
    values: Mapping,
) -> Mapping:
    return MappingProxyType(
        dict(
            sorted(
                values.items(),
                key=lambda item: item[0],
            )
        )
    )


def fit_strength_imputer(
    samples: Sequence[StrengthImputationSample],
    config: StrengthImputationConfig,
) -> FittedStrengthImputer:
    competition_position_buckets: dict[
        tuple[str, int],
        list[list[float]],
    ] = {}
    position_buckets: dict[
        int,
        list[list[float]],
    ] = {}
    global_buckets = _empty_buckets(config.skill_count)

    observed_cells = 0

    for sample in samples:
        strength = sample.strength

        _validate_matrix_shapes(
            strength,
            skill_count=config.skill_count,
        )

        for row_index, position in enumerate(strength.position_indices):
            competition_key = (
                sample.competition_name,
                position,
            )

            if competition_key not in competition_position_buckets:
                competition_position_buckets[competition_key] = _empty_buckets(config.skill_count)

            if position not in position_buckets:
                position_buckets[position] = _empty_buckets(config.skill_count)

            for skill_index in range(config.skill_count):
                provenance = _normalise_provenance(strength.provenance[row_index][skill_index])

                if provenance not in _PAST_ONLY_PROVENANCE:
                    raise ValueError("Training samples must " "contain only past-only " "provenance.")

                source_age = strength.source_age_days[row_index][skill_index]

                if provenance is (SkillProvenance.UNRESOLVED):
                    if source_age != UNRESOLVED_SOURCE_AGE_DAYS:
                        raise ValueError("Unresolved cells " "must have source " "age -1.")
                    continue

                if type(source_age) is not int or source_age < 0:
                    raise ValueError("Observed cells must " "have a non-negative " "source age.")

                value = _coerce_observed_value(strength.skills[row_index][skill_index])

                competition_position_buckets[competition_key][skill_index].append(value)
                position_buckets[position][skill_index].append(value)
                global_buckets[skill_index].append(value)

                observed_cells += 1

    competition_position_medians = {
        key: _median_vector(
            buckets,
            minimum_support=(config.minimum_group_support),
        )
        for key, buckets in (competition_position_buckets.items())
    }

    position_medians = {
        key: _median_vector(
            buckets,
            minimum_support=(config.minimum_group_support),
        )
        for key, buckets in (position_buckets.items())
    }

    global_medians = _median_vector(
        global_buckets,
        minimum_support=1,
    )

    return FittedStrengthImputer(
        config=config,
        competition_position_medians=(_freeze_mapping(competition_position_medians)),
        position_medians=(_freeze_mapping(position_medians)),
        global_medians=global_medians,
        training_team_count=len(samples),
        training_observed_cells=(observed_cells),
    )


def _estimate_value(
    estimate: SkillMedianVector | None,
    skill_index: int,
) -> float | None:
    if estimate is None:
        return None

    return estimate.values[skill_index]


def apply_strength_imputer(
    sample: StrengthImputationSample,
    imputer: FittedStrengthImputer,
) -> ImputedTeamStrengthResult:
    strength = sample.strength
    config = imputer.config

    _validate_matrix_shapes(
        strength,
        skill_count=config.skill_count,
    )

    skill_rows: list[tuple[float, ...]] = []
    provenance_rows: list[tuple[SkillProvenance, ...]] = []
    source_age_rows: list[tuple[int, ...]] = []
    observed_mask_rows: list[tuple[int, ...]] = []

    for row_index, position in enumerate(strength.position_indices):
        competition_estimate = imputer.competition_position_medians.get(
            (
                sample.competition_name,
                position,
            )
        )
        position_estimate = imputer.position_medians.get(position)

        output_values: list[float] = []
        output_provenance: list[SkillProvenance] = []
        output_source_ages: list[int] = []
        output_observed_mask: list[int] = []

        for skill_index in range(config.skill_count):
            provenance = _normalise_provenance(strength.provenance[row_index][skill_index])

            if provenance not in _PAST_ONLY_PROVENANCE:
                raise ValueError("Input samples must " "contain only past-only " "provenance.")

            if provenance in (_OBSERVED_PROVENANCE):
                value = _coerce_observed_value(strength.skills[row_index][skill_index])
                source_age = strength.source_age_days[row_index][skill_index]

                if type(source_age) is not int or source_age < 0:
                    raise ValueError("Observed cells must " "have a non-negative " "source age.")

                output_values.append(value)
                output_provenance.append(provenance)
                output_source_ages.append(source_age)
                output_observed_mask.append(1)
                continue

            source_age = strength.source_age_days[row_index][skill_index]

            if source_age != UNRESOLVED_SOURCE_AGE_DAYS:
                raise ValueError("Unresolved cells must " "have source age -1.")

            value = _estimate_value(
                competition_estimate,
                skill_index,
            )

            if value is not None:
                output_provenance.append(SkillProvenance.COMPETITION_POSITION_MEDIAN)
            else:
                value = _estimate_value(
                    position_estimate,
                    skill_index,
                )

                if value is not None:
                    output_provenance.append(SkillProvenance.POSITION_MEDIAN)
                else:
                    value = _estimate_value(
                        imputer.global_medians,
                        skill_index,
                    )

                    if value is not None:
                        output_provenance.append(SkillProvenance.GLOBAL_SKILL_MEDIAN)
                    else:
                        value = float(config.neutral_value)
                        output_provenance.append(SkillProvenance.NEUTRAL_FALLBACK)

            output_values.append(float(value))
            output_source_ages.append(UNRESOLVED_SOURCE_AGE_DAYS)
            output_observed_mask.append(0)

        skill_rows.append(tuple(output_values))
        provenance_rows.append(tuple(output_provenance))
        source_age_rows.append(tuple(output_source_ages))
        observed_mask_rows.append(tuple(output_observed_mask))

    result = ImputedTeamStrengthResult(
        side=strength.side,
        team_id=strength.team_id,
        skills=tuple(skill_rows),
        provenance=tuple(provenance_rows),
        source_age_days=tuple(source_age_rows),
        observed_mask=tuple(observed_mask_rows),
        fs_player_ids=(strength.fs_player_ids),
        sofifa_player_ids=(strength.sofifa_player_ids),
        position_indices=(strength.position_indices),
    )

    if result.unresolved_count:
        raise RuntimeError("Imputation left unresolved " "skill cells.")

    return result
