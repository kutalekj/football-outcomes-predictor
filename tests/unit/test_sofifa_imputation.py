from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from football_outcomes.data.sofifa_imputation import (
    StrengthImputationConfig,
    StrengthImputationSample,
    apply_strength_imputer,
    fit_strength_imputer,
)
from football_outcomes.data.sofifa_strength import (
    UNRESOLVED_SOURCE_AGE_DAYS,
    PastOnlyTeamStrengthResult,
)
from football_outcomes.data.sofifa_temporal import (
    SkillProvenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBSERVED = SkillProvenance.NEAREST_PAST_SOFIFA
MISSING = SkillProvenance.UNRESOLVED


def make_result(
    *,
    skills,
    provenance,
    positions,
    source_ages=None,
    team_id: int = 1,
) -> PastOnlyTeamStrengthResult:
    row_count = len(skills)

    if source_ages is None:
        source_ages = [
            [
                (
                    1
                    if cell
                    in (
                        SkillProvenance.NEAREST_PAST_SOFIFA,
                        SkillProvenance.OLDER_PAST_SOFIFA,
                    )
                    else (UNRESOLVED_SOURCE_AGE_DAYS)
                )
                for cell in row
            ]
            for row in provenance
        ]

    return PastOnlyTeamStrengthResult(
        side="home",
        team_id=team_id,
        skills=tuple(tuple(row) for row in skills),
        provenance=tuple(tuple(row) for row in provenance),
        source_age_days=tuple(tuple(row) for row in source_ages),
        fs_player_ids=tuple(
            range(
                1,
                row_count + 1,
            )
        ),
        sofifa_player_ids=tuple(
            range(
                101,
                101 + row_count,
            )
        ),
        position_indices=tuple(positions),
    )


def sample(
    competition: str,
    *,
    skills,
    provenance,
    positions,
    source_ages=None,
) -> StrengthImputationSample:
    return StrengthImputationSample(
        competition_name=competition,
        strength=make_result(
            skills=skills,
            provenance=provenance,
            positions=positions,
            source_ages=source_ages,
        ),
    )


def make_config() -> StrengthImputationConfig:
    return StrengthImputationConfig(
        skill_count=4,
        minimum_group_support=2,
        neutral_value=50.0,
    )


def test_imputation_module_is_pure_and_offline() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "sofifa_imputation.py"
    source = source_path.read_text(encoding="utf-8")

    assert "requests" not in source
    assert "fs_globals" not in source
    assert "Global" not in source
    assert "fs_settings" not in source
    assert "from football_outcomes.training" not in source
    assert "import football_outcomes.training" not in source


def test_fallback_hierarchy_is_applied() -> None:
    training = [
        sample(
            "League A",
            skills=[
                [
                    10.0,
                    30.0,
                    -1.0,
                    -1.0,
                ]
            ],
            provenance=[
                [
                    OBSERVED,
                    OBSERVED,
                    MISSING,
                    MISSING,
                ]
            ],
            positions=[1],
        ),
        sample(
            "League A",
            skills=[
                [
                    20.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ]
            ],
            provenance=[
                [
                    OBSERVED,
                    MISSING,
                    MISSING,
                    MISSING,
                ]
            ],
            positions=[1],
        ),
        sample(
            "League B",
            skills=[
                [
                    100.0,
                    50.0,
                    -1.0,
                    -1.0,
                ]
            ],
            provenance=[
                [
                    OBSERVED,
                    OBSERVED,
                    MISSING,
                    MISSING,
                ]
            ],
            positions=[1],
        ),
        sample(
            "League A",
            skills=[
                [
                    -1.0,
                    -1.0,
                    70.0,
                    -1.0,
                ]
            ],
            provenance=[
                [
                    MISSING,
                    MISSING,
                    OBSERVED,
                    MISSING,
                ]
            ],
            positions=[2],
        ),
        sample(
            "League B",
            skills=[
                [
                    -1.0,
                    -1.0,
                    90.0,
                    -1.0,
                ]
            ],
            provenance=[
                [
                    MISSING,
                    MISSING,
                    OBSERVED,
                    MISSING,
                ]
            ],
            positions=[3],
        ),
    ]

    imputer = fit_strength_imputer(
        training,
        make_config(),
    )

    target = sample(
        "League A",
        skills=[
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ]
        ],
        provenance=[
            [
                MISSING,
                MISSING,
                MISSING,
                MISSING,
            ]
        ],
        positions=[1],
    )

    result = apply_strength_imputer(
        target,
        imputer,
    )

    assert result.skills == (
        (
            15.0,
            40.0,
            80.0,
            50.0,
        ),
    )
    assert result.provenance == (
        (
            SkillProvenance.COMPETITION_POSITION_MEDIAN,
            SkillProvenance.POSITION_MEDIAN,
            SkillProvenance.GLOBAL_SKILL_MEDIAN,
            SkillProvenance.NEUTRAL_FALLBACK,
        ),
    )
    assert result.observed_mask == (
        (
            0,
            0,
            0,
            0,
        ),
    )
    assert result.imputed_count == 4
    assert result.unresolved_count == 0
    assert imputer.training_observed_cells == 7


def test_observed_values_and_ages_are_preserved() -> None:
    training = [
        sample(
            "League A",
            skills=[
                [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                ]
            ],
            provenance=[
                [
                    OBSERVED,
                    OBSERVED,
                    OBSERVED,
                    OBSERVED,
                ]
            ],
            positions=[1],
        ),
        sample(
            "League A",
            skills=[
                [
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                ]
            ],
            provenance=[
                [
                    OBSERVED,
                    OBSERVED,
                    OBSERVED,
                    OBSERVED,
                ]
            ],
            positions=[1],
        ),
    ]

    imputer = fit_strength_imputer(
        training,
        make_config(),
    )

    target = sample(
        "League A",
        skills=[
            [
                77.0,
                -1.0,
                -1.0,
                -1.0,
            ]
        ],
        provenance=[
            [
                OBSERVED,
                MISSING,
                MISSING,
                MISSING,
            ]
        ],
        positions=[1],
        source_ages=[
            [
                5,
                -1,
                -1,
                -1,
            ]
        ],
    )

    result = apply_strength_imputer(
        target,
        imputer,
    )

    assert result.skills[0][0] == 77.0
    assert result.provenance[0][0] is OBSERVED
    assert result.source_age_days[0][0] == 5
    assert result.observed_mask[0] == (
        1,
        0,
        0,
        0,
    )


def test_empty_training_uses_neutral_fallback() -> None:
    imputer = fit_strength_imputer(
        [],
        make_config(),
    )

    target = sample(
        "League A",
        skills=[
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ]
        ],
        provenance=[
            [
                MISSING,
                MISSING,
                MISSING,
                MISSING,
            ]
        ],
        positions=[1],
    )

    result = apply_strength_imputer(
        target,
        imputer,
    )

    assert result.skills == (
        (
            50.0,
            50.0,
            50.0,
            50.0,
        ),
    )
    assert all(provenance is SkillProvenance.NEUTRAL_FALLBACK for provenance in result.provenance[0])


def test_imputed_training_values_are_rejected() -> None:
    invalid = sample(
        "League A",
        skills=[
            [
                50.0,
                -1.0,
                -1.0,
                -1.0,
            ]
        ],
        provenance=[
            [
                SkillProvenance.GLOBAL_SKILL_MEDIAN,
                MISSING,
                MISSING,
                MISSING,
            ]
        ],
        positions=[1],
    )

    with pytest.raises(
        ValueError,
        match="only past-only provenance",
    ):
        fit_strength_imputer(
            [invalid],
            make_config(),
        )


def test_fit_and_apply_are_deterministic_without_mutation() -> None:
    training = [
        sample(
            "League A",
            skills=[
                [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                ]
            ],
            provenance=[
                [
                    OBSERVED,
                    OBSERVED,
                    OBSERVED,
                    OBSERVED,
                ]
            ],
            positions=[1],
        ),
        sample(
            "League A",
            skills=[
                [
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                ]
            ],
            provenance=[
                [
                    OBSERVED,
                    OBSERVED,
                    OBSERVED,
                    OBSERVED,
                ]
            ],
            positions=[1],
        ),
    ]

    target = sample(
        "League A",
        skills=[
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ]
        ],
        provenance=[
            [
                MISSING,
                MISSING,
                MISSING,
                MISSING,
            ]
        ],
        positions=[1],
    )

    training_before = deepcopy(training)
    target_before = deepcopy(target)

    first_imputer = fit_strength_imputer(
        training,
        make_config(),
    )
    second_imputer = fit_strength_imputer(
        training,
        make_config(),
    )

    first = apply_strength_imputer(
        target,
        first_imputer,
    )
    second = apply_strength_imputer(
        target,
        second_imputer,
    )

    assert first == second
    assert training == training_before
    assert target == target_before

    with pytest.raises(TypeError):
        first_imputer.position_medians[99] = first_imputer.global_medians


def test_invalid_observed_source_age_is_rejected() -> None:
    invalid = sample(
        "League A",
        skills=[
            [
                10.0,
                -1.0,
                -1.0,
                -1.0,
            ]
        ],
        provenance=[
            [
                OBSERVED,
                MISSING,
                MISSING,
                MISSING,
            ]
        ],
        positions=[1],
        source_ages=[
            [
                -1,
                -1,
                -1,
                -1,
            ]
        ],
    )

    with pytest.raises(
        ValueError,
        match="non-negative source age",
    ):
        fit_strength_imputer(
            [invalid],
            make_config(),
        )


@pytest.mark.parametrize(
    (
        "arguments",
        "message",
    ),
    [
        (
            {
                "skill_count": 0,
            },
            "skill_count must be",
        ),
        (
            {
                "minimum_group_support": 0,
            },
            ("minimum_group_support " "must be"),
        ),
        (
            {
                "neutral_value": -1.0,
            },
            "neutral_value must be",
        ),
        (
            {
                "neutral_value": 101.0,
            },
            "neutral_value must be",
        ),
        (
            {
                "neutral_value": (float("inf")),
            },
            "neutral_value must be",
        ),
    ],
)
def test_invalid_configuration_is_rejected(
    arguments,
    message,
) -> None:
    values = {
        "skill_count": 4,
        "minimum_group_support": 2,
        "neutral_value": 50.0,
    }
    values.update(arguments)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        StrengthImputationConfig(**values)
