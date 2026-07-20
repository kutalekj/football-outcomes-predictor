from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from football_outcomes.data.sofifa_imputation import (
    StrengthImputationConfig,
)
from football_outcomes.data.sofifa_strength import (
    PastOnlyMatchStrengthResult,
    PastOnlyStrengthConfig,
    PastOnlyTeamStrengthResult,
)
from football_outcomes.data.sofifa_temporal import (
    SkillProvenance,
)
from football_outcomes.datasets import (
    imputed_strength,
)
from football_outcomes.datasets.imputed_strength import (
    StrengthImputationContext,
    build_fold_imputed_arrays,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBSERVED = SkillProvenance.NEAREST_PAST_SOFIFA
MISSING = SkillProvenance.UNRESOLVED


def team_result(
    *,
    side: str,
    team_id: int,
    skills,
    provenance,
) -> PastOnlyTeamStrengthResult:
    return PastOnlyTeamStrengthResult(
        side=side,
        team_id=team_id,
        skills=(tuple(skills),),
        provenance=(tuple(provenance),),
        source_age_days=(tuple(1 if value is OBSERVED else -1 for value in provenance),),
        fs_player_ids=(team_id,),
        sofifa_player_ids=(1000 + team_id,),
        position_indices=(1,),
    )


def match_result(
    match_id: int,
    *,
    home_skills,
    home_provenance,
    away_skills,
    away_provenance,
) -> PastOnlyMatchStrengthResult:
    return PastOnlyMatchStrengthResult(
        match_id=match_id,
        home=team_result(
            side="home",
            team_id=1,
            skills=home_skills,
            provenance=(home_provenance),
        ),
        away=team_result(
            side="away",
            team_id=2,
            skills=away_skills,
            provenance=(away_provenance),
        ),
    )


def fake_base_arrays(
    matches,
    **kwargs,
):
    count = len(matches)

    return (
        np.arange(
            count * 2,
            dtype=np.float32,
        ).reshape(
            count,
            2,
        ),
        np.zeros(
            (
                count,
                1,
            ),
            dtype=np.int32,
        ),
        np.ones(
            (
                count,
                1,
            ),
            dtype=np.int32,
        ),
        np.zeros(
            (
                count,
                1,
            ),
            dtype=np.int32,
        ),
        np.zeros(
            (
                count,
                4,
                1,
                2,
            ),
            dtype=np.float32,
        ),
        np.zeros(
            (
                count,
                1,
            ),
            dtype=np.int32,
        ),
        np.zeros(
            (
                count,
                1,
            ),
            dtype=np.int32,
        ),
        np.arange(
            count,
            dtype=np.float32,
        ),
    )


def make_context() -> StrengthImputationContext:
    return StrengthImputationContext(
        snapshots=[],
        player_occurrences={},
        fs_to_sofifa_cache={},
        reconstruction_config=(
            PastOnlyStrengthConfig(
                player_count=1,
                skill_count=2,
                max_age_days=30,
                max_snapshots=2,
            )
        ),
    )


def test_adapter_module_has_no_training_or_global_dependency() -> None:
    path = PROJECT_ROOT / "src" / "football_outcomes" / "datasets" / "imputed_strength.py"
    source = path.read_text(encoding="utf-8")

    assert "football_outcomes.training" not in source
    assert "fs_globals" not in source
    assert "Global" not in source
    assert "fs_settings" not in source


def test_fold_arrays_use_training_fitted_imputer(
    monkeypatch,
) -> None:
    training_match = SimpleNamespace(
        id=1,
        comp_name="League A",
    )
    validation_match = SimpleNamespace(
        id=2,
        comp_name="League A",
    )

    results = {
        1: match_result(
            1,
            home_skills=(
                20.0,
                40.0,
            ),
            home_provenance=(
                OBSERVED,
                OBSERVED,
            ),
            away_skills=(
                30.0,
                60.0,
            ),
            away_provenance=(
                OBSERVED,
                OBSERVED,
            ),
        ),
        2: match_result(
            2,
            home_skills=(
                -1.0,
                -1.0,
            ),
            home_provenance=(
                MISSING,
                MISSING,
            ),
            away_skills=(
                50.0,
                80.0,
            ),
            away_provenance=(
                OBSERVED,
                OBSERVED,
            ),
        ),
    }

    monkeypatch.setattr(
        imputed_strength,
        "build_arrays_for_matches",
        fake_base_arrays,
    )
    monkeypatch.setattr(
        imputed_strength,
        ("reconstruct_past_only_" "match_strength"),
        lambda match, **kwargs: (results[match.id]),
    )

    results_before = deepcopy(results)

    (
        training,
        validation,
        diagnostics,
    ) = build_fold_imputed_arrays(
        training_matches=[training_match],
        validation_matches=[validation_match],
        cat_maps=SimpleNamespace(),
        competition_names=("League A",),
        mode="binary_u25",
        max_goals_class=10,
        context=make_context(),
        imputation_config=(
            StrengthImputationConfig(
                skill_count=2,
                minimum_group_support=1,
                neutral_value=50.0,
            )
        ),
    )

    np.testing.assert_allclose(
        training[4][0],
        np.asarray(
            [
                [
                    [
                        0.20,
                        0.40,
                    ]
                ],
                [
                    [
                        1.0,
                        1.0,
                    ]
                ],
                [
                    [
                        0.30,
                        0.60,
                    ]
                ],
                [
                    [
                        1.0,
                        1.0,
                    ]
                ],
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        validation[4][0],
        np.asarray(
            [
                [
                    [
                        0.25,
                        0.50,
                    ]
                ],
                [
                    [
                        0.0,
                        0.0,
                    ]
                ],
                [
                    [
                        0.50,
                        0.80,
                    ]
                ],
                [
                    [
                        1.0,
                        1.0,
                    ]
                ],
            ],
            dtype=np.float32,
        ),
    )

    assert validation[5].tolist() == [
        [
            1,
        ]
    ]
    assert validation[6].tolist() == [
        [
            1,
        ]
    ]

    assert diagnostics.training_team_count == 2
    assert diagnostics.training_observed_cells == 4
    assert dict(diagnostics.validation_provenance_counts) == {
        ("COMPETITION_" "POSITION_MEDIAN"): 2,
        "NEAREST_PAST_SOFIFA": 2,
    }

    assert results == results_before


def test_nonstructured_arrays_remain_unchanged(
    monkeypatch,
) -> None:
    match = SimpleNamespace(
        id=1,
        comp_name="League A",
    )
    raw = match_result(
        1,
        home_skills=(
            10.0,
            20.0,
        ),
        home_provenance=(
            OBSERVED,
            OBSERVED,
        ),
        away_skills=(
            30.0,
            40.0,
        ),
        away_provenance=(
            OBSERVED,
            OBSERVED,
        ),
    )

    monkeypatch.setattr(
        imputed_strength,
        "build_arrays_for_matches",
        fake_base_arrays,
    )
    monkeypatch.setattr(
        imputed_strength,
        ("reconstruct_past_only_" "match_strength"),
        lambda **kwargs: raw,
    )

    base = fake_base_arrays([match])

    training, validation, _ = build_fold_imputed_arrays(
        training_matches=[match],
        validation_matches=[match],
        cat_maps=SimpleNamespace(),
        competition_names=("League A",),
        mode="binary_u25",
        max_goals_class=10,
        context=make_context(),
        imputation_config=(
            StrengthImputationConfig(
                skill_count=2,
                minimum_group_support=1,
                neutral_value=50.0,
            )
        ),
    )

    for arrays in (
        training,
        validation,
    ):
        for index in (
            0,
            1,
            2,
            3,
            7,
        ):
            np.testing.assert_array_equal(
                arrays[index],
                base[index],
            )


def test_invalid_base_array_contract_is_rejected(
    monkeypatch,
) -> None:
    match = SimpleNamespace(
        id=1,
        comp_name="League A",
    )

    monkeypatch.setattr(
        imputed_strength,
        "build_arrays_for_matches",
        lambda *args, **kwargs: (np.zeros(1),) * 7,
    )

    with pytest.raises(
        RuntimeError,
        match="eight arrays",
    ):
        build_fold_imputed_arrays(
            training_matches=[match],
            validation_matches=[match],
            cat_maps=SimpleNamespace(),
            competition_names=("League A",),
            mode="binary_u25",
            max_goals_class=10,
            context=make_context(),
            imputation_config=(
                StrengthImputationConfig(
                    skill_count=2,
                    minimum_group_support=1,
                    neutral_value=50.0,
                )
            ),
        )
