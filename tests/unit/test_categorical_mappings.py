from types import SimpleNamespace

import pytest

from football_outcomes.config import (
    fs_settings,
)
from football_outcomes.datasets.mappings import (
    build_categorical_maps,
)
from football_outcomes.training.fs_training_utils import build_categorical_maps as build_legacy_maps


def make_match(
    match_id: int,
    home_id: int,
    away_id: int,
    competition_name,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=match_id,
        home_team=SimpleNamespace(id=home_id),
        away_team=SimpleNamespace(id=away_id),
        comp_name=competition_name,
    )


def test_explicit_competition_order_is_used() -> None:
    matches = [
        make_match(
            1,
            home_id=20,
            away_id=10,
            competition_name="League A",
        ),
        make_match(
            2,
            home_id=10,
            away_id=30,
            competition_name="League C",
        ),
    ]

    maps = build_categorical_maps(
        matches,
        competition_names=(
            "League A",
            "League B",
            "League C",
        ),
    )

    assert maps.team_id_map == {
        10: 0,
        20: 1,
        30: 2,
    }
    assert maps.comp_id_map == {
        0: 0,
        2: 1,
    }


def test_only_present_competitions_receive_dense_ids() -> None:
    matches = [
        make_match(
            1,
            home_id=1,
            away_id=2,
            competition_name="League B",
        )
    ]

    maps = build_categorical_maps(
        matches,
        competition_names=(
            "Unused League",
            "League B",
        ),
    )

    assert maps.comp_id_map == {1: 0}


def test_missing_or_unknown_competition_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="comp_name=None",
    ):
        build_categorical_maps(
            [
                make_match(
                    1,
                    home_id=1,
                    away_id=2,
                    competition_name=None,
                )
            ],
            competition_names=("League A",),
        )

    with pytest.raises(
        ValueError,
        match="COMPS_LEAGUE",
    ):
        build_categorical_maps(
            [
                make_match(
                    2,
                    home_id=1,
                    away_id=2,
                    competition_name="Unknown",
                )
            ],
            competition_names=("League A",),
        )


def test_legacy_wrapper_uses_fs_settings_order() -> None:
    competition_name = fs_settings.COMPS_LEAGUE[0]

    maps = build_legacy_maps(
        [
            make_match(
                1,
                home_id=4,
                away_id=8,
                competition_name=competition_name,
            )
        ]
    )

    assert maps.team_id_map == {
        4: 0,
        8: 1,
    }
    assert maps.comp_id_map == {0: 0}
