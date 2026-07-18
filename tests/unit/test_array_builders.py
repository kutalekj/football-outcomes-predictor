from types import SimpleNamespace

import numpy as np

from football_outcomes.config import fs_settings
from football_outcomes.datasets.arrays import (
    build_arrays_for_matches,
    build_flat_tabular_arrays_for_matches,
    build_strength_only_arrays_for_matches,
)
from football_outcomes.datasets.mappings import CatMaps
from football_outcomes.training.fs_training_utils import build_arrays_for_matches as build_legacy_arrays
from football_outcomes.training.fs_training_utils import (
    build_flat_tabular_arrays_for_matches as build_legacy_flat_arrays,
)
from football_outcomes.training.fs_training_utils import (
    build_strength_only_arrays_for_matches as build_legacy_strength_arrays,
)


class BuilderFeatures:
    def __init__(self) -> None:
        self.season = 0.5

        self.home_team_strength = np.full(
            (11, 34),
            80.0,
            dtype=np.float32,
        )
        self.away_team_strength = np.full(
            (11, 34),
            60.0,
            dtype=np.float32,
        )

        self.home_player_positions = [0] * 11
        self.away_player_positions = [1] * 11

    def __getattr__(self, name):
        return None


def make_match(
    competition_name: str,
):
    return SimpleNamespace(
        id=100,
        home_team=SimpleNamespace(id=10),
        away_team=SimpleNamespace(id=20),
        comp_name=competition_name,
        home_goals=1,
        away_goals=1,
        features_before_match=BuilderFeatures(),
    )


def test_explicit_builder_uses_supplied_competition_order() -> None:
    match = make_match("League B")

    category_maps = CatMaps(
        team_id_map={
            10: 0,
            20: 1,
        },
        comp_id_map={
            1: 0,
        },
    )

    (
        numerical,
        home_ids,
        away_ids,
        competition_ids,
        strength,
        home_positions,
        away_positions,
        targets,
    ) = build_arrays_for_matches(
        matches=[match],
        cat_maps=category_maps,
        competition_names=(
            "League A",
            "League B",
        ),
        mode="binary_u25",
    )

    assert numerical.shape[0] == 1
    assert numerical.dtype == np.float32

    np.testing.assert_array_equal(
        home_ids,
        np.asarray(
            [[0]],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        away_ids,
        np.asarray(
            [[1]],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        competition_ids,
        np.asarray(
            [[0]],
            dtype=np.int32,
        ),
    )

    assert strength.shape == (
        1,
        4,
        11,
        34,
    )
    np.testing.assert_allclose(
        strength[0, 0],
        0.8,
    )
    np.testing.assert_array_equal(
        strength[0, 1],
        np.ones(
            (11, 34),
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        strength[0, 2],
        0.6,
    )

    np.testing.assert_array_equal(
        home_positions,
        np.zeros(
            (1, 11),
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        away_positions,
        np.ones(
            (1, 11),
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        targets,
        np.asarray(
            [1.0],
            dtype=np.float32,
        ),
    )


def test_flat_and_strength_builders_match_main_arrays() -> None:
    match = make_match("League A")

    category_maps = CatMaps(
        team_id_map={
            10: 0,
            20: 1,
        },
        comp_id_map={
            0: 0,
        },
    )

    main_arrays = build_arrays_for_matches(
        matches=[match],
        cat_maps=category_maps,
        competition_names=("League A",),
        mode="binary_u25",
    )

    flat, flat_targets = build_flat_tabular_arrays_for_matches(
        matches=[match],
        cat_maps=category_maps,
        competition_names=("League A",),
        mode="binary_u25",
    )

    (
        strength,
        home_positions,
        away_positions,
        strength_targets,
    ) = build_strength_only_arrays_for_matches(
        matches=[match],
        mode="binary_u25",
    )

    np.testing.assert_array_equal(
        strength,
        main_arrays[4],
    )
    np.testing.assert_array_equal(
        home_positions,
        main_arrays[5],
    )
    np.testing.assert_array_equal(
        away_positions,
        main_arrays[6],
    )
    np.testing.assert_array_equal(
        strength_targets,
        main_arrays[7],
    )
    np.testing.assert_array_equal(
        flat_targets,
        main_arrays[7],
    )

    expected_width = main_arrays[0].shape[1] + 3 + int(np.prod(main_arrays[4].shape[1:])) + 22

    assert flat.shape == (
        1,
        expected_width,
    )
    assert flat.dtype == np.float32


def test_legacy_wrappers_match_explicit_builders() -> None:
    competition_name = fs_settings.COMPS_LEAGUE[0]
    match = make_match(competition_name)

    category_maps = CatMaps(
        team_id_map={
            10: 0,
            20: 1,
        },
        comp_id_map={
            0: 0,
        },
    )

    explicit_arrays = build_arrays_for_matches(
        matches=[match],
        cat_maps=category_maps,
        competition_names=(fs_settings.COMPS_LEAGUE),
        mode="binary_u25",
    )
    legacy_arrays = build_legacy_arrays(
        matches=[match],
        cat_maps=category_maps,
        mode="binary_u25",
    )

    for explicit, legacy in zip(
        explicit_arrays,
        legacy_arrays,
    ):
        np.testing.assert_array_equal(
            explicit,
            legacy,
        )

    explicit_flat = build_flat_tabular_arrays_for_matches(
        matches=[match],
        cat_maps=category_maps,
        competition_names=(fs_settings.COMPS_LEAGUE),
        mode="binary_u25",
    )
    legacy_flat = build_legacy_flat_arrays(
        matches=[match],
        cat_maps=category_maps,
        mode="binary_u25",
    )

    for explicit, legacy in zip(
        explicit_flat,
        legacy_flat,
    ):
        np.testing.assert_array_equal(
            explicit,
            legacy,
        )

    explicit_strength = build_strength_only_arrays_for_matches(
        matches=[match],
        mode="binary_u25",
    )
    legacy_strength = build_legacy_strength_arrays(
        matches=[match],
        mode="binary_u25",
    )

    for explicit, legacy in zip(
        explicit_strength,
        legacy_strength,
    ):
        np.testing.assert_array_equal(
            explicit,
            legacy,
        )
