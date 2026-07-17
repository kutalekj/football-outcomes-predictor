from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import tensorflow as tf

from football_outcomes.config import fs_settings as settings
from football_outcomes.data.fs_models import (
    FSMatch,
    FSMatchFeatures,
    FSTeam,
)
from football_outcomes.training.fs_classical_baselines import (
    evaluate_baseline_rolling,
)
from football_outcomes.training.fs_training_utils import (
    CatMaps,
    build_arrays_for_matches,
    build_strength_only_arrays_for_matches,
)
from football_outcomes.training.train_mlp_rolling import (
    TrainConfig,
    build_model,
    train_rolling,
    train_strength_pretrain_rolling,
)


def make_match_with_strength() -> tuple[FSMatch, CatMaps]:
    home_team = FSTeam(
        1,
        "Home Team",
        "home team",
        "Home Team",
        "Home Team",
        "HOME",
        "England",
    )
    away_team = FSTeam(
        2,
        "Away Team",
        "away team",
        "Away Team",
        "Away Team",
        "AWAY",
        "England",
    )

    competition_name = settings.COMPS_LEAGUE[0]

    features = FSMatchFeatures(
        comp_id=0,
        season=0.5,
        home_team_id=home_team.id,
        away_team_id=away_team.id,
        hours_sin=0.0,
        hours_cos=1.0,
        month_sin=0.0,
        month_cos=1.0,
    )

    features.home_team_strength = np.full(
        (11, 34),
        80.0,
        dtype=np.float32,
    )
    features.away_team_strength = np.full(
        (11, 34),
        60.0,
        dtype=np.float32,
    )

    features.home_player_positions = [0] * 11
    features.away_player_positions = [0] * 11

    match = FSMatch(100)
    match.home_team = home_team
    match.away_team = away_team
    match.comp_name = competition_name
    match.home_goals = 1
    match.away_goals = 1
    match.features_before_match = features

    category_maps = CatMaps(
        team_id_map={
            home_team.id: 0,
            away_team.id: 1,
        },
        comp_id_map={0: 0},
    )

    return match, category_maps


def test_main_array_builder_preserves_strength_tensor() -> None:
    match, category_maps = make_match_with_strength()

    main_arrays = build_arrays_for_matches(
        [match],
        category_maps,
        mode="binary_u25",
    )
    strength_only_arrays = build_strength_only_arrays_for_matches(
        [match],
        mode="binary_u25",
    )

    main_strength = main_arrays[4]
    reference_strength = strength_only_arrays[0]

    np.testing.assert_array_equal(
        main_strength,
        reference_strength,
    )

    assert main_strength.shape == (1, 4, 11, 34)

    np.testing.assert_allclose(
        main_strength[0, 0],
        0.8,
    )
    np.testing.assert_array_equal(
        main_strength[0, 1],
        np.ones((11, 34), dtype=np.float32),
    )
    np.testing.assert_allclose(
        main_strength[0, 2],
        0.6,
    )
    np.testing.assert_array_equal(
        main_strength[0, 3],
        np.ones((11, 34), dtype=np.float32),
    )


def rolling_range_stops(function) -> list[str]:
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)

    stops: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue

        iterator = node.iter

        if not isinstance(iterator, ast.Call):
            continue

        if not isinstance(iterator.func, ast.Name):
            continue

        if iterator.func.id != "range":
            continue

        if len(iterator.args) < 2:
            continue

        stops.append(ast.unparse(iterator.args[1]))

    return stops


def test_rolling_evaluators_include_final_round() -> None:
    functions = (
        train_rolling,
        train_strength_pretrain_rolling,
        evaluate_baseline_rolling,
    )

    stops_by_function = {function.__name__: rolling_range_stops(function) for function in functions}

    for function_name, stops in stops_by_function.items():
        assert "len(rounds)" in stops, f"{function_name} does not iterate through the final " f"round: {stops}"
        assert "len(rounds) - 1" not in stops, f"{function_name} still omits the final round: {stops}"

    train_source = textwrap.dedent(inspect.getsource(train_rolling))

    assert "total_train_rounds = max(" in train_source
    assert "len(rounds) - cfg.window_rounds" in train_source


def model_prediction(
    model: tf.keras.Model,
    inputs: list[np.ndarray],
) -> np.ndarray:
    output = model(inputs, training=False)

    if isinstance(output, (list, tuple)):
        output = output[0]

    return np.asarray(output.numpy())


def copy_inputs(
    inputs: list[np.ndarray],
) -> list[np.ndarray]:
    return [value.copy() for value in inputs]


def make_v2_inputs() -> list[np.ndarray]:
    numerical = np.zeros((2, 4), dtype=np.float32)

    home_ids = np.asarray(
        [[0], [1]],
        dtype=np.int32,
    )
    away_ids = np.asarray(
        [[1], [2]],
        dtype=np.int32,
    )
    competition_ids = np.asarray(
        [[0], [0]],
        dtype=np.int32,
    )

    strength = np.zeros(
        (2, 4, 11, 34),
        dtype=np.float32,
    )
    strength[:, 0, :, :] = 0.60
    strength[:, 1, :, :] = 1.00
    strength[:, 2, :, :] = 0.50
    strength[:, 3, :, :] = 1.00

    position_row = np.asarray(
        [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
        dtype=np.int32,
    )
    home_positions = np.tile(
        position_row,
        (2, 1),
    )
    away_positions = np.tile(
        position_row,
        (2, 1),
    )

    return [
        numerical,
        home_ids,
        away_ids,
        competition_ids,
        strength,
        home_positions,
        away_positions,
    ]


def build_test_v2_model(
    **config_overrides,
) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    tf.random.set_seed(123)
    np.random.seed(123)

    config_values = {
        "model_version": "v2",
        "mode": "binary_u25",
        "seed": 123,
        "enable_branch_diagnostics": False,
        "use_team_aux_head": False,
        "use_team_strength": True,
        "use_team_ids": True,
        "use_comp_embedding": True,
        "use_position_embedding": True,
        "use_strength_masks": True,
    }
    config_values.update(config_overrides)

    config = TrainConfig(**config_values)

    return build_model(
        num_num=4,
        num_teams=3,
        num_comps=2,
        cfg=config,
    )


def test_v2_branch_disable_flags_make_inputs_irrelevant() -> None:
    model = build_test_v2_model(
        use_team_strength=False,
        use_team_ids=False,
        use_comp_embedding=False,
        use_position_embedding=False,
        use_strength_masks=False,
    )

    base_inputs = make_v2_inputs()
    base_prediction = model_prediction(
        model,
        base_inputs,
    )

    altered_inputs: dict[str, list[np.ndarray]] = {}

    changed_team_ids = copy_inputs(base_inputs)
    changed_team_ids[1] = np.asarray(
        [[2], [0]],
        dtype=np.int32,
    )
    changed_team_ids[2] = np.asarray(
        [[0], [1]],
        dtype=np.int32,
    )
    altered_inputs["team IDs"] = changed_team_ids

    changed_competitions = copy_inputs(base_inputs)
    changed_competitions[3] = np.asarray(
        [[1], [1]],
        dtype=np.int32,
    )
    altered_inputs["competition IDs"] = changed_competitions

    changed_strength = copy_inputs(base_inputs)
    changed_strength[4][:, 0, :, :] = 0.05
    changed_strength[4][:, 2, :, :] = 0.95
    altered_inputs["strength"] = changed_strength

    changed_positions = copy_inputs(base_inputs)
    changed_positions[5] = np.flip(
        changed_positions[5],
        axis=1,
    ).copy()
    changed_positions[6] = np.flip(
        changed_positions[6],
        axis=1,
    ).copy()
    altered_inputs["positions"] = changed_positions

    for input_name, variant in altered_inputs.items():
        variant_prediction = model_prediction(
            model,
            variant,
        )

        np.testing.assert_allclose(
            variant_prediction,
            base_prediction,
            rtol=0.0,
            atol=1e-7,
            err_msg=(f"v2 predictions changed when disabled " f"{input_name} were altered"),
        )


def test_v2_disabled_position_embedding_ignores_positions() -> None:
    model = build_test_v2_model(
        use_position_embedding=False,
    )

    base_inputs = make_v2_inputs()
    base_prediction = model_prediction(
        model,
        base_inputs,
    )

    changed_inputs = copy_inputs(base_inputs)
    changed_inputs[5] = np.flip(
        changed_inputs[5],
        axis=1,
    ).copy()
    changed_inputs[6] = np.flip(
        changed_inputs[6],
        axis=1,
    ).copy()

    changed_prediction = model_prediction(
        model,
        changed_inputs,
    )

    np.testing.assert_allclose(
        changed_prediction,
        base_prediction,
        rtol=0.0,
        atol=1e-7,
    )


def test_v2_disabled_strength_masks_ignore_mask_channels() -> None:
    model = build_test_v2_model(
        use_strength_masks=False,
    )

    base_inputs = make_v2_inputs()
    base_prediction = model_prediction(
        model,
        base_inputs,
    )

    changed_inputs = copy_inputs(base_inputs)
    changed_inputs[4][:, 1, :, :] = 0.0
    changed_inputs[4][:, 3, :, :] = 0.0

    changed_prediction = model_prediction(
        model,
        changed_inputs,
    )

    np.testing.assert_allclose(
        changed_prediction,
        base_prediction,
        rtol=0.0,
        atol=1e-7,
    )
