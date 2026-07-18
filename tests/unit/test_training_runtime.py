import ast
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from football_outcomes.training import (
    runtime,
    train_mlp_rolling,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_module_has_no_rolling_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "training" / "runtime.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "football_outcomes.training." "train_mlp_rolling" not in imported_modules


def test_legacy_runtime_names_are_direct_aliases() -> None:
    assert train_mlp_rolling.set_global_seed is runtime.set_global_seed
    assert train_mlp_rolling._make_train_targets is runtime.make_train_targets
    assert train_mlp_rolling._extract_main_predictions is runtime.extract_main_predictions


def test_global_seed_reproduces_random_sequences() -> None:
    runtime.set_global_seed(123)

    first_python = random.random()
    first_numpy = np.random.random()
    first_tensorflow = tf.random.uniform(
        shape=(3,),
        dtype=tf.float32,
    ).numpy()

    runtime.set_global_seed(123)

    assert random.random() == first_python
    assert np.random.random() == first_numpy
    np.testing.assert_array_equal(
        tf.random.uniform(
            shape=(3,),
            dtype=tf.float32,
        ).numpy(),
        first_tensorflow,
    )


def test_target_packaging_and_prediction_extraction() -> None:
    y_main = np.asarray(
        [1.0, 0.0],
        dtype=np.float32,
    )

    main_only_config = SimpleNamespace(
        model_version="v1",
        use_team_aux_head=False,
        aux_task=None,
        max_goals_class=10,
    )

    main_only = runtime.make_train_targets(
        matches=[],
        y_main=y_main,
        cfg=main_only_config,
    )

    assert main_only is y_main

    matches = [
        SimpleNamespace(
            home_goals=1,
            away_goals=1,
        ),
        SimpleNamespace(
            home_goals=2,
            away_goals=1,
        ),
    ]
    auxiliary_config = SimpleNamespace(
        model_version="v2",
        use_team_aux_head=True,
        aux_task="binary_u25",
        max_goals_class=10,
    )

    packaged = runtime.make_train_targets(
        matches=matches,
        y_main=y_main,
        cfg=auxiliary_config,
    )

    assert set(packaged) == {
        "output_main",
        "output_team_aux",
    }
    assert packaged["output_main"] is y_main
    np.testing.assert_array_equal(
        packaged["output_team_aux"],
        np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    assert runtime.extract_main_predictions(y_main) is y_main
    assert (
        runtime.extract_main_predictions(
            [
                y_main,
                packaged["output_team_aux"],
            ]
        )
        is y_main
    )
