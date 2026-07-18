import ast
from pathlib import Path

import tensorflow as tf

from football_outcomes.modeling import (
    strength_pretraining,
)
from football_outcomes.modeling.team_strength import (
    abs_diff,
    build_team_repr_v2,
    split_strength_tensor,
    vec_diff,
)
from football_outcomes.modeling.v2 import build_model_v2 as build_model_v2_impl
from football_outcomes.training.train_mlp_rolling import (
    TrainConfig,
)
from football_outcomes.training.train_mlp_rolling import build_model_v2 as legacy_build_model_v2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELING_FILES = (
    PROJECT_ROOT / "src" / "football_outcomes" / "modeling" / "team_strength.py",
    PROJECT_ROOT / "src" / "football_outcomes" / "modeling" / "v2.py",
)


def model_signature(
    model: tf.keras.Model,
) -> tuple:
    return (
        tuple(tensor.name.split(":")[0] for tensor in model.inputs),
        tuple(model.output_names),
        model.count_params(),
    )


def test_v2_modeling_modules_do_not_import_training() -> None:
    for path in MODELING_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))

        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(
                node,
                ast.ImportFrom,
            )
            and node.module is not None
        }

        assert not any(module.startswith("football_outcomes.training") for module in imported_modules)


def test_legacy_v2_builder_matches_extracted_builder() -> None:
    config = TrainConfig(
        model_version="v2",
        mode="binary_u25",
        enable_branch_diagnostics=False,
        use_team_aux_head=False,
    )

    tf.keras.backend.clear_session()
    extracted_model = build_model_v2_impl(
        num_num=4,
        num_teams=3,
        num_comps=2,
        cfg=config,
    )
    extracted_signature = model_signature(extracted_model)

    tf.keras.backend.clear_session()
    legacy_model = legacy_build_model_v2(
        num_num=4,
        num_teams=3,
        num_comps=2,
        cfg=config,
    )
    legacy_signature = model_signature(legacy_model)

    assert legacy_signature == extracted_signature


def test_pretraining_uses_extracted_team_helpers() -> None:
    assert strength_pretraining.abs_diff is abs_diff
    assert strength_pretraining.vec_diff is vec_diff
    assert strength_pretraining.split_strength_tensor is split_strength_tensor
    assert strength_pretraining.build_team_repr_v2 is build_team_repr_v2
