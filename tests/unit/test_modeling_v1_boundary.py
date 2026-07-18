import ast
from pathlib import Path

import tensorflow as tf

from football_outcomes.modeling.v1 import build_model_v1 as build_model_v1_impl
from football_outcomes.training.train_mlp_rolling import (
    TrainConfig,
)
from football_outcomes.training.train_mlp_rolling import build_model_v1 as legacy_build_model_v1

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELING_FILES = (
    PROJECT_ROOT / "src" / "football_outcomes" / "modeling" / "common.py",
    PROJECT_ROOT / "src" / "football_outcomes" / "modeling" / "v1.py",
)


def model_signature(
    model: tf.keras.Model,
) -> tuple:
    return (
        tuple(tensor.name.split(":")[0] for tensor in model.inputs),
        tuple(model.output_names),
        model.count_params(),
    )


def test_modeling_modules_do_not_import_training() -> None:
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


def test_legacy_v1_builder_matches_extracted_builder() -> None:
    config = TrainConfig(
        model_version="v1",
        mode="binary_u25",
        enable_branch_diagnostics=False,
    )

    tf.keras.backend.clear_session()
    extracted_model = build_model_v1_impl(
        num_num=4,
        num_teams=3,
        num_comps=2,
        cfg=config,
    )
    extracted_signature = model_signature(extracted_model)

    tf.keras.backend.clear_session()
    legacy_model = legacy_build_model_v1(
        num_num=4,
        num_teams=3,
        num_comps=2,
        cfg=config,
    )
    legacy_signature = model_signature(legacy_model)

    assert legacy_signature == extracted_signature
