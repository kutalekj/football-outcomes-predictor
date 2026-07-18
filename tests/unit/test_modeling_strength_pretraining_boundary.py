import ast
from pathlib import Path

import tensorflow as tf

from football_outcomes.modeling.strength_pretraining import build_strength_pretrain_model_v1 as extracted_v1
from football_outcomes.modeling.strength_pretraining import build_strength_pretrain_model_v2 as extracted_v2
from football_outcomes.training.train_mlp_rolling import (
    StrengthPretrainConfig,
)
from football_outcomes.training.train_mlp_rolling import build_strength_pretrain_model_v1 as legacy_v1
from football_outcomes.training.train_mlp_rolling import build_strength_pretrain_model_v2 as legacy_v2

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def signature(model):
    return (
        tuple(tensor.name.split(":")[0] for tensor in model.inputs),
        tuple(model.output_names),
        model.count_params(),
    )


def test_pretraining_module_does_not_import_training() -> None:
    path = PROJECT_ROOT / "src" / "football_outcomes" / "modeling" / "strength_pretraining.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}

    assert not any(module.startswith("football_outcomes.training") for module in imports)


def test_v1_pretraining_wrapper_matches_extracted_builder() -> None:
    config = StrengthPretrainConfig(
        branch_version="v1",
    )

    tf.keras.backend.clear_session()
    extracted = signature(extracted_v1(config))

    tf.keras.backend.clear_session()
    legacy = signature(legacy_v1(config))

    assert legacy == extracted


def test_v2_pretraining_wrapper_matches_extracted_builder() -> None:
    config = StrengthPretrainConfig(
        branch_version="v2",
    )

    tf.keras.backend.clear_session()
    extracted = signature(extracted_v2(config))

    tf.keras.backend.clear_session()
    legacy = signature(legacy_v2(config))

    assert legacy == extracted
