import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Input,
)
from tensorflow.keras.models import Model

from football_outcomes.modeling import (
    compilation,
)
from football_outcomes.training import (
    train_mlp_rolling,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_config(
    *,
    model_version: str,
    mode: str = "binary_u25",
    use_auxiliary: bool = False,
    auxiliary_task: str | None = None,
):
    return SimpleNamespace(
        model_version=model_version,
        mode=mode,
        learning_rate=0.01,
        use_team_aux_head=(use_auxiliary),
        aux_task=auxiliary_task,
        aux_weight=0.2,
    )


def make_single_output_model(
    output_name: str,
) -> Model:
    inputs = Input(
        shape=(2,),
        name="features",
    )
    output = Dense(
        1,
        name=output_name,
    )(inputs)

    return Model(inputs, output)


def make_auxiliary_model() -> Model:
    inputs = Input(
        shape=(2,),
        name="features",
    )
    main_output = Dense(
        1,
        name="output_main",
    )(inputs)
    auxiliary_output = Dense(
        1,
        name="output_team_aux",
    )(inputs)

    return Model(
        inputs,
        [
            main_output,
            auxiliary_output,
        ],
    )


def test_compilation_module_has_no_training_dependency() -> None:
    path = PROJECT_ROOT / "src" / "football_outcomes" / "modeling" / "compilation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("football_outcomes.training") for module in imported_modules)

    assert train_mlp_rolling.compile_model_for_cfg is compilation.compile_model_for_config


def test_loss_policy_for_supported_tasks() -> None:
    binary_config = make_config(
        model_version="v2",
    )

    binary_loss, binary_metrics = compilation.main_loss_and_metrics_for_mode(binary_config)

    assert binary_loss == "binary_crossentropy"
    assert binary_metrics[0] == "accuracy"

    auxiliary_loss, auxiliary_metrics = compilation.auxiliary_loss_and_metrics_for_task("goals_reg")

    assert auxiliary_loss == "mae"
    assert auxiliary_metrics == ["mae"]

    with pytest.raises(
        ValueError,
        match="Unknown aux_task",
    ):
        (compilation.auxiliary_loss_and_metrics_for_task("unsupported"))


def test_v1_and_v2_main_models_are_compiled() -> None:
    v1_model = make_single_output_model("output_binary")
    v1_config = make_config(
        model_version="v1",
    )

    compilation.compile_model_for_config(
        v1_model,
        v1_config,
    )

    assert v1_model.loss == "binary_crossentropy"
    assert float(tf.keras.backend.get_value(v1_model.optimizer.learning_rate)) == pytest.approx(0.01)

    v2_model = make_single_output_model("output_main")
    v2_config = make_config(
        model_version="v2",
    )

    compilation.compile_model_for_config(
        v2_model,
        v2_config,
    )

    assert v2_model.loss == {"output_main": ("binary_crossentropy")}


def test_auxiliary_and_invalid_compilation_modes() -> None:
    model = make_auxiliary_model()
    config = make_config(
        model_version="v2",
        use_auxiliary=True,
        auxiliary_task="binary_u25",
    )

    compilation.compile_model_for_config(
        model,
        config,
    )

    assert set(model.loss) == {
        "output_main",
        "output_team_aux",
    }

    invalid_config = make_config(
        model_version="unsupported",
    )

    with pytest.raises(
        ValueError,
        match="Unknown model_version",
    ):
        compilation.compile_model_for_config(
            make_single_output_model("output"),
            invalid_config,
        )
