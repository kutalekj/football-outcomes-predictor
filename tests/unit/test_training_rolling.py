import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_outcomes.modeling import (
    factory,
)
from football_outcomes.training import (
    rolling,
    train_mlp_rolling,
)
from football_outcomes.training.configs import (
    TrainConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def imported_modules(
    relative_path: str,
) -> set[str]:
    source_path = PROJECT_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ImportFrom,
        )
        and node.module is not None
    }


def test_factory_has_no_training_dependency() -> None:
    imports = imported_modules(("src/football_outcomes/" "modeling/factory.py"))

    assert not any(module.startswith("football_outcomes.training") for module in imports)


def test_rolling_has_no_legacy_module_dependency() -> None:
    imports = imported_modules(("src/football_outcomes/" "training/rolling.py"))

    assert "football_outcomes.training." "train_mlp_rolling" not in imports


def test_legacy_exports_are_direct_aliases() -> None:
    assert train_mlp_rolling.build_model is factory.build_model
    assert train_mlp_rolling.train_rolling is rolling.train_rolling


def test_main_rolling_loop_includes_final_round() -> None:
    source = inspect.getsource(rolling.train_rolling)
    tree = ast.parse(source)

    range_stops = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.For,
        ):
            continue

        iterator = node.iter

        if not (
            isinstance(iterator, ast.Call)
            and isinstance(
                iterator.func,
                ast.Name,
            )
            and iterator.func.id == "range"
            and len(iterator.args) >= 2
        ):
            continue

        range_stops.append(ast.unparse(iterator.args[1]))

    assert "len(rounds)" in range_stops
    assert "len(rounds) - 1" not in range_stops


def test_enabled_imputation_requires_context() -> None:
    config = TrainConfig(enable_strength_imputation=True)

    with pytest.raises(
        ValueError,
        match=("no imputation context"),
    ):
        rolling.train_rolling(
            matches_sorted=[],
            cat_maps=SimpleNamespace(),
            cfg=config,
            competition_names=("Test League",),
        )


def test_rolling_contains_fold_local_imputation_boundary() -> None:
    source = inspect.getsource(rolling.train_rolling)

    assert "build_fold_imputed_arrays" in source
    assert "training_matches=train_ms" in source
    assert "validation_matches=val_ms" in source
