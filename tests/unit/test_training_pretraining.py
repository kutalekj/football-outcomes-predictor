import ast
import inspect
from pathlib import Path

from football_outcomes.training import (
    pretraining,
    train_mlp_rolling,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pretraining_module_has_no_legacy_rolling_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "training" / "pretraining.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ImportFrom,
        )
        and node.module is not None
    }

    assert "football_outcomes.training." "train_mlp_rolling" not in imported_modules


def test_legacy_pretraining_name_is_direct_alias() -> None:
    assert train_mlp_rolling.train_strength_pretrain_rolling is pretraining.train_strength_pretrain_rolling


def test_pretraining_loop_includes_final_round() -> None:
    source = inspect.getsource(pretraining.train_strength_pretrain_rolling)
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
