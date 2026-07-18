import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACTIVE_MODULES = (
    PROJECT_ROOT / "src" / "football_outcomes" / "training" / "train_mlp_rolling.py",
    PROJECT_ROOT / "src" / "football_outcomes" / "training" / "fs_classical_baselines.py",
    PROJECT_ROOT / "scripts" / "main_footystats.py",
)

LEGACY_DATASET_MODULE = "football_outcomes.training." "fs_training_utils"


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def called_functions(
    tree: ast.Module,
    function_name: str,
) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function_name
    ]


def assert_keyword_present(
    calls: list[ast.Call],
    keyword: str,
) -> None:
    assert calls, "Expected at least one matching " "function call."

    for call in calls:
        keyword_names = {argument.arg for argument in call.keywords if argument.arg is not None}
        assert keyword in keyword_names


def test_active_modules_do_not_import_legacy_dataset_utilities() -> None:
    for path in ACTIVE_MODULES:
        tree = parse_module(path)

        imported_modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}

        assert LEGACY_DATASET_MODULE not in imported_modules, f"{path.name} still imports " f"{LEGACY_DATASET_MODULE}"


def test_active_dataset_calls_use_explicit_competition_order() -> None:
    trainer_tree = parse_module(ACTIVE_MODULES[0])
    baseline_tree = parse_module(ACTIVE_MODULES[1])
    main_tree = parse_module(ACTIVE_MODULES[2])

    assert_keyword_present(
        called_functions(
            trainer_tree,
            "build_arrays_for_matches",
        ),
        "competition_names",
    )
    assert_keyword_present(
        called_functions(
            baseline_tree,
            ("build_flat_tabular_" "arrays_for_matches"),
        ),
        "competition_names",
    )
    assert_keyword_present(
        called_functions(
            main_tree,
            "build_categorical_maps",
        ),
        "competition_names",
    )
    assert_keyword_present(
        called_functions(
            main_tree,
            "train_rolling",
        ),
        "competition_names",
    )
