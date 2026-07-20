import ast
from pathlib import Path

from football_outcomes.data import (
    snapshots,
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


def test_snapshot_module_has_no_global_dependency() -> None:
    imports = imported_modules(("src/football_outcomes/" "data/snapshots.py"))

    assert "football_outcomes.config." "fs_globals" not in imports


def test_snapshot_module_has_no_csv_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "snapshots.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}

    assert "csv" not in imported_names


def test_sofifa_module_has_no_pickle_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "sofifa_ingestion.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}

    assert "pickle" not in imported_names


def test_missing_snapshot_returns_none(
    tmp_path,
) -> None:
    missing_path = tmp_path / "missing.pkl"

    assert snapshots.try_load_snapshot(missing_path) is None
