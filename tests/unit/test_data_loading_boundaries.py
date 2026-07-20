import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_application_pipeline_uses_state_restoration() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "application" / "footystats_pipeline.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ImportFrom,
        )
        for alias in node.names
    }

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Call,
        )
        and isinstance(
            node.func,
            ast.Name,
        )
    }

    assert "apply_bundle_to_global" in imported_names
    assert "apply_bundle_to_global" in called_names

    assert "fill_globals_with_cache" not in imported_names
    assert "fill_globals_with_cache" not in called_names


def test_retrieval_module_does_not_restore_snapshots() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "fs_retrieve.py"
    source = source_path.read_text(encoding="utf-8")

    assert "apply_bundle_to_global" not in source
    assert "fill_globals_with_cache" not in source
    assert "load_snapshot" not in source
