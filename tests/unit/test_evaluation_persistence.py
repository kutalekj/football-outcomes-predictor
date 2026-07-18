import csv
import json

from football_outcomes.evaluation.persistence import (
    write_json,
    write_records_csv,
)


def test_records_csv_uses_sorted_union_of_keys(
    tmp_path,
) -> None:
    path = tmp_path / "records.csv"

    written = write_records_csv(
        path,
        [
            {
                "round": 1,
                "accuracy": 0.8,
            },
            {
                "round": 2,
                "loss": 0.4,
            },
        ],
    )

    assert written is True

    with path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert list(rows[0]) == [
        "accuracy",
        "loss",
        "round",
    ]
    assert rows[0]["accuracy"] == "0.8"
    assert rows[0]["loss"] == ""
    assert rows[1]["accuracy"] == ""
    assert rows[1]["loss"] == "0.4"


def test_empty_records_do_not_create_csv(
    tmp_path,
) -> None:
    path = tmp_path / "empty.csv"

    written = write_records_csv(
        path,
        [],
    )

    assert written is False
    assert not path.exists()


def test_json_payload_is_written(
    tmp_path,
) -> None:
    path = tmp_path / "nested" / "summary.json"
    payload = {
        "run_name": "test",
        "score": 0.75,
    }

    write_json(
        path,
        payload,
    )

    with path.open(
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    assert loaded == payload


def test_persistence_module_has_no_training_dependency() -> None:
    import ast
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    source_path = project_root / "src" / "football_outcomes" / "evaluation" / "persistence.py"

    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("football_outcomes.training") for module in imported_modules)
