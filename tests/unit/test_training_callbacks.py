import ast
import csv
from pathlib import Path

from football_outcomes.training import (
    callbacks,
    train_mlp_rolling,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_callback_module_has_no_rolling_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "training" / "callbacks.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "football_outcomes.training." "train_mlp_rolling" not in imported_modules


def test_legacy_callback_names_are_direct_aliases() -> None:
    assert train_mlp_rolling.LayerDriftLogger is callbacks.LayerDriftLogger
    assert train_mlp_rolling.BranchProbeLogger is callbacks.BranchProbeLogger
    assert train_mlp_rolling.BranchDiagnosticsCsvLogger is callbacks.BranchDiagnosticsCsvLogger
    assert train_mlp_rolling.EpochMetricsCsvLogger is callbacks.EpochMetricsCsvLogger


def test_epoch_metrics_logger_writes_round_context(
    tmp_path,
) -> None:
    path = tmp_path / "epoch_metrics.csv"

    logger = callbacks.EpochMetricsCsvLogger(path)
    logger.set_round_context(
        round_idx=7,
        train_size=100,
        val_size=12,
        learning_rate=0.001,
    )
    logger.on_epoch_end(
        epoch=0,
        logs={
            "loss": 0.4,
            "accuracy": 0.75,
            "val_loss": 0.5,
        },
    )

    with path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["round_idx"] == "7"
    assert rows[0]["train_size"] == "100"
    assert rows[0]["val_size"] == "12"
    assert rows[0]["loss"] == "0.4"
    assert rows[0]["accuracy"] == "0.75"
    assert rows[0]["val_loss"] == "0.5"


def test_diagnostics_logger_initializes_csv(
    tmp_path,
) -> None:
    path = tmp_path / "diagnostics.csv"

    callbacks.BranchDiagnosticsCsvLogger(
        csv_path=path,
        drift_layer_names=[],
        probe_layer_names=[],
        probe_inputs=None,
    )

    with path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file)
        header = next(reader)

    assert header == [
        "diag_step",
        "round_idx",
        "train_size",
        "val_size",
        "local_epoch",
        "metric_family",
        "layer",
        "value",
        "probe_std",
    ]
