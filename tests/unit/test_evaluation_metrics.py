import ast
from pathlib import Path

import numpy as np
import pytest

from football_outcomes.evaluation import (
    metrics,
)
from football_outcomes.training import (
    train_mlp_rolling,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_binary_summary() -> None:
    y_true = np.asarray(
        [0.0, 0.0, 1.0, 1.0],
        dtype=np.float32,
    )
    y_probability = np.asarray(
        [0.1, 0.8, 0.7, 0.9],
        dtype=np.float32,
    )

    summary = metrics.binary_summary(
        y_true,
        y_probability,
    )

    assert summary == pytest.approx(
        {
            "pooled_accuracy": 0.75,
            "pooled_brier": 0.1875,
            "pooled_auc": 0.75,
        }
    )


def test_regression_summary() -> None:
    summary = metrics.regression_summary(
        y_true=np.asarray(
            [1.0, 3.0],
            dtype=np.float32,
        ),
        y_prediction=np.asarray(
            [2.0, 1.0],
            dtype=np.float32,
        ),
    )

    assert summary == pytest.approx(
        {
            "pooled_mae": 1.5,
            "pooled_rmse": np.sqrt(2.5),
        }
    )


def test_multiclass_summary() -> None:
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float32,
    )

    summary = metrics.multiclass_summary(
        y_true=np.asarray(
            [0, 2],
            dtype=np.int32,
        ),
        probabilities=probabilities,
        max_goals_class=2,
    )

    expected_log_loss = float(
        np.mean(
            -np.log(
                np.asarray(
                    [0.8, 0.7],
                    dtype=np.float32,
                )
            )
        )
    )

    assert summary == pytest.approx(
        {
            "pooled_accuracy": 1.0,
            ("pooled_expected_" "goals_mae"): 0.35,
            "pooled_log_loss": (expected_log_loss),
        }
    )


def test_metrics_boundary_and_legacy_aliases() -> None:
    path = PROJECT_ROOT / "src" / "football_outcomes" / "evaluation" / "metrics.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("football_outcomes.training") for module in imported_modules)

    assert train_mlp_rolling._binary_summary is metrics.binary_summary
    assert train_mlp_rolling._reg_summary is metrics.regression_summary
    assert train_mlp_rolling._multiclass_summary is metrics.multiclass_summary
