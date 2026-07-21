from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from football_outcomes.experiments.canary import (
    CanaryConfig,
    _render_summary,
    binary_metrics,
    choose_canary_fold_indices,
    validate_fold_chronology,
    validate_prediction_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_canary_module_does_not_use_global_or_rolling_trainer() -> None:
    path = PROJECT_ROOT / "src" / "football_outcomes" / "experiments" / "canary.py"
    source = path.read_text(encoding="utf-8")

    assert "fs_globals" not in source
    assert "Global" not in source
    assert "train_rolling" not in source
    assert "requests" not in source


def test_canary_indices_are_consecutive_and_one_window_late() -> None:
    assert choose_canary_fold_indices(
        round_count=320,
        window_rounds=25,
        fold_count=3,
        start_fold_offset=0,
    ) == (25, 26, 27)

    assert choose_canary_fold_indices(
        round_count=320,
        window_rounds=25,
        fold_count=2,
        start_fold_offset=4,
    ) == (29, 30)


def test_canary_indices_reject_out_of_range_request() -> None:
    with pytest.raises(ValueError, match="exceed"):
        choose_canary_fold_indices(
            round_count=30,
            window_rounds=25,
            fold_count=6,
            start_fold_offset=0,
        )


def test_binary_metrics_cover_required_canary_metrics() -> None:
    result = binary_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.4, 0.6, 0.9]),
    )

    assert result["prediction_count"] == 4
    assert result["positive_class_prevalence"] == 0.5
    assert result["accuracy_at_0_5"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["brier_score"] == pytest.approx(0.085)
    assert result["binary_log_loss"] > 0.0


def test_single_class_auc_is_explicitly_unavailable() -> None:
    result = binary_metrics(
        np.asarray([1, 1]),
        np.asarray([0.4, 0.8]),
    )

    assert result["roc_auc"] is None


def test_fold_chronology_accepts_equal_boundary_time() -> None:
    training = [
        SimpleNamespace(datetime=datetime(2024, 1, 1, 12)),
        SimpleNamespace(datetime=datetime(2024, 1, 2, 12)),
    ]
    validation = [
        SimpleNamespace(datetime=datetime(2024, 1, 2, 12)),
        SimpleNamespace(datetime=datetime(2024, 1, 3, 12)),
    ]

    training_max, validation_min = validate_fold_chronology(
        training,
        validation,
    )

    assert training_max == "2024-01-02T12:00:00"
    assert validation_min == "2024-01-02T12:00:00"


def test_fold_chronology_rejects_future_training_match() -> None:
    training = [SimpleNamespace(datetime=datetime(2024, 1, 3))]
    validation = [SimpleNamespace(datetime=datetime(2024, 1, 2))]

    with pytest.raises(RuntimeError, match="chronology"):
        validate_fold_chronology(training, validation)


def test_prediction_rows_require_unique_matches_and_valid_probabilities() -> None:
    valid = [
        {
            "match_id": 1,
            "y_true": 1,
            "probability_under_2_5": 0.7,
        },
        {
            "match_id": 2,
            "y_true": 0,
            "probability_under_2_5": 0.2,
        },
    ]
    validate_prediction_rows(valid)

    duplicate = [valid[0], dict(valid[0])]
    with pytest.raises(RuntimeError, match="more than once"):
        validate_prediction_rows(duplicate)

    invalid_probability = [dict(valid[0], probability_under_2_5=1.1)]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_prediction_rows(invalid_probability)


def test_canary_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="fold_count"):
        CanaryConfig(fold_count=0)

    with pytest.raises(ValueError, match="neutral_value"):
        CanaryConfig(neutral_value=101.0)

    with pytest.raises(ValueError, match="model_version"):
        CanaryConfig(model_version="v3")


def test_summary_contains_acceptance_and_metrics() -> None:
    text = _render_summary(
        run_id="canary-123",
        git_dirty=False,
        configuration={"model": {"model_version": "v2"}},
        aggregate={
            "overall_ok": True,
            "metrics": {
                "prediction_count": 10,
                "roc_auc": 0.55,
            },
        },
        fold_rows=[
            {
                "round_index": 26,
                "training_matches": 100,
                "validation_matches": 10,
            }
        ],
        runtime={"total_seconds": 1.25},
    )

    assert "**PASS**" in text
    assert "roc_auc" in text
    assert "carry-forward" in text.lower()
    assert "Round" in text
