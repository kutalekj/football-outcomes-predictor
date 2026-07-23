from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_outcomes.experiments.baselines import (
    BASELINE_MODEL_NAMES,
    BaselineConfig,
    ReferencePrediction,
    _binary_metrics,
    _render_summary,
    fit_logistic_regression_probabilities,
    majority_probabilities,
    prevalence_probabilities,
    validate_common_fold_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def reference_rows() -> list[ReferencePrediction]:
    return [
        ReferencePrediction(
            fold_number=1,
            round_index=26,
            match_id=10,
            match_datetime="2024-01-01T12:00:00",
            competition="League A",
            season=2024,
            y_true=1,
        ),
        ReferencePrediction(
            fold_number=1,
            round_index=26,
            match_id=11,
            match_datetime="2024-01-01T15:00:00",
            competition="League A",
            season=2024,
            y_true=0,
        ),
    ]


def prediction_rows() -> list[dict]:
    rows = []
    for model_name in BASELINE_MODEL_NAMES:
        for reference in reference_rows():
            rows.append(
                {
                    "model_name": model_name,
                    "fold_number": reference.fold_number,
                    "round_index": reference.round_index,
                    "match_id": reference.match_id,
                    "y_true": reference.y_true,
                    "probability_under_2_5": 0.5,
                }
            )
    return rows


def test_baseline_module_has_no_tensorflow_modeling_or_training_dependency() -> None:
    path = PROJECT_ROOT / "src" / "football_outcomes" / "experiments" / "baselines.py"
    source = path.read_text(encoding="utf-8")

    assert "tensorflow" not in source
    assert "football_outcomes.modeling" not in source
    assert "football_outcomes.training" not in source
    assert "fs_globals" not in source
    assert "Global" not in source
    assert "requests" not in source


def test_prevalence_baseline_uses_training_positive_rate() -> None:
    result = prevalence_probabilities(
        np.asarray([0, 1, 1, 0, 1]),
        3,
    )
    np.testing.assert_allclose(result, [0.6, 0.6, 0.6])


def test_majority_baseline_uses_hard_class_and_positive_tie_policy() -> None:
    positive = majority_probabilities(np.asarray([0, 1]), 2)
    negative = majority_probabilities(np.asarray([0, 0, 1]), 2)

    np.testing.assert_array_equal(positive, [1.0, 1.0])
    np.testing.assert_array_equal(negative, [0.0, 0.0])


def test_logistic_regression_is_deterministic_and_bounded() -> None:
    X_train = np.asarray(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ]
    )
    y_train = np.asarray([0, 0, 1, 1])
    X_validation = np.asarray([[-1.5, -1.0], [1.5, 1.0]])
    config = BaselineConfig(seed=123)

    first = fit_logistic_regression_probabilities(
        X_train,
        y_train,
        X_validation,
        config,
    )
    second = fit_logistic_regression_probabilities(
        X_train,
        y_train,
        X_validation,
        config,
    )

    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    assert first.fit_status == "fitted"
    assert first.feature_count == 2
    assert np.all(first.probabilities >= 0.0)
    assert np.all(first.probabilities <= 1.0)
    assert first.probabilities[0] < first.probabilities[1]


def test_logistic_single_class_training_uses_constant_fallback() -> None:
    result = fit_logistic_regression_probabilities(
        np.asarray([[0.0], [1.0]]),
        np.asarray([1, 1]),
        np.asarray([[2.0], [3.0]]),
        BaselineConfig(),
    )

    np.testing.assert_array_equal(result.probabilities, [1.0, 1.0])
    assert result.fit_status == "single-class-constant"


def test_binary_metrics_match_required_schema() -> None:
    result = _binary_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.4, 0.6, 0.9]),
    )

    assert result["prediction_count"] == 4
    assert result["positive_class_prevalence"] == 0.5
    assert result["accuracy_at_0_5"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["brier_score"] == pytest.approx(0.085)
    assert result["binary_log_loss"] > 0.0


def test_common_fold_validation_accepts_exact_rows() -> None:
    validate_common_fold_predictions(prediction_rows(), reference_rows())


def test_common_fold_validation_rejects_missing_model_rows() -> None:
    rows = [row for row in prediction_rows() if row["model_name"] != "training-majority"]

    with pytest.raises(RuntimeError, match="exact reference validation rows"):
        validate_common_fold_predictions(rows, reference_rows())


def test_common_fold_validation_rejects_reordered_matches() -> None:
    rows = prediction_rows()
    rows[0], rows[1] = rows[1], rows[0]

    with pytest.raises(RuntimeError, match="exact reference validation rows"):
        validate_common_fold_predictions(rows, reference_rows())


def test_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="logistic_c"):
        BaselineConfig(logistic_c=0.0)

    with pytest.raises(ValueError, match="logistic_max_iter"):
        BaselineConfig(logistic_max_iter=0)

    with pytest.raises(ValueError, match="logistic_solver"):
        BaselineConfig(logistic_solver="lbfgs")


def test_summary_names_all_required_models() -> None:
    reference = type(
        "Reference",
        (),
        {
            "run_id": "canary-1",
            "model_name": "v2-canary",
        },
    )()
    metrics = {
        "prediction_count": 2,
        "positive_class_prevalence": 0.5,
        "accuracy_at_0_5": 0.5,
        "roc_auc": 0.5,
        "brier_score": 0.25,
        "binary_log_loss": 0.693,
    }
    text = _render_summary(
        run_id="baselines-1",
        reference=reference,
        aggregate={
            "overall_ok": True,
            "validation_match_count": 2,
            "prediction_row_count": 6,
            "models": {name: metrics for name in BASELINE_MODEL_NAMES},
        },
        runtime={"total_seconds": 1.0},
    )

    assert "**PASS**" in text
    for name in BASELINE_MODEL_NAMES:
        assert name in text
    assert "numerical pre-match" in text
