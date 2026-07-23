from __future__ import annotations

import ast
from pathlib import Path

import pytest

from football_outcomes.experiments.comparison import (
    BASELINE_MODEL_NAMES,
    ComparisonConfig,
    binary_metrics,
    build_calibration_rows,
    build_comparison_payload,
    build_scope_metric_rows,
    render_comparison_summary,
    validate_common_prediction_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = ("v2-benchmark", *BASELINE_MODEL_NAMES)


def row(
    *,
    model_name: str,
    match_id: int,
    fold_number: int = 1,
    round_index: int = 26,
    competition: str = "League A",
    season: int = 2024,
    y_true: int = 1,
    probability: float = 0.6,
) -> dict[str, object]:
    return {
        "run_id": "run",
        "model_name": model_name,
        "fold_number": fold_number,
        "round_index": round_index,
        "match_id": match_id,
        "match_datetime": f"2024-01-{match_id:02d}T12:00:00",
        "competition": competition,
        "season": season,
        "y_true": y_true,
        "probability_under_2_5": probability,
    }


def common_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    neural = [
        row(model_name="v2-benchmark", match_id=1, y_true=0, probability=0.4),
        row(model_name="v2-benchmark", match_id=2, y_true=1, probability=0.7),
    ]
    baselines = []
    for model_name in BASELINE_MODEL_NAMES:
        baselines.extend(
            [
                row(model_name=model_name, match_id=1, y_true=0, probability=0.5),
                row(model_name=model_name, match_id=2, y_true=1, probability=0.5),
            ]
        )
    return neural, baselines


def test_comparison_module_has_no_training_modeling_or_tensorflow_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "experiments" / "comparison.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}
    imported_names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}

    assert not any(
        module.startswith("football_outcomes.training") or module.startswith("football_outcomes.modeling")
        for module in modules
    )
    assert "tensorflow" not in imported_names


def test_binary_metrics_match_required_schema() -> None:
    metrics = binary_metrics([0, 1], [0.2, 0.8])

    assert metrics["prediction_count"] == 2
    assert metrics["positive_class_prevalence"] == 0.5
    assert metrics["accuracy_at_0_5"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["brier_score"] == pytest.approx(0.04)
    assert metrics["binary_log_loss"] > 0.0


def test_common_prediction_validation_accepts_exact_rows() -> None:
    neural, baselines = common_rows()
    validate_common_prediction_rows(neural, baselines)


def test_common_prediction_validation_rejects_metadata_change() -> None:
    neural, baselines = common_rows()
    baselines[1]["competition"] = "Different"

    with pytest.raises(RuntimeError, match="exact neural rows"):
        validate_common_prediction_rows(neural, baselines)


def test_common_prediction_validation_rejects_missing_model() -> None:
    neural, baselines = common_rows()
    baselines = [value for value in baselines if value["model_name"] != "training-majority"]

    with pytest.raises(RuntimeError, match="required models"):
        validate_common_prediction_rows(neural, baselines)


def test_scope_metrics_include_all_required_levels() -> None:
    rows = []
    for model_name in MODELS:
        rows.extend(
            [
                row(
                    model_name=model_name,
                    match_id=1,
                    y_true=0,
                    probability=0.4,
                ),
                row(
                    model_name=model_name,
                    match_id=2,
                    fold_number=2,
                    round_index=27,
                    competition="League B",
                    season=2025,
                    y_true=1,
                    probability=0.6,
                ),
            ]
        )

    metric_rows = build_scope_metric_rows(rows, model_names=MODELS)
    scope_types = {value["scope_type"] for value in metric_rows}

    assert scope_types == {
        "pooled",
        "fold",
        "competition",
        "season",
        "competition-season",
    }
    pooled = [value for value in metric_rows if value["scope_type"] == "pooled"]
    assert [value["model_name"] for value in pooled] == list(MODELS)


def test_calibration_bins_conserve_predictions() -> None:
    rows = [
        row(model_name="v2-benchmark", match_id=1, y_true=0, probability=0.05),
        row(model_name="v2-benchmark", match_id=2, y_true=1, probability=1.0),
    ]
    calibration_rows, summaries = build_calibration_rows(
        rows,
        model_names=("v2-benchmark",),
        bin_count=10,
    )

    assert len(calibration_rows) == 10
    assert sum(value["prediction_count"] for value in calibration_rows) == 2
    assert calibration_rows[-1]["prediction_count"] == 1
    assert summaries["v2-benchmark"]["nonempty_bin_count"] == 2
    assert summaries["v2-benchmark"]["expected_calibration_error"] >= 0.0


def test_comparison_delta_orientation() -> None:
    pooled_rows = []
    metrics = {
        "v2-benchmark": (0.60, 0.55, 0.23, 0.66),
        "training-prevalence": (0.50, 0.50, 0.25, 0.69),
        "training-majority": (0.50, 0.49, 0.51, 8.0),
        "logistic-regression": (0.58, 0.54, 0.24, 0.68),
    }
    for model_name, (auc, accuracy, brier, log_loss) in metrics.items():
        pooled_rows.append(
            {
                "scope_type": "pooled",
                "scope_key": "all",
                "model_name": model_name,
                "prediction_count": 100,
                "positive_class_prevalence": 0.5,
                "accuracy_at_0_5": accuracy,
                "roc_auc": auc,
                "brier_score": brier,
                "binary_log_loss": log_loss,
            }
        )
    calibration = {
        model_name: {
            "bin_count": 10,
            "nonempty_bin_count": 5,
            "expected_calibration_error": 0.1,
            "maximum_calibration_error": 0.2,
        }
        for model_name in MODELS
    }

    payload = build_comparison_payload(
        run_id="comparison-test",
        neural_run_id="neural",
        baseline_run_id="baselines",
        neural_model_name="v2-benchmark",
        baseline_model_names=BASELINE_MODEL_NAMES,
        pooled_rows=pooled_rows,
        calibration_summaries=calibration,
    )

    assert payload["best_model_by_roc_auc"] == "v2-benchmark"
    delta = payload["neural_vs_baselines"]["training-prevalence"]
    assert delta["roc_auc_delta"] == pytest.approx(0.10)
    assert delta["brier_score_improvement"] == pytest.approx(0.02)
    assert delta["binary_log_loss_improvement"] == pytest.approx(0.03)


def test_summary_names_models_and_acceptance() -> None:
    neural, baselines = common_rows()
    combined = [*neural, *baselines]
    scope_rows = build_scope_metric_rows(combined, model_names=MODELS)
    _, calibration = build_calibration_rows(
        combined,
        model_names=MODELS,
        bin_count=10,
    )
    comparison = build_comparison_payload(
        run_id="comparison-test",
        neural_run_id="neural",
        baseline_run_id="baselines",
        neural_model_name="v2-benchmark",
        baseline_model_names=BASELINE_MODEL_NAMES,
        pooled_rows=[value for value in scope_rows if value["scope_type"] == "pooled"],
        calibration_summaries=calibration,
    )

    summary = render_comparison_summary(comparison, scope_rows=scope_rows)

    assert "Result: **PASS**" in summary
    for model_name in MODELS:
        assert model_name in summary
    assert "competition-season" in summary


@pytest.mark.parametrize("bin_count", [0, 1, 1.5])
def test_configuration_rejects_invalid_calibration_bins(bin_count) -> None:
    with pytest.raises(ValueError, match="calibration_bins"):
        ComparisonConfig(calibration_bins=bin_count)
