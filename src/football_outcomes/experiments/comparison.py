from __future__ import annotations

import csv
import json
import math
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from football_outcomes.evaluation.persistence import write_records_csv
from football_outcomes.experiments.manifest import (
    build_experiment_manifest,
    collect_artifact_identities,
    collect_environment_identity,
    collect_git_identity,
    collect_snapshot_identity,
    derive_run_id,
    sha256_file,
    write_canonical_json,
    write_experiment_manifest,
)

BASELINE_MODEL_NAMES = (
    "training-prevalence",
    "training-majority",
    "logistic-regression",
)
SCOPE_ORDER = {
    "pooled": 0,
    "fold": 1,
    "competition": 2,
    "season": 3,
    "competition-season": 4,
}


@dataclass(frozen=True)
class ComparisonConfig:
    calibration_bins: int = 10
    neural_model_name: str = "v2-benchmark"

    def __post_init__(self) -> None:
        if type(self.calibration_bins) is not int or self.calibration_bins < 2:
            raise ValueError("calibration_bins must be an integer of at least 2.")

        if not self.neural_model_name:
            raise ValueError("neural_model_name must not be empty.")


@dataclass(frozen=True)
class LoadedRun:
    directory: Path
    manifest: Mapping[str, Any]
    predictions: tuple[dict[str, Any], ...]
    manifest_sha256: str
    predictions_sha256: str


def binary_metrics(
    y_true: Sequence[int | float],
    y_probability: Sequence[int | float],
) -> dict[str, float | int | None]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    probability = np.asarray(y_probability, dtype=np.float64).reshape(-1)

    if true.size == 0 or true.size != probability.size:
        raise ValueError("Targets and predictions must have equal non-zero length.")

    if not np.isfinite(true).all() or not np.isfinite(probability).all():
        raise ValueError("Targets and predictions must be finite.")

    if not np.isin(true, (0.0, 1.0)).all():
        raise ValueError("Binary targets must contain only 0 and 1.")

    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("Predicted probabilities must lie in [0, 1].")

    predicted = (probability >= 0.5).astype(np.float64)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    log_loss = -np.mean(true * np.log(clipped) + (1.0 - true) * np.log(1.0 - clipped))

    auc: float | None
    if np.unique(true).size < 2:
        auc = None
    else:
        auc = float(roc_auc_score(true, probability))

    return {
        "prediction_count": int(true.size),
        "positive_class_prevalence": float(np.mean(true)),
        "accuracy_at_0_5": float(np.mean(predicted == true)),
        "roc_auc": auc,
        "brier_score": float(np.mean((probability - true) ** 2)),
        "binary_log_loss": float(log_loss),
    }


def validate_common_prediction_rows(
    neural_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    neural_model_name: str = "v2-benchmark",
    baseline_model_names: Sequence[str] = BASELINE_MODEL_NAMES,
) -> None:
    if not neural_rows or not baseline_rows:
        raise ValueError("Neural and baseline prediction rows must not be empty.")

    neural_models = {str(row.get("model_name", "")) for row in neural_rows}
    if neural_models != {neural_model_name}:
        raise RuntimeError("Neural predictions do not contain exactly the declared model.")

    expected = [_row_signature(row) for row in neural_rows]
    neural_match_ids = [signature[2] for signature in expected]
    if len(neural_match_ids) != len(set(neural_match_ids)):
        raise RuntimeError("Neural predictions contain duplicate match IDs.")

    expected_baseline_models = tuple(baseline_model_names)
    actual_baseline_models = {str(row.get("model_name", "")) for row in baseline_rows}
    if actual_baseline_models != set(expected_baseline_models):
        raise RuntimeError("Baseline predictions do not contain all required models.")

    for model_name in expected_baseline_models:
        model_rows = [row for row in baseline_rows if row.get("model_name") == model_name]
        actual = [_row_signature(row) for row in model_rows]
        if actual != expected:
            raise RuntimeError(f"Baseline model {model_name!r} does not use the exact neural rows.")

    for row in (*neural_rows, *baseline_rows):
        _validated_probability(row.get("probability_under_2_5"))


def build_scope_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_names: Sequence[str],
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Prediction rows must not be empty.")

    ordered_models = tuple(model_names)
    if len(ordered_models) != len(set(ordered_models)):
        raise ValueError("model_names must be unique.")

    grouped_rows: list[tuple[str, str, dict[str, Any], list[Mapping[str, Any]]]] = []
    grouped_rows.append(("pooled", "all", {}, list(rows)))

    fold_groups: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    competition_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    season_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    competition_season_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)

    for row in rows:
        fold_number = _strict_int(row.get("fold_number"), "fold_number")
        round_index = _strict_int(row.get("round_index"), "round_index")
        competition = _strict_text(row.get("competition"), "competition")
        season = _strict_int(row.get("season"), "season")

        fold_groups[(fold_number, round_index)].append(row)
        competition_groups[competition].append(row)
        season_groups[season].append(row)
        competition_season_groups[(competition, season)].append(row)

    for (fold_number, round_index), group in sorted(fold_groups.items()):
        grouped_rows.append(
            (
                "fold",
                str(round_index),
                {
                    "fold_number": fold_number,
                    "round_index": round_index,
                },
                group,
            )
        )

    for competition, group in sorted(competition_groups.items()):
        grouped_rows.append(
            (
                "competition",
                competition,
                {"competition": competition},
                group,
            )
        )

    for season, group in sorted(season_groups.items()):
        grouped_rows.append(
            (
                "season",
                str(season),
                {"season": season},
                group,
            )
        )

    for (competition, season), group in sorted(competition_season_groups.items()):
        grouped_rows.append(
            (
                "competition-season",
                f"{competition}|{season}",
                {
                    "competition": competition,
                    "season": season,
                },
                group,
            )
        )

    metric_rows: list[dict[str, Any]] = []
    for scope_type, scope_key, dimensions, group in grouped_rows:
        for model_name in ordered_models:
            model_rows = [row for row in group if row.get("model_name") == model_name]
            if not model_rows:
                raise RuntimeError(f"Scope {scope_type!r} {scope_key!r} has no rows for " f"model {model_name!r}.")

            metrics = binary_metrics(
                [_strict_binary(row.get("y_true"), "y_true") for row in model_rows],
                [_validated_probability(row.get("probability_under_2_5")) for row in model_rows],
            )
            metric_rows.append(
                {
                    "scope_type": scope_type,
                    "scope_key": scope_key,
                    **dimensions,
                    "model_name": model_name,
                    **metrics,
                }
            )

    metric_rows.sort(
        key=lambda row: (
            SCOPE_ORDER[str(row["scope_type"])],
            str(row["scope_key"]),
            ordered_models.index(str(row["model_name"])),
        )
    )
    return metric_rows


def build_calibration_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_names: Sequence[str],
    bin_count: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int]]]:
    if type(bin_count) is not int or bin_count < 2:
        raise ValueError("bin_count must be an integer of at least 2.")

    calibration_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, float | int]] = {}

    for model_name in model_names:
        model_rows = [row for row in rows if row.get("model_name") == model_name]
        if not model_rows:
            raise RuntimeError(f"No rows found for model {model_name!r}.")

        true = np.asarray(
            [_strict_binary(row.get("y_true"), "y_true") for row in model_rows],
            dtype=np.float64,
        )
        probability = np.asarray(
            [_validated_probability(row.get("probability_under_2_5")) for row in model_rows],
            dtype=np.float64,
        )
        bin_indices = np.minimum(
            (probability * bin_count).astype(np.int64),
            bin_count - 1,
        )

        weighted_gap = 0.0
        maximum_gap = 0.0
        nonempty_bins = 0

        for bin_index in range(bin_count):
            mask = bin_indices == bin_index
            count = int(np.sum(mask))
            lower = float(bin_index / bin_count)
            upper = float((bin_index + 1) / bin_count)

            if count:
                mean_probability = float(np.mean(probability[mask]))
                observed_rate = float(np.mean(true[mask]))
                absolute_gap = abs(mean_probability - observed_rate)
                weighted_gap += (count / true.size) * absolute_gap
                maximum_gap = max(maximum_gap, absolute_gap)
                nonempty_bins += 1
            else:
                mean_probability = None
                observed_rate = None
                absolute_gap = None

            calibration_rows.append(
                {
                    "model_name": model_name,
                    "bin_index": bin_index,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "prediction_count": count,
                    "mean_probability": mean_probability,
                    "observed_positive_rate": observed_rate,
                    "absolute_gap": absolute_gap,
                }
            )

        summaries[model_name] = {
            "bin_count": bin_count,
            "nonempty_bin_count": nonempty_bins,
            "expected_calibration_error": float(weighted_gap),
            "maximum_calibration_error": float(maximum_gap),
        }

    return calibration_rows, summaries


def build_comparison_payload(
    *,
    run_id: str,
    neural_run_id: str,
    baseline_run_id: str,
    neural_model_name: str,
    baseline_model_names: Sequence[str],
    pooled_rows: Sequence[Mapping[str, Any]],
    calibration_summaries: Mapping[str, Mapping[str, float | int]],
) -> dict[str, Any]:
    metrics_by_model = {
        str(row["model_name"]): {
            key: row[key]
            for key in (
                "prediction_count",
                "positive_class_prevalence",
                "accuracy_at_0_5",
                "roc_auc",
                "brier_score",
                "binary_log_loss",
            )
        }
        for row in pooled_rows
        if row.get("scope_type") == "pooled"
    }

    required_models = (neural_model_name, *tuple(baseline_model_names))
    if set(metrics_by_model) != set(required_models):
        raise RuntimeError("Pooled metrics do not contain the required models.")

    auc_ranking = sorted(
        (
            (model_name, metrics["roc_auc"])
            for model_name, metrics in metrics_by_model.items()
            if metrics["roc_auc"] is not None
        ),
        key=lambda item: (-float(item[1]), item[0]),
    )
    best_model = auc_ranking[0][0] if auc_ranking else None

    neural_metrics = metrics_by_model[neural_model_name]
    comparisons: dict[str, dict[str, float | None]] = {}
    for baseline_name in baseline_model_names:
        baseline_metrics = metrics_by_model[baseline_name]
        neural_auc = neural_metrics["roc_auc"]
        baseline_auc = baseline_metrics["roc_auc"]
        comparisons[baseline_name] = {
            "roc_auc_delta": (
                None if neural_auc is None or baseline_auc is None else float(neural_auc) - float(baseline_auc)
            ),
            "accuracy_delta": float(neural_metrics["accuracy_at_0_5"]) - float(baseline_metrics["accuracy_at_0_5"]),
            "brier_score_improvement": float(baseline_metrics["brier_score"]) - float(neural_metrics["brier_score"]),
            "binary_log_loss_improvement": float(baseline_metrics["binary_log_loss"])
            - float(neural_metrics["binary_log_loss"]),
        }

    prediction_counts = {int(metrics["prediction_count"]) for metrics in metrics_by_model.values()}
    if len(prediction_counts) != 1:
        raise RuntimeError("Models do not have equal pooled prediction counts.")

    return {
        "overall_ok": True,
        "run_id": run_id,
        "source_runs": {
            "neural": neural_run_id,
            "baselines": baseline_run_id,
        },
        "primary_metric": "roc_auc",
        "validation_match_count": prediction_counts.pop(),
        "model_count": len(required_models),
        "best_model_by_roc_auc": best_model,
        "roc_auc_ranking": [
            {
                "rank": rank,
                "model_name": model_name,
                "roc_auc": float(auc),
            }
            for rank, (model_name, auc) in enumerate(auc_ranking, start=1)
        ],
        "models": {
            model_name: {
                "metrics": metrics_by_model[model_name],
                "calibration": dict(calibration_summaries[model_name]),
            }
            for model_name in required_models
        },
        "neural_vs_baselines": comparisons,
    }


def render_comparison_summary(
    comparison: Mapping[str, Any],
    *,
    scope_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Step 8 benchmark comparison",
        "",
        f"- Run ID: `{comparison['run_id']}`",
        f"- Result: **{'PASS' if comparison['overall_ok'] else 'FAIL'}**",
        f"- Validation matches: `{comparison['validation_match_count']}`",
        f"- Primary metric: `{comparison['primary_metric']}`",
        f"- Best pooled ROC AUC: `{comparison['best_model_by_roc_auc']}`",
        "",
        "## Pooled metrics",
        "",
        "| Model | ROC AUC | Accuracy | Brier | Log loss | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for ranking_row in comparison["roc_auc_ranking"]:
        model_name = ranking_row["model_name"]
        model = comparison["models"][model_name]
        metrics = model["metrics"]
        calibration = model["calibration"]
        lines.append(
            f"| {model_name} | {_render_number(metrics['roc_auc'])} | "
            f"{_render_number(metrics['accuracy_at_0_5'])} | "
            f"{_render_number(metrics['brier_score'])} | "
            f"{_render_number(metrics['binary_log_loss'])} | "
            f"{_render_number(calibration['expected_calibration_error'])} |"
        )

    lines.extend(
        [
            "",
            "## Neural deltas against baselines",
            "",
            "Positive ROC AUC and accuracy deltas favor the neural model. ",
            "Positive Brier and log-loss improvements also favor the neural model.",
            "",
            "| Baseline | ROC AUC delta | Accuracy delta | Brier improvement | " "Log-loss improvement |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for baseline_name, deltas in comparison["neural_vs_baselines"].items():
        lines.append(
            f"| {baseline_name} | {_render_number(deltas['roc_auc_delta'])} | "
            f"{_render_number(deltas['accuracy_delta'])} | "
            f"{_render_number(deltas['brier_score_improvement'])} | "
            f"{_render_number(deltas['binary_log_loss_improvement'])} |"
        )

    scope_counts: dict[str, int] = defaultdict(int)
    seen_scope_keys: set[tuple[str, str]] = set()
    for row in scope_rows:
        key = (str(row["scope_type"]), str(row["scope_key"]))
        if key not in seen_scope_keys:
            scope_counts[key[0]] += 1
            seen_scope_keys.add(key)

    lines.extend(
        [
            "",
            "## Reporting scope",
            "",
            "| Scope | Groups |",
            "|---|---:|",
        ]
    )
    for scope_type in SCOPE_ORDER:
        lines.append(f"| {scope_type} | {scope_counts.get(scope_type, 0)} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report compares all four models on identical chronological",
            "out-of-sample matches. Structural acceptance does not require the",
            "neural benchmark to outperform a baseline; performance differences",
            "are recorded as evidence for subsequent experiments.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark_comparison(
    *,
    repository_root: Path,
    snapshot_path: Path,
    neural_run: Path,
    baseline_run: Path,
    output_root: Path,
    config: ComparisonConfig,
    command: Sequence[str],
    overwrite: bool = False,
) -> Path:
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()

    neural = _load_run(neural_run, expected_run_kind="benchmark")
    baselines = _load_run(baseline_run, expected_run_kind="baselines")
    _validate_source_relationship(neural, baselines, config)

    validate_common_prediction_rows(
        neural.predictions,
        baselines.predictions,
        neural_model_name=config.neural_model_name,
    )

    snapshot_sha256 = _strict_text(
        neural.manifest.get("snapshot", {}).get("sha256"),
        "neural snapshot SHA-256",
    )
    snapshot_identity = collect_snapshot_identity(
        snapshot_path,
        expected_sha256=snapshot_sha256,
    )
    git_identity = collect_git_identity(repository_root)
    environment_identity = collect_environment_identity()

    model_names = (config.neural_model_name, *BASELINE_MODEL_NAMES)
    combined_rows = [*neural.predictions, *baselines.predictions]
    configuration = {
        "experiment_tier": "full-benchmark-comparison",
        "source_runs": {
            "neural": {
                "run_id": neural.manifest["run_id"],
                "model_name": config.neural_model_name,
                "manifest_sha256": neural.manifest_sha256,
                "predictions_sha256": neural.predictions_sha256,
            },
            "baselines": {
                "run_id": baselines.manifest["run_id"],
                "model_names": list(BASELINE_MODEL_NAMES),
                "manifest_sha256": baselines.manifest_sha256,
                "predictions_sha256": baselines.predictions_sha256,
            },
        },
        "model_names": list(model_names),
        "primary_metric": "roc_auc",
        "calibration_bins": config.calibration_bins,
        "reporting_scopes": list(SCOPE_ORDER),
    }
    run_id = derive_run_id(
        run_kind="comparison",
        git_commit=git_identity.commit,
        snapshot_sha256=snapshot_identity.sha256,
        seed=None,
        configuration=configuration,
    )
    run_directory = output_root / run_id

    if run_directory.exists():
        if not overwrite:
            raise FileExistsError(
                f"Comparison output already exists: {run_directory}. " "Use overwrite=True to replace it."
            )
        shutil.rmtree(run_directory)

    run_directory.mkdir(parents=True)
    configuration_path = run_directory / "configuration.json"
    pooled_path = run_directory / "pooled_metrics.csv"
    scopes_path = run_directory / "scope_metrics.csv"
    calibration_path = run_directory / "calibration.csv"
    comparison_path = run_directory / "comparison.json"
    runtime_path = run_directory / "runtime.json"
    summary_path = run_directory / "summary.md"
    manifest_path = run_directory / "manifest.json"

    write_canonical_json(configuration_path, configuration)
    scope_rows = build_scope_metric_rows(combined_rows, model_names=model_names)
    pooled_rows = [row for row in scope_rows if row["scope_type"] == "pooled"]
    calibration_rows, calibration_summaries = build_calibration_rows(
        combined_rows,
        model_names=model_names,
        bin_count=config.calibration_bins,
    )
    comparison = build_comparison_payload(
        run_id=run_id,
        neural_run_id=str(neural.manifest["run_id"]),
        baseline_run_id=str(baselines.manifest["run_id"]),
        neural_model_name=config.neural_model_name,
        baseline_model_names=BASELINE_MODEL_NAMES,
        pooled_rows=pooled_rows,
        calibration_summaries=calibration_summaries,
    )

    for path, rows in (
        (pooled_path, pooled_rows),
        (scopes_path, scope_rows),
        (calibration_path, calibration_rows),
    ):
        if not write_records_csv(path, rows):
            raise RuntimeError(f"Required CSV artifact was not written: {path}")

    write_canonical_json(comparison_path, comparison)
    runtime = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc),
        "total_seconds": float(time.perf_counter() - started_clock),
    }
    write_canonical_json(runtime_path, runtime)
    summary_path.write_text(
        render_comparison_summary(comparison, scope_rows=scope_rows),
        encoding="utf-8",
        newline="\n",
    )

    artifacts = collect_artifact_identities(
        (
            configuration_path,
            pooled_path,
            scopes_path,
            calibration_path,
            comparison_path,
            runtime_path,
            summary_path,
        ),
        root=run_directory,
    )
    manifest = build_experiment_manifest(
        run_kind="comparison",
        command=command,
        created_at_utc=started_at,
        git=git_identity,
        snapshot=snapshot_identity,
        environment=environment_identity,
        seed=None,
        configuration=configuration,
        artifacts=artifacts,
    )
    if manifest["run_id"] != run_id:
        raise RuntimeError("Manifest run ID does not match the output directory.")
    write_experiment_manifest(manifest_path, manifest)
    return run_directory


def _load_run(directory: Path, *, expected_run_kind: str) -> LoadedRun:
    resolved = directory.resolve()
    manifest_path = resolved / "manifest.json"
    predictions_path = resolved / "predictions.csv"

    manifest = _read_json(manifest_path)
    if manifest.get("run_kind") != expected_run_kind:
        raise RuntimeError(f"Expected run kind {expected_run_kind!r}, found " f"{manifest.get('run_kind')!r}.")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("Source manifest artifact index is missing.")

    artifact_by_path = {str(item.get("relative_path")): item for item in artifacts if isinstance(item, Mapping)}
    for relative_path, item in artifact_by_path.items():
        path = resolved / relative_path
        if not path.is_file():
            raise RuntimeError(f"Source artifact is missing: {path}")
        expected_hash = _strict_text(item.get("sha256"), "artifact SHA-256")
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"Source artifact hash mismatch: {path}")

    if "predictions.csv" not in artifact_by_path:
        raise RuntimeError("Source manifest does not index predictions.csv.")

    return LoadedRun(
        directory=resolved,
        manifest=manifest,
        predictions=tuple(_read_csv(predictions_path)),
        manifest_sha256=sha256_file(manifest_path),
        predictions_sha256=sha256_file(predictions_path),
    )


def _validate_source_relationship(
    neural: LoadedRun,
    baselines: LoadedRun,
    config: ComparisonConfig,
) -> None:
    neural_snapshot = neural.manifest.get("snapshot", {}).get("sha256")
    baseline_snapshot = baselines.manifest.get("snapshot", {}).get("sha256")
    if neural_snapshot != baseline_snapshot:
        raise RuntimeError("Neural and baseline snapshots differ.")

    reference = baselines.manifest.get("configuration", {}).get("reference", {})
    if reference.get("run_id") != neural.manifest.get("run_id"):
        raise RuntimeError("Baseline run does not reference the neural run.")

    if reference.get("model_name") != config.neural_model_name:
        raise RuntimeError("Baseline run references a different neural model.")


def _row_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _strict_int(row.get("fold_number"), "fold_number"),
        _strict_int(row.get("round_index"), "round_index"),
        _strict_int(row.get("match_id"), "match_id"),
        _strict_text(row.get("match_datetime"), "match_datetime"),
        _strict_text(row.get("competition"), "competition"),
        _strict_int(row.get("season"), "season"),
        _strict_binary(row.get("y_true"), "y_true"),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV artifact does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer.") from error
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer.")
    return numeric


def _strict_binary(value: Any, name: str) -> int:
    numeric = _strict_int(value, name)
    if numeric not in (0, 1):
        raise ValueError(f"{name} must contain only 0 and 1.")
    return numeric


def _strict_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text.")
    return value


def _validated_probability(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Prediction probabilities must be numeric.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Prediction probabilities must be numeric.") from error
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError("Prediction probabilities must lie in [0, 1].")
    return numeric


def _render_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"
