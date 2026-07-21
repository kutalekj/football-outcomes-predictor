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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.snapshots import load_snapshot
from football_outcomes.datasets.arrays import build_arrays_for_matches
from football_outcomes.datasets.mappings import build_categorical_maps
from football_outcomes.datasets.rounds import distribute_matches_into_rounds
from football_outcomes.evaluation.persistence import write_records_csv
from football_outcomes.experiments.manifest import (
    build_experiment_manifest,
    canonical_payload_sha256,
    collect_artifact_identities,
    collect_environment_identity,
    collect_git_identity,
    collect_snapshot_identity,
    derive_run_id,
    sha256_file,
    write_canonical_json,
    write_experiment_manifest,
)
from football_outcomes.utils.fs_feature_utils import match_sort_key
from football_outcomes.validation.selection import (
    SelectionValidationConfig,
    select_validation_matches,
)

EXPECTED_SNAPSHOT_SHA256 = "AEC8C575156346AB1C255433C3D1E92E" "8782A5A04666149B929EE336FE27A51C"
EXPECTED_SELECTED_MATCHES = 30469
EXPECTED_ARRAY_READY_MATCHES = 30468
EXPECTED_ROUND_COUNT = 320
BASELINE_MODEL_NAMES = (
    "training-prevalence",
    "training-majority",
    "logistic-regression",
)


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 123
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    logistic_solver: str = "liblinear"

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer.")

        if not math.isfinite(self.logistic_c) or self.logistic_c <= 0.0:
            raise ValueError("logistic_c must be a positive finite number.")

        if type(self.logistic_max_iter) is not int or self.logistic_max_iter <= 0:
            raise ValueError("logistic_max_iter must be a positive integer.")

        if self.logistic_solver != "liblinear":
            raise ValueError("logistic_solver must be 'liblinear'.")


@dataclass(frozen=True)
class ReferencePrediction:
    fold_number: int
    round_index: int
    match_id: int
    match_datetime: str
    competition: str
    season: int
    y_true: int


@dataclass(frozen=True)
class ReferenceRun:
    run_id: str
    model_name: str
    snapshot_sha256: str
    window_rounds: int
    fold_indices: tuple[int, ...]
    predictions_by_fold: Mapping[int, tuple[ReferencePrediction, ...]]
    manifest_sha256: str
    configuration_sha256: str
    folds_sha256: str
    predictions_sha256: str


@dataclass(frozen=True)
class LogisticPredictionResult:
    probabilities: np.ndarray
    fit_status: str
    feature_count: int


def _binary_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
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


def prevalence_probabilities(
    y_train: np.ndarray,
    validation_count: int,
) -> np.ndarray:
    target = _validated_binary_target(y_train)
    _validate_validation_count(validation_count)
    return np.full(validation_count, float(np.mean(target)), dtype=np.float64)


def majority_probabilities(
    y_train: np.ndarray,
    validation_count: int,
) -> np.ndarray:
    target = _validated_binary_target(y_train)
    _validate_validation_count(validation_count)
    majority = 1.0 if float(np.mean(target)) >= 0.5 else 0.0
    return np.full(validation_count, majority, dtype=np.float64)


def fit_logistic_regression_probabilities(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    config: BaselineConfig,
) -> LogisticPredictionResult:
    training = _validated_feature_matrix(X_train, name="training")
    validation = _validated_feature_matrix(X_validation, name="validation")
    target = _validated_binary_target(y_train)

    if training.shape[0] != target.size:
        raise ValueError("Training features and targets have different lengths.")

    if training.shape[1] != validation.shape[1]:
        raise ValueError("Training and validation feature widths differ.")

    classes = np.unique(target)

    if classes.size == 1:
        probabilities = np.full(
            validation.shape[0],
            float(classes[0]),
            dtype=np.float64,
        )
        return LogisticPredictionResult(
            probabilities=probabilities,
            fit_status="single-class-constant",
            feature_count=int(training.shape[1]),
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.logistic_c,
            max_iter=config.logistic_max_iter,
            solver=config.logistic_solver,
            random_state=config.seed,
            class_weight=None,
        ),
    )
    model.fit(training, target.astype(np.int32))
    probabilities = model.predict_proba(validation)[:, 1].astype(np.float64)

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Logistic regression produced non-finite probabilities.")

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError("Logistic regression probabilities lie outside [0, 1].")

    return LogisticPredictionResult(
        probabilities=probabilities,
        fit_status="fitted",
        feature_count=int(training.shape[1]),
    )


def validate_common_fold_predictions(
    rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[ReferencePrediction],
    *,
    model_names: Sequence[str] = BASELINE_MODEL_NAMES,
) -> None:
    if not rows or not reference_rows:
        raise ValueError("Baseline and reference prediction rows must not be empty.")

    expected = [
        (
            reference.fold_number,
            reference.round_index,
            reference.match_id,
            reference.y_true,
        )
        for reference in reference_rows
    ]

    for model_name in model_names:
        model_rows = [row for row in rows if row.get("model_name") == model_name]

        actual = [
            (
                _strict_int(row.get("fold_number"), "fold_number"),
                _strict_int(row.get("round_index"), "round_index"),
                _strict_int(row.get("match_id"), "match_id"),
                _strict_binary(row.get("y_true"), "y_true"),
            )
            for row in model_rows
        ]

        if actual != expected:
            raise RuntimeError(f"Baseline model {model_name!r} does not use the exact " "reference validation rows.")

        probabilities = np.asarray(
            [row.get("probability_under_2_5") for row in model_rows],
            dtype=np.float64,
        )

        if not np.isfinite(probabilities).all():
            raise ValueError("Baseline probabilities must be finite.")

        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("Baseline probabilities must lie in [0, 1].")

    expected_row_count = len(reference_rows) * len(tuple(model_names))
    if len(rows) != expected_row_count:
        raise RuntimeError("Unexpected baseline prediction row count.")


def _validated_binary_target(values: np.ndarray) -> np.ndarray:
    target = np.asarray(values, dtype=np.float64).reshape(-1)

    if target.size == 0 or not np.isfinite(target).all():
        raise ValueError("Binary targets must be non-empty and finite.")

    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("Binary targets must contain only 0 and 1.")

    return target


def _validated_feature_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"The {name} feature matrix must be non-empty and 2-D.")

    if not np.isfinite(matrix).all():
        raise ValueError(f"The {name} feature matrix must be finite.")

    return matrix


def _validate_validation_count(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("validation_count must be a positive integer.")


def _strict_int(value: object, name: str) -> int:
    if type(value) is int:
        return value

    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer.") from error
        return parsed

    raise ValueError(f"{name} must be an integer.")


def _strict_binary(value: object, name: str) -> int:
    parsed = _strict_int(value, name)
    if parsed not in (0, 1):
        raise ValueError(f"{name} must contain only 0 and 1.")
    return parsed


def _selection_config() -> SelectionValidationConfig:
    return SelectionValidationConfig(
        competitions=tuple(sett.COMPS_LEAGUE),
        first_season=sett.FIRST_SEASON,
        last_season_exclusive=sett.LAST_SEASON,
        excluded_competition_seasons=frozenset(sett.EXCLUDED_COMP_SEASONS),
        valid_round_ids_by_season=sett.LEAGUE_VALID_ROUND_IDS_BY_SEASON,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required reference artifact is missing: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required reference artifact is missing: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _artifact_hash_from_manifest(manifest: Mapping[str, Any], name: str) -> str:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Reference manifest has no artifact index.")

    for artifact in artifacts:
        if isinstance(artifact, Mapping) and artifact.get("relative_path") == name:
            digest = artifact.get("sha256")
            if not isinstance(digest, str):
                break
            return digest

    raise ValueError(f"Reference manifest does not index {name}.")


def _verify_reference_artifact(
    reference_directory: Path,
    manifest: Mapping[str, Any],
    name: str,
) -> str:
    expected = _artifact_hash_from_manifest(manifest, name)
    actual = sha256_file(reference_directory / name)

    if actual != expected:
        raise RuntimeError(f"Reference artifact hash mismatch for {name}.")

    return actual


def load_reference_run(
    reference_directory: Path,
    *,
    reference_model_name: str | None = None,
) -> ReferenceRun:
    root = reference_directory.resolve()
    manifest_path = root / "manifest.json"
    configuration_path = root / "configuration.json"
    folds_path = root / "folds.csv"
    predictions_path = root / "predictions.csv"

    manifest = _read_json(manifest_path)
    configuration = _read_json(configuration_path)
    fold_rows = _read_csv(folds_path)
    prediction_rows = _read_csv(predictions_path)

    if manifest.get("run_kind") not in ("canary", "benchmark"):
        raise ValueError("Reference run must be a canary or benchmark run.")

    target = manifest.get("target")
    if not isinstance(target, Mapping) or target.get("positive_class") != 1:
        raise ValueError("Reference run has incompatible target semantics.")

    snapshot = manifest.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("Reference manifest has no snapshot identity.")
    snapshot_sha256 = snapshot.get("sha256")
    if not isinstance(snapshot_sha256, str):
        raise ValueError("Reference snapshot SHA-256 is missing.")

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Reference run ID is missing.")

    configuration_hash = _verify_reference_artifact(
        root,
        manifest,
        "configuration.json",
    )
    folds_hash = _verify_reference_artifact(root, manifest, "folds.csv")
    predictions_hash = _verify_reference_artifact(root, manifest, "predictions.csv")

    model_names = sorted({row.get("model_name", "") for row in prediction_rows if row.get("model_name")})

    if reference_model_name is None:
        if len(model_names) != 1:
            raise ValueError(
                "Reference predictions contain multiple models; provide " "reference_model_name explicitly."
            )
        selected_model_name = model_names[0]
    else:
        if reference_model_name not in model_names:
            raise ValueError("Requested reference model is not present.")
        selected_model_name = reference_model_name

    selected_predictions = [row for row in prediction_rows if row.get("model_name") == selected_model_name]

    if not selected_predictions:
        raise ValueError("Reference model has no prediction rows.")

    fold_selection = configuration.get("fold_selection")
    if not isinstance(fold_selection, Mapping):
        raise ValueError("Reference configuration has no fold selection.")

    window_rounds = _strict_int(
        fold_selection.get("window_rounds"),
        "window_rounds",
    )
    zero_based_fold_indices = fold_selection.get("zero_based_fold_indices")

    if not isinstance(zero_based_fold_indices, list) or not zero_based_fold_indices:
        raise ValueError("Reference fold indices are missing.")

    fold_indices = tuple(_strict_int(value, "zero_based_fold_index") for value in zero_based_fold_indices)

    fold_rounds = {
        _strict_int(row.get("fold_number"), "fold_number"): _strict_int(
            row.get("round_index"),
            "round_index",
        )
        for row in fold_rows
    }

    grouped: dict[int, list[ReferencePrediction]] = defaultdict(list)

    for row in selected_predictions:
        fold_number = _strict_int(row.get("fold_number"), "fold_number")
        round_index = _strict_int(row.get("round_index"), "round_index")

        if fold_rounds.get(fold_number) != round_index:
            raise RuntimeError("Reference fold and prediction round indices differ.")

        grouped[fold_number].append(
            ReferencePrediction(
                fold_number=fold_number,
                round_index=round_index,
                match_id=_strict_int(row.get("match_id"), "match_id"),
                match_datetime=str(row.get("match_datetime")),
                competition=str(row.get("competition")),
                season=_strict_int(row.get("season"), "season"),
                y_true=_strict_binary(row.get("y_true"), "y_true"),
            )
        )

    expected_fold_numbers = tuple(range(1, len(fold_indices) + 1))
    if tuple(sorted(grouped)) != expected_fold_numbers:
        raise RuntimeError("Reference prediction folds are incomplete.")

    predictions_by_fold = {fold_number: tuple(grouped[fold_number]) for fold_number in expected_fold_numbers}

    all_match_ids = [
        prediction.match_id for fold_predictions in predictions_by_fold.values() for prediction in fold_predictions
    ]
    if len(all_match_ids) != len(set(all_match_ids)):
        raise RuntimeError("Reference validation matches are not unique.")

    return ReferenceRun(
        run_id=run_id,
        model_name=selected_model_name,
        snapshot_sha256=snapshot_sha256,
        window_rounds=window_rounds,
        fold_indices=fold_indices,
        predictions_by_fold=predictions_by_fold,
        manifest_sha256=sha256_file(manifest_path),
        configuration_sha256=configuration_hash,
        folds_sha256=folds_hash,
        predictions_sha256=predictions_hash,
    )


def _identifier_hash(values: Sequence[int]) -> str:
    return canonical_payload_sha256([int(value) for value in values])


def _render_summary(
    *,
    run_id: str,
    reference: ReferenceRun,
    aggregate: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str:
    lines = [
        "# Step 8 common-fold baselines",
        "",
        f"- Run ID: `{run_id}`",
        f"- Result: **{'PASS' if aggregate['overall_ok'] else 'FAIL'}**",
        f"- Reference run: `{reference.run_id}`",
        f"- Reference model: `{reference.model_name}`",
        f"- Validation matches: `{aggregate['validation_match_count']}`",
        f"- Prediction rows: `{aggregate['prediction_row_count']}`",
        f"- Runtime seconds: `{runtime['total_seconds']:.3f}`",
        "",
        "## Pooled metrics",
        "",
        "| Model | ROC AUC | Accuracy | Brier | Log loss |",
        "|---|---:|---:|---:|---:|",
    ]

    for model_name in BASELINE_MODEL_NAMES:
        metrics = aggregate["models"][model_name]
        auc = "n/a" if metrics["roc_auc"] is None else metrics["roc_auc"]
        lines.append(
            f"| {model_name} | {auc} | {metrics['accuracy_at_0_5']} | "
            f"{metrics['brier_score']} | {metrics['binary_log_loss']} |"
        )

    lines.extend(
        [
            "",
            "## Feature policy",
            "",
            "The logistic-regression baseline uses only the deterministic",
            "numerical pre-match feature array. Team IDs, competition IDs,",
            "SoFIFA strength tensors, masks, and player positions are excluded.",
            "",
            "## Acceptance",
            "",
            "Every baseline is fitted only on the declared training window and",
            "produces predictions for the exact ordered validation rows of the",
            "reference neural run.",
            "",
        ]
    )

    return "\n".join(lines)


def run_common_fold_baselines(
    *,
    repository_root: Path,
    snapshot_path: Path,
    reference_run_directory: Path,
    output_root: Path,
    config: BaselineConfig,
    command: Sequence[str],
    reference_model_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()

    repository_root = repository_root.resolve()
    output_root = output_root.resolve()
    reference = load_reference_run(
        reference_run_directory,
        reference_model_name=reference_model_name,
    )

    snapshot_identity = collect_snapshot_identity(
        snapshot_path,
        expected_sha256=EXPECTED_SNAPSHOT_SHA256,
    )

    if reference.snapshot_sha256 != snapshot_identity.sha256:
        raise RuntimeError("Reference run and baseline snapshot identities differ.")

    git_identity = collect_git_identity(repository_root)
    environment_identity = collect_environment_identity()
    bundle = load_snapshot(snapshot_path)
    selected = select_validation_matches(bundle.matches, _selection_config())
    array_ready = sorted(
        (match for match in selected if getattr(match, "features_before_match", None) is not None),
        key=match_sort_key,
    )
    rounds = distribute_matches_into_rounds(array_ready)

    if len(selected) != EXPECTED_SELECTED_MATCHES:
        raise RuntimeError("Selected match count changed.")
    if len(array_ready) != EXPECTED_ARRAY_READY_MATCHES:
        raise RuntimeError("Array-ready match count changed.")
    if len(rounds) != EXPECTED_ROUND_COUNT:
        raise RuntimeError("Chronological round count changed.")

    if any(index < reference.window_rounds or index >= len(rounds) for index in reference.fold_indices):
        raise RuntimeError("Reference fold indices are outside the available scope.")

    competition_names = tuple(sett.COMPS_LEAGUE)
    cat_maps = build_categorical_maps(
        matches=array_ready,
        competition_names=competition_names,
    )

    all_reference_rows = [
        prediction
        for fold_number in sorted(reference.predictions_by_fold)
        for prediction in reference.predictions_by_fold[fold_number]
    ]

    configuration = {
        "experiment_tier": "common-fold-baselines",
        "reference": {
            "run_id": reference.run_id,
            "model_name": reference.model_name,
            "manifest_sha256": reference.manifest_sha256,
            "configuration_sha256": reference.configuration_sha256,
            "folds_sha256": reference.folds_sha256,
            "predictions_sha256": reference.predictions_sha256,
        },
        "scope": {
            "selected_matches": len(selected),
            "array_ready_matches": len(array_ready),
            "round_count": len(rounds),
            "competition_count": len(cat_maps.comp_id_map),
            "team_count": len(cat_maps.team_id_map),
        },
        "fold_selection": {
            "window_rounds": reference.window_rounds,
            "fold_count": len(reference.fold_indices),
            "zero_based_fold_indices": list(reference.fold_indices),
            "one_based_round_indices": [index + 1 for index in reference.fold_indices],
            "validation_match_count": len(all_reference_rows),
            "validation_match_ids_sha256": _identifier_hash([row.match_id for row in all_reference_rows]),
        },
        "models": {
            "training-prevalence": {
                "probability": "training-window positive-class prevalence",
            },
            "training-majority": {
                "probability": "hard majority class from the training window",
                "tie_policy": "positive class",
            },
            "logistic-regression": {
                "C": config.logistic_c,
                "class_weight": None,
                "max_iter": config.logistic_max_iter,
                "random_state": config.seed,
                "solver": config.logistic_solver,
                "standardize": True,
            },
        },
        "feature_subset": {
            "array_index": 0,
            "description": "deterministic numerical pre-match features only",
            "excluded": [
                "team IDs",
                "competition IDs",
                "SoFIFA strength values",
                "strength masks",
                "player positions",
            ],
        },
        "seed": config.seed,
    }

    run_id = derive_run_id(
        run_kind="baselines",
        git_commit=git_identity.commit,
        snapshot_sha256=snapshot_identity.sha256,
        seed=config.seed,
        configuration=configuration,
    )
    run_directory = output_root / run_id

    if run_directory.exists():
        if not overwrite:
            raise FileExistsError(
                f"Baseline output already exists: {run_directory}. " "Use overwrite=True to replace it."
            )
        shutil.rmtree(run_directory)

    run_directory.mkdir(parents=True)

    configuration_path = run_directory / "configuration.json"
    folds_path = run_directory / "folds.csv"
    predictions_path = run_directory / "predictions.csv"
    fold_metrics_path = run_directory / "fold_metrics.csv"
    aggregate_path = run_directory / "aggregate_metrics.json"
    runtime_path = run_directory / "runtime.json"
    summary_path = run_directory / "summary.md"
    manifest_path = run_directory / "manifest.json"

    write_canonical_json(configuration_path, configuration)

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for fold_number, round_index in enumerate(reference.fold_indices, start=1):
        fold_clock = time.perf_counter()
        training_matches = [
            match
            for round_matches in rounds[round_index - reference.window_rounds : round_index]
            for match in round_matches
        ]
        validation_matches = list(rounds[round_index])
        reference_rows = reference.predictions_by_fold[fold_number]

        actual_validation_ids = tuple(int(match.id) for match in validation_matches)
        expected_validation_ids = tuple(row.match_id for row in reference_rows)

        if actual_validation_ids != expected_validation_ids:
            raise RuntimeError("Reconstructed validation order differs from the reference run.")

        training_arrays = build_arrays_for_matches(
            matches=training_matches,
            cat_maps=cat_maps,
            competition_names=competition_names,
            mode="binary_u25",
            max_goals_class=10,
        )
        validation_arrays = build_arrays_for_matches(
            matches=validation_matches,
            cat_maps=cat_maps,
            competition_names=competition_names,
            mode="binary_u25",
            max_goals_class=10,
        )

        y_train = _validated_binary_target(training_arrays[-1])
        y_validation = _validated_binary_target(validation_arrays[-1])
        reference_target = np.asarray(
            [row.y_true for row in reference_rows],
            dtype=np.float64,
        )

        if not np.array_equal(y_validation, reference_target):
            raise RuntimeError("Validation targets differ from the reference run.")

        X_train = _validated_feature_matrix(training_arrays[0], name="training")
        X_validation = _validated_feature_matrix(
            validation_arrays[0],
            name="validation",
        )

        prevalence = prevalence_probabilities(y_train, len(validation_matches))
        majority = majority_probabilities(y_train, len(validation_matches))
        logistic = fit_logistic_regression_probabilities(
            X_train,
            y_train,
            X_validation,
            config,
        )
        predictions = {
            "training-prevalence": prevalence,
            "training-majority": majority,
            "logistic-regression": logistic.probabilities,
        }

        positive_rate = float(np.mean(y_train))
        majority_class = 1 if positive_rate >= 0.5 else 0

        fold_rows.append(
            {
                "run_id": run_id,
                "reference_run_id": reference.run_id,
                "fold_number": fold_number,
                "round_index": round_index + 1,
                "training_round_start": round_index - reference.window_rounds + 1,
                "training_round_end": round_index,
                "training_matches": len(training_matches),
                "validation_matches": len(validation_matches),
                "training_match_ids_sha256": _identifier_hash([int(match.id) for match in training_matches]),
                "validation_match_ids_sha256": _identifier_hash(list(actual_validation_ids)),
                "training_positive_rate": positive_rate,
                "training_majority_class": majority_class,
                "numerical_feature_count": int(X_train.shape[1]),
            }
        )

        for model_name in BASELINE_MODEL_NAMES:
            probability = predictions[model_name]
            metrics = _binary_metrics(y_validation, probability)
            fit_status = logistic.fit_status if model_name == "logistic-regression" else "closed-form"

            fold_metric_rows.append(
                {
                    "run_id": run_id,
                    "reference_run_id": reference.run_id,
                    "model_name": model_name,
                    "fold_number": fold_number,
                    "round_index": round_index + 1,
                    "fit_status": fit_status,
                    "training_positive_rate": positive_rate,
                    "training_majority_class": majority_class,
                    "numerical_feature_count": int(X_train.shape[1]),
                    **metrics,
                }
            )

            for match, y_true, y_probability in zip(
                validation_matches,
                y_validation,
                probability,
            ):
                prediction_rows.append(
                    {
                        "run_id": run_id,
                        "model_name": model_name,
                        "fold_number": fold_number,
                        "round_index": round_index + 1,
                        "match_id": int(match.id),
                        "match_datetime": match.datetime.isoformat(),
                        "competition": str(match.comp_name),
                        "season": int(match.season),
                        "y_true": int(y_true),
                        "probability_under_2_5": float(y_probability),
                    }
                )

        runtime_rows.append(
            {
                "fold_number": fold_number,
                "round_index": round_index + 1,
                "seconds": float(time.perf_counter() - fold_clock),
            }
        )

    validate_common_fold_predictions(
        prediction_rows,
        all_reference_rows,
    )

    aggregate_models = {}
    for model_name in BASELINE_MODEL_NAMES:
        model_rows = [row for row in prediction_rows if row["model_name"] == model_name]
        aggregate_models[model_name] = _binary_metrics(
            np.asarray([row["y_true"] for row in model_rows], dtype=np.float64),
            np.asarray(
                [row["probability_under_2_5"] for row in model_rows],
                dtype=np.float64,
            ),
        )

    aggregate = {
        "overall_ok": True,
        "run_id": run_id,
        "reference_run_id": reference.run_id,
        "reference_model_name": reference.model_name,
        "fold_count": len(reference.fold_indices),
        "validation_match_count": len(all_reference_rows),
        "prediction_row_count": len(prediction_rows),
        "models": aggregate_models,
    }

    for path, rows in (
        (folds_path, fold_rows),
        (predictions_path, prediction_rows),
        (fold_metrics_path, fold_metric_rows),
    ):
        if not write_records_csv(path, rows):
            raise RuntimeError(f"Required CSV artifact was not written: {path}")

    write_canonical_json(aggregate_path, aggregate)

    finished_at = datetime.now(timezone.utc)
    runtime = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "total_seconds": float(time.perf_counter() - started_clock),
        "folds": runtime_rows,
    }
    write_canonical_json(runtime_path, runtime)

    summary_path.write_text(
        _render_summary(
            run_id=run_id,
            reference=reference,
            aggregate=aggregate,
            runtime=runtime,
        ),
        encoding="utf-8",
        newline="\n",
    )

    artifacts = collect_artifact_identities(
        (
            configuration_path,
            folds_path,
            predictions_path,
            fold_metrics_path,
            aggregate_path,
            runtime_path,
            summary_path,
        ),
        root=run_directory,
    )
    manifest = build_experiment_manifest(
        run_kind="baselines",
        command=command,
        created_at_utc=started_at,
        git=git_identity,
        snapshot=snapshot_identity,
        environment=environment_identity,
        seed=config.seed,
        configuration=configuration,
        artifacts=artifacts,
    )

    if manifest["run_id"] != run_id:
        raise RuntimeError("Manifest run ID does not match the output directory.")

    write_experiment_manifest(manifest_path, manifest)
    return run_directory
