from __future__ import annotations

import csv
import gc
import json
import math
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PUBLICATION_BINARY_MODEL_NAMES = (
    "proposed-v1",
    "probe-logistic",
    "probe-random-forest",
    "probe-xgboost",
    "flat-logistic",
    "flat-random-forest",
    "flat-xgboost",
    "flat-mlp",
)


@dataclass(frozen=True)
class BinaryEstimatorConfig:
    seed: int = 123

    logistic_c: float = 1.0
    logistic_max_iter: int = 1000

    random_forest_estimators: int = 120
    random_forest_max_depth: int | None = 12
    random_forest_min_samples_leaf: int = 2
    random_forest_max_features: str = "sqrt"
    random_forest_n_jobs: int = -1

    xgboost_estimators: int = 160
    xgboost_max_depth: int = 4
    xgboost_learning_rate: float = 0.05
    xgboost_subsample: float = 0.90
    xgboost_colsample_bytree: float = 0.80
    xgboost_reg_lambda: float = 1.0
    xgboost_n_jobs: int = -1

    mlp_hidden_layers: tuple[int, ...] = (128, 64)
    mlp_alpha: float = 1e-4
    mlp_learning_rate_init: float = 1e-3
    mlp_max_iter: int = 100
    mlp_batch_size: int = 64

    def validate(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer.")
        if not math.isfinite(self.logistic_c) or self.logistic_c <= 0.0:
            raise ValueError("logistic_c must be a positive finite number.")
        if type(self.logistic_max_iter) is not int or self.logistic_max_iter <= 0:
            raise ValueError("logistic_max_iter must be a positive integer.")
        if type(self.random_forest_estimators) is not int or self.random_forest_estimators <= 0:
            raise ValueError("random_forest_estimators must be positive.")
        if self.random_forest_max_depth is not None and (
            type(self.random_forest_max_depth) is not int or self.random_forest_max_depth <= 0
        ):
            raise ValueError("random_forest_max_depth must be positive or None.")
        if type(self.random_forest_min_samples_leaf) is not int or self.random_forest_min_samples_leaf <= 0:
            raise ValueError("random_forest_min_samples_leaf must be positive.")
        if type(self.xgboost_estimators) is not int or self.xgboost_estimators <= 0:
            raise ValueError("xgboost_estimators must be positive.")
        if type(self.xgboost_max_depth) is not int or self.xgboost_max_depth <= 0:
            raise ValueError("xgboost_max_depth must be positive.")
        for name in (
            "xgboost_learning_rate",
            "xgboost_subsample",
            "xgboost_colsample_bytree",
            "xgboost_reg_lambda",
            "mlp_alpha",
            "mlp_learning_rate_init",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number.")
        if self.xgboost_subsample > 1.0:
            raise ValueError("xgboost_subsample must not exceed 1.")
        if self.xgboost_colsample_bytree > 1.0:
            raise ValueError("xgboost_colsample_bytree must not exceed 1.")
        if not self.mlp_hidden_layers or any(type(width) is not int or width <= 0 for width in self.mlp_hidden_layers):
            raise ValueError("mlp_hidden_layers must contain positive integers.")
        if type(self.mlp_max_iter) is not int or self.mlp_max_iter <= 0:
            raise ValueError("mlp_max_iter must be positive.")
        if type(self.mlp_batch_size) is not int or self.mlp_batch_size <= 0:
            raise ValueError("mlp_batch_size must be positive.")


@dataclass(frozen=True)
class PublicationBinaryConfig:
    window_rounds: int = 25
    fold_count: int | None = None
    start_fold_offset: int = 0

    proposed_epochs_per_fold: int = 2
    proposed_batch_size: int = 64
    proposed_learning_rate: float = 8e-5
    proposed_lr_decay_rate: float = 0.997
    proposed_min_learning_rate: float = 2e-5

    seed: int = 123
    minimum_group_support: int = 20
    neutral_value: float = 50.0
    latent_batch_size: int = 256

    # Publication reproduction defaults: rebuild current normalized features from
    # raw snapshot match state and keep the historical selected-v1 strength path.
    rebuild_match_features: bool = True
    enable_strength_imputation: bool = False

    estimators: BinaryEstimatorConfig = BinaryEstimatorConfig()

    def validate(self) -> None:
        for name in (
            "window_rounds",
            "proposed_epochs_per_fold",
            "proposed_batch_size",
            "minimum_group_support",
            "latent_batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.fold_count is not None and (type(self.fold_count) is not int or self.fold_count <= 0):
            raise ValueError("fold_count must be a positive integer or None.")
        if type(self.start_fold_offset) is not int or self.start_fold_offset < 0:
            raise ValueError("start_fold_offset must be a non-negative integer.")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer.")
        for name in (
            "proposed_learning_rate",
            "proposed_lr_decay_rate",
            "proposed_min_learning_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number.")
        if self.proposed_lr_decay_rate > 1.0:
            raise ValueError("proposed_lr_decay_rate must not exceed 1.")
        if not math.isfinite(self.neutral_value) or self.neutral_value < 0.0 or self.neutral_value > 100.0:
            raise ValueError("neutral_value must be finite and in [0, 100].")
        self.estimators.validate()


@dataclass(frozen=True)
class ProbabilityResult:
    probabilities: np.ndarray
    fit_status: str
    feature_count: int


def choose_publication_fold_indices(
    *,
    round_count: int,
    window_rounds: int,
    fold_count: int | None,
    start_fold_offset: int,
) -> tuple[int, ...]:
    for name, value in (("round_count", round_count), ("window_rounds", window_rounds)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if fold_count is not None and (type(fold_count) is not int or fold_count <= 0):
        raise ValueError("fold_count must be a positive integer or None.")
    if type(start_fold_offset) is not int or start_fold_offset < 0:
        raise ValueError("start_fold_offset must be a non-negative integer.")

    first = window_rounds + start_fold_offset
    if first >= round_count:
        raise ValueError("No chronological validation fold is available.")
    stop = round_count if fold_count is None else first + fold_count
    if stop > round_count:
        raise ValueError("Requested folds exceed the available chronological rounds.")
    return tuple(range(first, stop))


def binary_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float | int | None]:
    true = _validated_binary_target(y_true)
    probability = np.asarray(y_probability, dtype=np.float64).reshape(-1)
    if probability.size != true.size or probability.size == 0:
        raise ValueError("Targets and probabilities must have equal non-zero length.")
    if not np.isfinite(probability).all():
        raise ValueError("Probabilities must be finite.")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("Probabilities must lie in [0, 1].")

    predicted = (probability >= 0.5).astype(np.int32)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    auc = None
    if np.unique(true).size >= 2:
        auc = float(roc_auc_score(true, probability))

    return {
        "prediction_count": int(true.size),
        "positive_class_prevalence": float(np.mean(true)),
        "accuracy_at_0_5": float(np.mean(predicted == true)),
        "roc_auc": auc,
        "brier_score": float(np.mean((probability - true) ** 2)),
        "binary_log_loss": float(-np.mean(true * np.log(clipped) + (1.0 - true) * np.log(1.0 - clipped))),
    }


def _validated_binary_target(values: np.ndarray) -> np.ndarray:
    target = np.asarray(values, dtype=np.int32).reshape(-1)
    if target.size == 0:
        raise ValueError("Binary target must not be empty.")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("Binary target must contain only 0 and 1.")
    return target


def _validated_feature_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} feature matrix must be a non-empty 2-D array.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} feature matrix must be finite.")
    return matrix


def _constant_or_none(
    y_train: np.ndarray,
    validation_count: int,
    feature_count: int,
) -> ProbabilityResult | None:
    target = _validated_binary_target(y_train)
    classes = np.unique(target)
    if classes.size != 1:
        return None
    return ProbabilityResult(
        probabilities=np.full(validation_count, float(classes[0]), dtype=np.float64),
        fit_status="single-class-constant",
        feature_count=feature_count,
    )


def fit_logistic_probabilities(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    config: BinaryEstimatorConfig,
    *,
    standardize: bool = True,
) -> ProbabilityResult:
    config.validate()
    training = _validated_feature_matrix(X_train, name="training")
    validation = _validated_feature_matrix(X_validation, name="validation")
    target = _validated_binary_target(y_train)
    _validate_shapes(training, target, validation)

    constant = _constant_or_none(target, len(validation), training.shape[1])
    if constant is not None:
        return constant

    estimator = LogisticRegression(
        C=config.logistic_c,
        max_iter=config.logistic_max_iter,
        solver="liblinear",
        random_state=config.seed,
    )
    model = make_pipeline(StandardScaler(), estimator) if standardize else estimator
    model.fit(training, target)
    return _probability_result(model.predict_proba(validation)[:, 1], training.shape[1])


def fit_random_forest_probabilities(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    config: BinaryEstimatorConfig,
) -> ProbabilityResult:
    config.validate()
    training = _validated_feature_matrix(X_train, name="training")
    validation = _validated_feature_matrix(X_validation, name="validation")
    target = _validated_binary_target(y_train)
    _validate_shapes(training, target, validation)

    constant = _constant_or_none(target, len(validation), training.shape[1])
    if constant is not None:
        return constant

    model = RandomForestClassifier(
        n_estimators=config.random_forest_estimators,
        max_depth=config.random_forest_max_depth,
        min_samples_leaf=config.random_forest_min_samples_leaf,
        max_features=config.random_forest_max_features,
        random_state=config.seed,
        n_jobs=config.random_forest_n_jobs,
        class_weight=None,
    )
    model.fit(training, target)
    return _probability_result(model.predict_proba(validation)[:, 1], training.shape[1])


def fit_xgboost_probabilities(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    config: BinaryEstimatorConfig,
) -> ProbabilityResult:
    config.validate()
    training = _validated_feature_matrix(X_train, name="training")
    validation = _validated_feature_matrix(X_validation, name="validation")
    target = _validated_binary_target(y_train)
    _validate_shapes(training, target, validation)

    constant = _constant_or_none(target, len(validation), training.shape[1])
    if constant is not None:
        return constant

    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError(
            "XGBoost is required for the PRL publication comparison. "
            "Install xgboost in the active project environment."
        ) from error

    model = XGBClassifier(
        n_estimators=config.xgboost_estimators,
        max_depth=config.xgboost_max_depth,
        learning_rate=config.xgboost_learning_rate,
        subsample=config.xgboost_subsample,
        colsample_bytree=config.xgboost_colsample_bytree,
        reg_lambda=config.xgboost_reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.seed,
        n_jobs=config.xgboost_n_jobs,
        verbosity=0,
    )
    model.fit(training, target)
    return _probability_result(model.predict_proba(validation)[:, 1], training.shape[1])


def fit_flat_mlp_probabilities(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    config: BinaryEstimatorConfig,
) -> ProbabilityResult:
    """Fit the fixed shallow feed-forward MLP used in Experiment III.

    The two-hidden-layer scikit-learn implementation is intentionally simple and
    fixed. L2 regularization is used instead of a dropout implementation so this
    baseline remains independent from the TensorFlow RNG/state of the proposed
    carry-forward neural model.
    """

    config.validate()
    training = _validated_feature_matrix(X_train, name="training")
    validation = _validated_feature_matrix(X_validation, name="validation")
    target = _validated_binary_target(y_train)
    _validate_shapes(training, target, validation)

    constant = _constant_or_none(target, len(validation), training.shape[1])
    if constant is not None:
        return constant

    model = MLPClassifier(
        hidden_layer_sizes=config.mlp_hidden_layers,
        activation="relu",
        solver="adam",
        alpha=config.mlp_alpha,
        batch_size=config.mlp_batch_size,
        learning_rate_init=config.mlp_learning_rate_init,
        max_iter=config.mlp_max_iter,
        shuffle=False,
        random_state=config.seed,
        early_stopping=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(training, target)
    convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return _probability_result(
        model.predict_proba(validation)[:, 1],
        training.shape[1],
        fit_status=("fitted-convergence-warning" if convergence_warning else "fitted"),
    )


def _validate_shapes(
    training: np.ndarray,
    target: np.ndarray,
    validation: np.ndarray,
) -> None:
    if training.shape[0] != target.size:
        raise ValueError("Training features and targets have different lengths.")
    if training.shape[1] != validation.shape[1]:
        raise ValueError("Training and validation feature widths differ.")


def _probability_result(
    values: np.ndarray,
    feature_count: int,
    *,
    fit_status: str = "fitted",
) -> ProbabilityResult:
    probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
    if probabilities.size == 0 or not np.isfinite(probabilities).all():
        raise RuntimeError("Estimator produced empty or non-finite probabilities.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError("Estimator probabilities lie outside [0, 1].")
    return ProbabilityResult(
        probabilities=probabilities,
        fit_status=fit_status,
        feature_count=int(feature_count),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _selected_v1_train_config(config: PublicationBinaryConfig) -> Any:
    # Keep the publication control tied to the application's authoritative
    # selected-v1 architecture instead of duplicating architecture constants here.
    from football_outcomes.application.footystats_pipeline import selected_model_config

    base = selected_model_config(
        run_name="prl-publication-v1",
        enable_strength_imputation=config.enable_strength_imputation,
    )
    return replace(
        base,
        window_rounds=config.window_rounds,
        epochs_per_step=config.proposed_epochs_per_fold,
        learning_rate=config.proposed_learning_rate,
        batch_size=config.proposed_batch_size,
        seed=config.seed,
        run_name=None,
        enable_branch_diagnostics=False,
        save_oos_predictions=False,
        lr_decay_rate=config.proposed_lr_decay_rate,
        min_learning_rate=config.proposed_min_learning_rate,
        strength_imputation_minimum_support=config.minimum_group_support,
        strength_imputation_neutral_value=config.neutral_value,
    )


def _validate_numerical_unit_interval(values: np.ndarray, *, name: str) -> dict[str, float]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise RuntimeError(f"{name} numerical feature matrix must be a non-empty 2-D array.")
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{name} numerical feature matrix contains non-finite values.")

    minimum = float(matrix.min())
    maximum = float(matrix.max())
    tolerance = 1e-6
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        row, column = np.argwhere((matrix < -tolerance) | (matrix > 1.0 + tolerance))[0]
        value = float(matrix[int(row), int(column)])
        raise RuntimeError(
            f"{name} numerical features must lie in [0, 1]; "
            f"found value={value} at row={int(row)}, column={int(column)} "
            f"with overall range [{minimum}, {maximum}]."
        )
    return {"minimum": minimum, "maximum": maximum}


def _rebuild_selected_match_features(bundle: Any) -> tuple[list[Any], list[Any]]:
    """Reproduce the active FootyStats feature-preparation path from raw snapshot state."""

    from football_outcomes.data.match_features import calculate_match_features
    from football_outcomes.data.sofifa_team_matching import match_fs_teams_to_sofifa_teams
    from football_outcomes.data.state import apply_bundle_to_global
    from football_outcomes.utils import fs_common as common
    from football_outcomes.utils import fs_feature_utils as feature_utils
    from football_outcomes.validation.selection import select_validation_matches

    global_instance = apply_bundle_to_global(bundle)
    common.link_matches_to_comp_seasons()
    common.ensure_comp_season_dates(force=False)
    common.initialize_league_tables(precompute_positions=True, force_rebuild=False)
    match_fs_teams_to_sofifa_teams(force=False)

    all_matches_sorted = sorted(global_instance.all_matches, key=feature_utils.match_sort_key)
    selected = sorted(
        select_validation_matches(all_matches_sorted, _selection_config()),
        key=feature_utils.match_sort_key,
    )
    team_index_all = feature_utils.build_team_match_index(all_matches_sorted)
    team_index_league = feature_utils.build_team_match_index(selected)

    array_ready: list[Any] = []
    for match in selected:
        try:
            match.features_before_match = calculate_match_features(
                match=match,
                team_index_league=team_index_league,
                team_index_all=team_index_all,
            )
        except ValueError as exc:
            print(
                f"[prl-binary][feature-skip] match_id={match.id} "
                f"competition={match.comp_name!r} season={match.season} error={exc!r}",
                flush=True,
            )
            continue
        array_ready.append(match)

    return selected, array_ready


def _selection_config() -> Any:
    from football_outcomes.config import fs_settings as sett
    from football_outcomes.validation.selection import SelectionValidationConfig

    return SelectionValidationConfig(
        competitions=tuple(sett.COMPS_LEAGUE),
        first_season=sett.FIRST_SEASON,
        last_season_exclusive=sett.LAST_SEASON,
        excluded_competition_seasons=frozenset(sett.EXCLUDED_COMP_SEASONS),
        valid_round_ids_by_season=sett.LEAGUE_VALID_ROUND_IDS_BY_SEASON,
    )


def _strength_context(bundle: Any) -> Any:
    from football_outcomes.config import fs_settings as sett
    from football_outcomes.data.sofifa_strength import PastOnlyStrengthConfig
    from football_outcomes.datasets.imputed_strength import StrengthImputationContext

    return StrengthImputationContext(
        snapshots=bundle.sofifa_snapshots,
        player_occurrences=bundle.sofifa_player_occurrences,
        fs_to_sofifa_cache=bundle.fs_to_sofifa_cache,
        reconstruction_config=PastOnlyStrengthConfig(
            player_count=sett.TEAM_STRENGTH_NUM_PLAYERS,
            skill_count=len(sett.PLAYER_SKILLS),
            max_age_days=sett.SF_MAX_TIMEDELTA_DAYS,
            max_snapshots=sett.SF_MAX_SNAPSHOTS_TO_SCAN,
        ),
    )


def _validate_prediction_population(rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[str, list[tuple[int, int, int]]] = {}
    for row in rows:
        model_name = str(row["model_name"])
        grouped.setdefault(model_name, []).append(
            (
                int(row["fold_number"]),
                int(row["round_index"]),
                int(row["match_id"]),
            )
        )
        probability = float(row["probability_under_2_5"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Prediction probabilities must be finite and in [0, 1].")

    if set(grouped) != set(PUBLICATION_BINARY_MODEL_NAMES):
        raise RuntimeError("Unexpected publication binary model population.")

    reference = grouped[PUBLICATION_BINARY_MODEL_NAMES[0]]
    if len(reference) != len(set(reference)):
        raise RuntimeError("Duplicate reference validation rows detected.")
    for model_name, keys in grouped.items():
        if keys != reference:
            raise RuntimeError(f"Model {model_name!r} does not use the exact common validation rows.")


def run_publication_binary_experiment(
    *,
    snapshot_path: Path,
    output_root: Path,
    config: PublicationBinaryConfig,
    overwrite: bool = False,
) -> Path:
    """Run Experiments II and III on common chronological binary folds.

    The proposed v1 model is carried forward chronologically. Probe models and
    flat-input baselines are freshly fitted inside each rolling training window
    and evaluated only on that fold's next chronological round.
    """

    config.validate()

    from tensorflow.keras.callbacks import EarlyStopping

    from football_outcomes.config import fs_settings as sett
    from football_outcomes.data.snapshots import load_snapshot
    from football_outcomes.data.sofifa_imputation import StrengthImputationConfig
    from football_outcomes.datasets.arrays import build_arrays_for_matches
    from football_outcomes.datasets.imputed_strength import build_fold_imputed_arrays
    from football_outcomes.datasets.mappings import build_categorical_maps
    from football_outcomes.datasets.rounds import distribute_matches_into_rounds
    from football_outcomes.experiments.publication_representations import (
        build_prelearning_flat_representation,
        extract_final_hidden_representation,
    )
    from football_outcomes.modeling.factory import build_model
    from football_outcomes.training.control import (
        learning_rate_for_round,
        set_optimizer_learning_rate,
    )
    from football_outcomes.training.runtime import (
        extract_main_predictions,
        make_train_targets,
        set_global_seed,
    )
    from football_outcomes.utils.fs_feature_utils import match_sort_key
    from football_outcomes.validation.selection import select_validation_matches

    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()

    snapshot_path = snapshot_path.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    bundle = load_snapshot(snapshot_path)
    if config.rebuild_match_features:
        selected, array_ready = _rebuild_selected_match_features(bundle)
    else:
        selected = select_validation_matches(bundle.matches, _selection_config())
        array_ready = sorted(
            (match for match in selected if getattr(match, "features_before_match", None) is not None),
            key=match_sort_key,
        )
    rounds = distribute_matches_into_rounds(array_ready)
    fold_indices = choose_publication_fold_indices(
        round_count=len(rounds),
        window_rounds=config.window_rounds,
        fold_count=config.fold_count,
        start_fold_offset=config.start_fold_offset,
    )

    competition_names = tuple(sett.COMPS_LEAGUE)
    cat_maps = build_categorical_maps(
        matches=array_ready,
        competition_names=competition_names,
    )
    proposed_train_config = _selected_v1_train_config(config)
    context = _strength_context(bundle) if config.enable_strength_imputation else None
    imputation_config = (
        StrengthImputationConfig(
            skill_count=len(sett.PLAYER_SKILLS),
            minimum_group_support=config.minimum_group_support,
            neutral_value=config.neutral_value,
        )
        if config.enable_strength_imputation
        else None
    )

    run_kind = "prl-binary-full" if config.fold_count is None else "prl-binary-partial"
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    run_directory = output_root / f"{run_kind}-{timestamp}"
    if run_directory.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {run_directory}")
        import shutil

        shutil.rmtree(run_directory)
    run_directory.mkdir(parents=True)

    configuration = {
        "run_kind": run_kind,
        "snapshot_path": str(snapshot_path),
        "scope": {
            "selected_matches": len(selected),
            "array_ready_matches": len(array_ready),
            "round_count": len(rounds),
            "fold_count": len(fold_indices),
            "fold_indices_zero_based": list(fold_indices),
            "validation_rounds_one_based": [index + 1 for index in fold_indices],
            "team_count": len(cat_maps.team_id_map),
            "competition_count": len(cat_maps.comp_id_map),
        },
        "publication_binary_config": asdict(config),
        "proposed_v1_train_config": asdict(proposed_train_config),
        "model_names": list(PUBLICATION_BINARY_MODEL_NAMES),
        "experiment_tracks": {
            "experiment_ii_representation_ablation": [
                "proposed-v1",
                "probe-logistic",
                "probe-random-forest",
                "probe-xgboost",
            ],
            "experiment_iii_prelearning_baselines": [
                "proposed-v1",
                "flat-logistic",
                "flat-random-forest",
                "flat-xgboost",
                "flat-mlp",
            ],
        },
        "notes": {
            "proposed_model_policy": "carry-forward across chronological folds",
            "classical_model_policy": "fresh fit per rolling training window",
            "proposed_early_stopping": True,
            "proposed_restore_best_weights": True,
            "proposed_shuffle": True,
            "proposed_fixed_epochs_per_fold": config.proposed_epochs_per_fold,
            "rebuild_match_features": config.rebuild_match_features,
            "enable_strength_imputation": config.enable_strength_imputation,
        },
    }
    _write_json(run_directory / "configuration.json", configuration)

    set_global_seed(config.seed)
    proposed_model = None
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    representation_layout_payload: dict[str, Any] | None = None

    for fold_number, round_index in enumerate(fold_indices, start=1):
        fold_clock = time.perf_counter()
        training_matches = [
            match
            for round_matches in rounds[round_index - config.window_rounds : round_index]
            for match in round_matches
        ]
        validation_matches = list(rounds[round_index])

        print(
            f"[prl-binary] fold {fold_number}/{len(fold_indices)} "
            f"round={round_index + 1} train={len(training_matches)} "
            f"validation={len(validation_matches)}",
            flush=True,
        )

        diagnostics = None
        if config.enable_strength_imputation:
            assert context is not None
            assert imputation_config is not None
            training_arrays, validation_arrays, diagnostics = build_fold_imputed_arrays(
                training_matches=training_matches,
                validation_matches=validation_matches,
                cat_maps=cat_maps,
                competition_names=competition_names,
                mode="binary_u25",
                max_goals_class=proposed_train_config.max_goals_class,
                context=context,
                imputation_config=imputation_config,
            )
        else:
            training_arrays = build_arrays_for_matches(
                matches=training_matches,
                cat_maps=cat_maps,
                competition_names=competition_names,
                mode="binary_u25",
                max_goals_class=proposed_train_config.max_goals_class,
            )
            validation_arrays = build_arrays_for_matches(
                matches=validation_matches,
                cat_maps=cat_maps,
                competition_names=competition_names,
                mode="binary_u25",
                max_goals_class=proposed_train_config.max_goals_class,
            )

        training_num_range = _validate_numerical_unit_interval(
            training_arrays[0],
            name=f"fold {fold_number} training",
        )
        validation_num_range = _validate_numerical_unit_interval(
            validation_arrays[0],
            name=f"fold {fold_number} validation",
        )
        if fold_number == 1:
            print(
                "[prl-binary] normalized X_num range "
                f"train=[{training_num_range['minimum']:.6g}, {training_num_range['maximum']:.6g}] "
                f"validation=[{validation_num_range['minimum']:.6g}, {validation_num_range['maximum']:.6g}]",
                flush=True,
            )

        if proposed_model is None:
            proposed_model = build_model(
                num_num=int(training_arrays[0].shape[1]),
                num_teams=len(cat_maps.team_id_map),
                num_comps=len(cat_maps.comp_id_map),
                cfg=proposed_train_config,
            )

        y_train_for_model = make_train_targets(
            training_matches,
            training_arrays[-1],
            proposed_train_config,
        )
        y_validation_for_model = make_train_targets(
            validation_matches,
            validation_arrays[-1],
            proposed_train_config,
        )
        y_train = _validated_binary_target(training_arrays[-1])
        y_validation = _validated_binary_target(validation_arrays[-1])

        round_offset = round_index - proposed_train_config.window_rounds
        total_train_rounds = max(1, len(rounds) - proposed_train_config.window_rounds)
        learning_rate = learning_rate_for_round(
            proposed_train_config,
            round_offset,
            total_train_rounds,
        )
        set_optimizer_learning_rate(proposed_model, learning_rate)

        early_stopping = EarlyStopping(
            patience=proposed_train_config.early_stopping_patience,
            min_delta=proposed_train_config.early_stopping_min_delta,
            restore_best_weights=True,
            monitor="val_loss",
            mode="min",
        )
        proposed_model.fit(
            training_arrays[:-1],
            y_train_for_model,
            validation_data=(validation_arrays[:-1], y_validation_for_model),
            epochs=config.proposed_epochs_per_fold,
            batch_size=config.proposed_batch_size,
            callbacks=[early_stopping],
            shuffle=True,
            verbose=2,
        )

        proposed_probability = np.asarray(
            extract_main_predictions(proposed_model.predict(validation_arrays[:-1], verbose=0)),
            dtype=np.float64,
        ).reshape(-1)

        training_latent = extract_final_hidden_representation(
            proposed_model,
            training_arrays,
            batch_size=config.latent_batch_size,
            verbose=0,
        )
        validation_latent = extract_final_hidden_representation(
            proposed_model,
            validation_arrays,
            batch_size=config.latent_batch_size,
            verbose=0,
        )

        flat_train, layout = build_prelearning_flat_representation(
            training_arrays,
            team_count=len(cat_maps.team_id_map),
            competition_count=len(cat_maps.comp_id_map),
        )
        flat_validation, validation_layout = build_prelearning_flat_representation(
            validation_arrays,
            team_count=len(cat_maps.team_id_map),
            competition_count=len(cat_maps.comp_id_map),
        )
        if layout != validation_layout:
            raise RuntimeError("Training and validation flat layouts differ.")
        if representation_layout_payload is None:
            representation_layout_payload = {
                "numerical_features": layout.numerical_features,
                "team_count": layout.team_count,
                "competition_count": layout.competition_count,
                "position_count": layout.position_count,
                "strength_shape": list(layout.strength_shape),
                "strength_features": layout.strength_features,
                "position_features_per_team": layout.position_features_per_team,
                "total_features": layout.total_features,
                "group_slices": {name: [group.start, group.stop] for name, group in layout.group_slices.items()},
                "final_latent_features": int(training_latent.shape[1]),
            }
        elif representation_layout_payload["total_features"] != layout.total_features:
            raise RuntimeError("Flat representation width changed between folds.")

        estimator_outputs: dict[str, ProbabilityResult] = {
            "probe-logistic": fit_logistic_probabilities(
                training_latent, y_train, validation_latent, config.estimators
            ),
            "probe-random-forest": fit_random_forest_probabilities(
                training_latent, y_train, validation_latent, config.estimators
            ),
            "probe-xgboost": fit_xgboost_probabilities(training_latent, y_train, validation_latent, config.estimators),
            "flat-logistic": fit_logistic_probabilities(
                flat_train,
                y_train,
                flat_validation,
                config.estimators,
                standardize=False,
            ),
            "flat-random-forest": fit_random_forest_probabilities(
                flat_train, y_train, flat_validation, config.estimators
            ),
            "flat-xgboost": fit_xgboost_probabilities(flat_train, y_train, flat_validation, config.estimators),
            "flat-mlp": fit_flat_mlp_probabilities(flat_train, y_train, flat_validation, config.estimators),
        }

        probabilities_by_model = {
            "proposed-v1": proposed_probability,
            **{model_name: result.probabilities for model_name, result in estimator_outputs.items()},
        }

        for model_name in PUBLICATION_BINARY_MODEL_NAMES:
            probability = np.asarray(probabilities_by_model[model_name], dtype=np.float64).reshape(-1)
            if probability.size != y_validation.size:
                raise RuntimeError(f"Prediction count mismatch for {model_name}.")
            metrics = binary_metrics(y_validation, probability)
            fit_status = (
                "carry-forward-neural" if model_name == "proposed-v1" else estimator_outputs[model_name].fit_status
            )
            feature_count = (
                int(training_latent.shape[1])
                if model_name.startswith("probe-")
                else layout.total_features if model_name.startswith("flat-") else None
            )
            track = (
                "proposed"
                if model_name == "proposed-v1"
                else "representation-ablation" if model_name.startswith("probe-") else "prelearning-baseline"
            )
            fold_metric_rows.append(
                {
                    "model_name": model_name,
                    "experiment_track": track,
                    "fold_number": fold_number,
                    "round_index": round_index + 1,
                    "fit_status": fit_status,
                    "feature_count": feature_count,
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
                        "model_name": model_name,
                        "experiment_track": track,
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

        fold_rows.append(
            {
                "fold_number": fold_number,
                "round_index": round_index + 1,
                "training_round_start": round_index - config.window_rounds + 1,
                "training_round_end": round_index,
                "training_matches": len(training_matches),
                "validation_matches": len(validation_matches),
                "learning_rate": learning_rate,
                "training_observed_strength_cells": (
                    diagnostics.training_observed_cells if diagnostics is not None else None
                ),
                "numerical_train_min": training_num_range["minimum"],
                "numerical_train_max": training_num_range["maximum"],
                "numerical_validation_min": validation_num_range["minimum"],
                "numerical_validation_max": validation_num_range["maximum"],
                "latent_features": int(training_latent.shape[1]),
                "flat_features": layout.total_features,
            }
        )
        runtime_rows.append(
            {
                "fold_number": fold_number,
                "round_index": round_index + 1,
                "seconds": float(time.perf_counter() - fold_clock),
            }
        )

        del training_latent, validation_latent, flat_train, flat_validation
        gc.collect()

    _validate_prediction_population(prediction_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for model_name in PUBLICATION_BINARY_MODEL_NAMES:
        model_rows = [row for row in prediction_rows if row["model_name"] == model_name]
        target = np.asarray([row["y_true"] for row in model_rows], dtype=np.int32)
        probability = np.asarray([row["probability_under_2_5"] for row in model_rows], dtype=np.float64)
        aggregate_rows.append(
            {
                "model_name": model_name,
                "experiment_track": model_rows[0]["experiment_track"],
                **binary_metrics(target, probability),
            }
        )

    _write_csv(run_directory / "folds.csv", fold_rows)
    _write_csv(run_directory / "predictions.csv", prediction_rows)
    _write_csv(run_directory / "fold_metrics.csv", fold_metric_rows)
    _write_csv(run_directory / "aggregate_metrics.csv", aggregate_rows)
    _write_json(
        run_directory / "representation_layout.json",
        representation_layout_payload or {},
    )
    _write_json(
        run_directory / "runtime.json",
        {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc),
            "total_seconds": float(time.perf_counter() - started_clock),
            "folds": runtime_rows,
        },
    )

    print(f"[prl-binary] PASS: {run_directory}", flush=True)
    return run_directory
