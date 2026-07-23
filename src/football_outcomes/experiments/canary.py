from __future__ import annotations

import math
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.snapshots import load_snapshot
from football_outcomes.data.sofifa_imputation import StrengthImputationConfig
from football_outcomes.data.sofifa_strength import PastOnlyStrengthConfig
from football_outcomes.datasets.imputed_strength import (
    StrengthImputationContext,
    build_fold_imputed_arrays,
)
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
    write_canonical_json,
    write_experiment_manifest,
)
from football_outcomes.modeling.factory import build_model
from football_outcomes.training.configs import TrainConfig
from football_outcomes.training.runtime import (
    extract_main_predictions,
    make_train_targets,
    set_global_seed,
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


@dataclass(frozen=True)
class CanaryConfig:
    window_rounds: int = 25
    fold_count: int = 2
    start_fold_offset: int = 0
    epochs_per_fold: int = 1
    batch_size: int = 64
    learning_rate: float = 0.0001
    seed: int = 123
    minimum_group_support: int = 20
    neutral_value: float = 50.0
    model_version: str = "v2"

    def __post_init__(self) -> None:
        for name in (
            "window_rounds",
            "fold_count",
            "epochs_per_fold",
            "batch_size",
            "minimum_group_support",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if type(self.start_fold_offset) is not int or self.start_fold_offset < 0:
            raise ValueError("start_fold_offset must be a non-negative integer.")

        if type(self.seed) is not int:
            raise ValueError("seed must be an integer.")

        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive finite number.")

        if not math.isfinite(self.neutral_value) or self.neutral_value < 0.0 or self.neutral_value > 100.0:
            raise ValueError("neutral_value must be finite and in [0, 100].")

        if self.model_version not in {"v1", "v2"}:
            raise ValueError("model_version must be 'v1' or 'v2'.")


def choose_canary_fold_indices(
    *,
    round_count: int,
    window_rounds: int,
    fold_count: int,
    start_fold_offset: int,
) -> tuple[int, ...]:
    for name, value in (
        ("round_count", round_count),
        ("window_rounds", window_rounds),
        ("fold_count", fold_count),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if type(start_fold_offset) is not int or start_fold_offset < 0:
        raise ValueError("start_fold_offset must be a non-negative integer.")

    first = window_rounds + start_fold_offset
    stop = first + fold_count

    if first >= round_count or stop > round_count:
        raise ValueError("Requested canary folds exceed the available chronological rounds.")

    return tuple(range(first, stop))


def binary_metrics(
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


def validate_fold_chronology(
    training_matches: Sequence[Any],
    validation_matches: Sequence[Any],
) -> tuple[str, str]:
    if not training_matches or not validation_matches:
        raise ValueError("Training and validation matches must not be empty.")

    train_datetimes = [getattr(match, "datetime", None) for match in training_matches]
    validation_datetimes = [getattr(match, "datetime", None) for match in validation_matches]

    if any(value is None for value in train_datetimes + validation_datetimes):
        raise ValueError("Every canary match must have a datetime.")

    training_max = max(train_datetimes)
    validation_min = min(validation_datetimes)

    if training_max > validation_min:
        raise RuntimeError("Canary fold chronology is invalid.")

    return training_max.isoformat(), validation_min.isoformat()


def validate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Prediction rows must not be empty.")

    match_ids: list[int] = []

    for row in rows:
        match_id = row.get("match_id")
        target = row.get("y_true")
        probability = row.get("probability_under_2_5")

        if type(match_id) is not int:
            raise ValueError("Prediction match IDs must be integers.")

        if target not in (0, 1, 0.0, 1.0):
            raise ValueError("Prediction targets must contain only 0 and 1.")

        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("Prediction probabilities must be numeric.")

        numeric = float(probability)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError("Prediction probabilities must lie in [0, 1].")

        match_ids.append(match_id)

    if len(match_ids) != len(set(match_ids)):
        raise RuntimeError("A validation match appears more than once.")


def _categorical_map_payload(cat_maps: Any) -> dict[str, Any]:
    return {
        "team_id_map": [[int(source), int(target)] for source, target in sorted(cat_maps.team_id_map.items())],
        "competition_id_map": [[int(source), int(target)] for source, target in sorted(cat_maps.comp_id_map.items())],
    }


def _training_config(config: CanaryConfig) -> TrainConfig:
    return TrainConfig(
        mode="binary_u25",
        model_version=config.model_version,
        window_rounds=config.window_rounds,
        epochs_per_step=config.epochs_per_fold,
        learning_rate=config.learning_rate,
        lr_schedule="constant",
        batch_size=config.batch_size,
        seed=config.seed,
        use_team_aux_head=False,
        aux_task=None,
        run_name=None,
        save_oos_predictions=False,
        enable_branch_diagnostics=False,
        enable_strength_imputation=True,
        strength_imputation_minimum_support=config.minimum_group_support,
        strength_imputation_neutral_value=config.neutral_value,
    )


def _selection_config() -> SelectionValidationConfig:
    return SelectionValidationConfig(
        competitions=tuple(sett.COMPS_LEAGUE),
        first_season=sett.FIRST_SEASON,
        last_season_exclusive=sett.LAST_SEASON,
        excluded_competition_seasons=frozenset(sett.EXCLUDED_COMP_SEASONS),
        valid_round_ids_by_season=sett.LEAGUE_VALID_ROUND_IDS_BY_SEASON,
    )


def _strength_context(bundle: Any) -> StrengthImputationContext:
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


def _final_history_value(history: Mapping[str, Sequence[float]], *names: str) -> float | None:
    for name in names:
        values = history.get(name)
        if values:
            value = float(values[-1])
            if math.isfinite(value):
                return value
    return None


def _identifier_hash(values: Sequence[int]) -> str:
    return canonical_payload_sha256([int(value) for value in values])


def _render_summary(
    *,
    run_id: str,
    git_dirty: bool,
    configuration: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    fold_rows: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    title: str = "Step 8 chronological modeling canary",
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Result: **{'PASS' if aggregate['overall_ok'] else 'FAIL'}**",
        f"- Git dirty at start: `{str(git_dirty).lower()}`",
        f"- Model: `{configuration['model']['model_version']}`",
        "- Carry-forward model state: `true`",
        f"- Audited folds: `{len(fold_rows)}`",
        f"- Predictions: `{aggregate['metrics']['prediction_count']}`",
        f"- Runtime seconds: `{runtime['total_seconds']:.3f}`",
        "",
        "## Pooled metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for name, value in aggregate["metrics"].items():
        rendered = "n/a" if value is None else str(value)
        lines.append(f"| {name} | {rendered} |")

    lines.extend(
        [
            "",
            "## Fold scope",
            "",
            "| Round | Training matches | Validation matches |",
            "|---:|---:|---:|",
        ]
    )

    for row in fold_rows:
        lines.append(f"| {row['round_index']} | {row['training_matches']} | " f"{row['validation_matches']} |")

    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            "The canary passes only when chronological folds are valid,",
            "all validation predictions are unique and finite, probabilities",
            "remain in `[0, 1]`, and the complete manifest-backed artifact set",
            "is written successfully.",
            "",
        ]
    )

    return "\n".join(lines)


def run_modeling_canary(
    *,
    repository_root: Path,
    snapshot_path: Path,
    output_root: Path,
    config: CanaryConfig,
    command: Sequence[str],
    overwrite: bool = False,
    run_kind: str = "canary",
    experiment_tier: str = "chronological-canary",
    model_name: str | None = None,
    summary_title: str = "Step 8 chronological modeling canary",
) -> Path:
    if not run_kind:
        raise ValueError("run_kind must not be empty.")

    if not experiment_tier:
        raise ValueError("experiment_tier must not be empty.")

    if not summary_title:
        raise ValueError("summary_title must not be empty.")

    resolved_model_name = model_name or f"{config.model_version}-canary"

    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()

    repository_root = repository_root.resolve()
    output_root = output_root.resolve()

    snapshot_identity = collect_snapshot_identity(
        snapshot_path,
        expected_sha256=EXPECTED_SNAPSHOT_SHA256,
    )
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
        raise RuntimeError(
            f"Selected match count changed: expected {EXPECTED_SELECTED_MATCHES}, " f"found {len(selected)}."
        )

    if len(array_ready) != EXPECTED_ARRAY_READY_MATCHES:
        raise RuntimeError(
            "Array-ready match count changed: expected " f"{EXPECTED_ARRAY_READY_MATCHES}, found {len(array_ready)}."
        )

    if len(rounds) != EXPECTED_ROUND_COUNT:
        raise RuntimeError(f"Round count changed: expected {EXPECTED_ROUND_COUNT}, " f"found {len(rounds)}.")

    fold_indices = choose_canary_fold_indices(
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
    categorical_payload = _categorical_map_payload(cat_maps)
    training_config = _training_config(config)

    configuration = {
        "experiment_tier": experiment_tier,
        "scope": {
            "selected_matches": len(selected),
            "array_ready_matches": len(array_ready),
            "round_count": len(rounds),
            "competition_count": len(cat_maps.comp_id_map),
            "team_count": len(cat_maps.team_id_map),
        },
        "fold_selection": {
            "window_rounds": config.window_rounds,
            "fold_count": config.fold_count,
            "start_fold_offset": config.start_fold_offset,
            "zero_based_fold_indices": list(fold_indices),
            "one_based_round_indices": [index + 1 for index in fold_indices],
            "model_state_policy": "carry-forward",
        },
        "imputation": {
            "enabled": True,
            "minimum_group_support": config.minimum_group_support,
            "neutral_value": config.neutral_value,
            "past_only_max_age_days": sett.SF_MAX_TIMEDELTA_DAYS,
            "past_only_max_snapshots": sett.SF_MAX_SNAPSHOTS_TO_SCAN,
        },
        "model": asdict(training_config),
        "categorical_maps": {
            "sha256": canonical_payload_sha256(categorical_payload),
            "team_count": len(cat_maps.team_id_map),
            "competition_count": len(cat_maps.comp_id_map),
        },
    }

    run_id = derive_run_id(
        run_kind=run_kind,
        git_commit=git_identity.commit,
        snapshot_sha256=snapshot_identity.sha256,
        seed=config.seed,
        configuration=configuration,
    )
    run_directory = output_root / run_id

    if run_directory.exists():
        if not overwrite:
            raise FileExistsError(
                f"Modeling output already exists: {run_directory}. " "Use overwrite=True to replace it."
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

    context = _strength_context(bundle)
    imputation_config = StrengthImputationConfig(
        skill_count=len(sett.PLAYER_SKILLS),
        minimum_group_support=config.minimum_group_support,
        neutral_value=config.neutral_value,
    )

    set_global_seed(config.seed)
    model = None
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    fold_runtime_rows: list[dict[str, Any]] = []

    for fold_number, round_index in enumerate(fold_indices, start=1):
        fold_clock = time.perf_counter()
        training_matches = [
            match
            for round_matches in rounds[round_index - config.window_rounds : round_index]
            for match in round_matches
        ]
        validation_matches = list(rounds[round_index])

        print(
            f"[{run_kind}] fold {fold_number}/{len(fold_indices)} "
            f"round={round_index + 1} train={len(training_matches)} "
            f"validation={len(validation_matches)}",
            flush=True,
        )

        training_max, validation_min = validate_fold_chronology(
            training_matches,
            validation_matches,
        )

        training_arrays, validation_arrays, diagnostics = build_fold_imputed_arrays(
            training_matches=training_matches,
            validation_matches=validation_matches,
            cat_maps=cat_maps,
            competition_names=competition_names,
            mode="binary_u25",
            max_goals_class=training_config.max_goals_class,
            context=context,
            imputation_config=imputation_config,
        )

        if model is None:
            model = build_model(
                num_num=int(training_arrays[0].shape[1]),
                num_teams=len(cat_maps.team_id_map),
                num_comps=len(cat_maps.comp_id_map),
                cfg=training_config,
            )

        y_train = make_train_targets(
            training_matches,
            training_arrays[-1],
            training_config,
        )
        y_validation = make_train_targets(
            validation_matches,
            validation_arrays[-1],
            training_config,
        )

        history = model.fit(
            training_arrays[:-1],
            y_train,
            validation_data=(validation_arrays[:-1], y_validation),
            epochs=config.epochs_per_fold,
            batch_size=config.batch_size,
            shuffle=False,
            verbose=2,
        )

        raw_prediction = model.predict(validation_arrays[:-1], verbose=0)
        probability = extract_main_predictions(raw_prediction).reshape(-1).astype(np.float64)
        target = validation_arrays[-1].reshape(-1).astype(np.float64)
        metrics = binary_metrics(target, probability)

        training_ids = [int(match.id) for match in training_matches]
        validation_ids = [int(match.id) for match in validation_matches]

        fold_rows.append(
            {
                "run_id": run_id,
                "fold_number": fold_number,
                "round_index": round_index + 1,
                "training_round_start": round_index - config.window_rounds + 1,
                "training_round_end": round_index,
                "training_matches": len(training_matches),
                "validation_matches": len(validation_matches),
                "training_max_datetime": training_max,
                "validation_min_datetime": validation_min,
                "training_match_ids_sha256": _identifier_hash(training_ids),
                "validation_match_ids_sha256": _identifier_hash(validation_ids),
                "training_observed_strength_cells": diagnostics.training_observed_cells,
            }
        )

        fold_metric_rows.append(
            {
                "run_id": run_id,
                "model_name": resolved_model_name,
                "fold_number": fold_number,
                "round_index": round_index + 1,
                **metrics,
                "training_loss": _final_history_value(history.history, "loss"),
                "validation_loss": _final_history_value(
                    history.history,
                    "val_loss",
                    "val_output_main_loss",
                ),
            }
        )

        for match, y_true, y_probability in zip(
            validation_matches,
            target,
            probability,
        ):
            prediction_rows.append(
                {
                    "run_id": run_id,
                    "model_name": resolved_model_name,
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

        fold_runtime_rows.append(
            {
                "fold_number": fold_number,
                "round_index": round_index + 1,
                "seconds": float(time.perf_counter() - fold_clock),
            }
        )

    validate_prediction_rows(prediction_rows)

    pooled_target = np.asarray(
        [row["y_true"] for row in prediction_rows],
        dtype=np.float64,
    )
    pooled_probability = np.asarray(
        [row["probability_under_2_5"] for row in prediction_rows],
        dtype=np.float64,
    )
    aggregate_metrics = binary_metrics(pooled_target, pooled_probability)
    aggregate = {
        "overall_ok": True,
        "run_id": run_id,
        "model_name": resolved_model_name,
        "fold_count": len(fold_rows),
        "metrics": aggregate_metrics,
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
        "folds": fold_runtime_rows,
    }
    write_canonical_json(runtime_path, runtime)

    summary_path.write_text(
        _render_summary(
            run_id=run_id,
            git_dirty=git_identity.is_dirty,
            configuration=configuration,
            aggregate=aggregate,
            fold_rows=fold_rows,
            runtime=runtime,
            title=summary_title,
        ),
        encoding="utf-8",
        newline="\n",
    )

    artifact_paths = (
        configuration_path,
        folds_path,
        predictions_path,
        fold_metrics_path,
        aggregate_path,
        runtime_path,
        summary_path,
    )
    artifacts = collect_artifact_identities(artifact_paths, root=run_directory)
    manifest = build_experiment_manifest(
        run_kind=run_kind,
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
