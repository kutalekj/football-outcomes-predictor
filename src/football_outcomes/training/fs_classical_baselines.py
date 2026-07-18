from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import football_outcomes.config.fs_settings as sett
from football_outcomes.data.fs_models import FSMatch
from football_outcomes.datasets.arrays import (
    build_flat_tabular_arrays_for_matches,
)
from football_outcomes.datasets.mappings import CatMaps
from football_outcomes.datasets.rounds import (
    distribute_matches_into_rounds,
    summarize_rounds,
)


@dataclass
class BaselineConfig:
    mode: str
    model_name: str
    window_rounds: int = 25
    max_goals_class: int = 10
    run_name: str | None = None
    random_state: int = 42


def _binary_summary(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = y_true.astype(np.float32)
    y_prob = y_prob.astype(np.float32)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
        "pooled_auc": auc,
        "pooled_brier": float(np.mean((y_prob - y_true) ** 2)),
    }


def _reg_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    return {
        "pooled_mae": float(mean_absolute_error(y_true, y_pred)),
        "pooled_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _multiclass_summary(y_true: np.ndarray, prob: np.ndarray, max_goals_class: int) -> dict:
    y_true = y_true.astype(np.int32)
    y_pred = np.argmax(prob, axis=1).astype(np.int32)
    classes = np.arange(max_goals_class + 1)

    expected_goals = (prob * classes[None, :]).sum(axis=1)

    try:
        ll = float(log_loss(y_true, prob, labels=classes))
    except ValueError:
        ll = float("nan")

    return {
        "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
        "pooled_log_loss": ll,
        "pooled_expected_goals_mae": float(mean_absolute_error(y_true.astype(np.float32), expected_goals)),
    }


def _fit_predict_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    cfg: BaselineConfig,
):
    if cfg.model_name == "majority":
        if cfg.mode == "binary_u25":
            pos_rate = float(np.mean(y_train))
            majority_class = 1.0 if pos_rate >= 0.5 else 0.0
            prob = np.full(len(X_val), pos_rate, dtype=np.float32)
            pred = np.full(len(X_val), majority_class, dtype=np.float32)
            return pred, prob

        if cfg.mode == "goals_dist":
            counts = np.bincount(y_train.astype(np.int32), minlength=cfg.max_goals_class + 1)
            cls = int(np.argmax(counts))
            prob = counts.astype(np.float32) / max(1.0, float(counts.sum()))
            prob_val = np.tile(prob[None, :], (len(X_val), 1))
            pred = np.full(len(X_val), cls, dtype=np.int32)
            return pred, prob_val

        if cfg.mode == "goals_reg":
            mean_val = float(np.mean(y_train))
            pred = np.full(len(X_val), mean_val, dtype=np.float32)
            return pred, pred

    if cfg.model_name == "logreg":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(X_train, y_train.astype(np.int32))
        prob = model.predict_proba(X_val)[:, 1].astype(np.float32)
        pred = (prob >= 0.5).astype(np.float32)
        return pred, prob

    if cfg.model_name == "multinomial_logreg":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, multi_class="multinomial", class_weight="balanced"),
        )
        model.fit(X_train, y_train.astype(np.int32))

        raw_prob = model.predict_proba(X_val)
        prob = np.zeros((len(X_val), cfg.max_goals_class + 1), dtype=np.float32)
        for j, cls in enumerate(model[-1].classes_):
            if 0 <= int(cls) <= cfg.max_goals_class:
                prob[:, int(cls)] = raw_prob[:, j]

        row_sums = prob.sum(axis=1, keepdims=True)
        prob = np.divide(prob, np.maximum(row_sums, 1e-8))
        pred = np.argmax(prob, axis=1).astype(np.int32)
        return pred, prob

    if cfg.model_name == "rf":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=cfg.random_state,
                n_jobs=-1,
            ),
        )
        model.fit(X_train, y_train.astype(np.int32))
        prob = model.predict_proba(X_val)[:, 1].astype(np.float32)
        pred = (prob >= 0.5).astype(np.float32)
        return pred, prob

    if cfg.model_name == "ridge":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        )
        model.fit(X_train, y_train.astype(np.float32))
        pred = model.predict(X_val).astype(np.float32)
        return pred, pred

    raise ValueError(f"Unknown baseline model: {cfg.model_name}")


def evaluate_baseline_rolling(
    matches_sorted: List[FSMatch],
    cat_maps: CatMaps,
    cfg: BaselineConfig,
    competition_names: Sequence[str] | None = None,
) -> dict:
    if competition_names is None:
        competition_names = sett.COMPS_LEAGUE

    rounds = distribute_matches_into_rounds(matches_sorted)
    round_info = summarize_rounds(rounds)

    run_name = cfg.run_name or f"baseline_{cfg.model_name}_{cfg.mode}"
    log_root = Path(sett.DATA_DIR) / "baseline_logs"
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    with (log_dir / "baseline_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    round_records = []
    oos_rows = []

    for i in range(cfg.window_rounds, len(rounds)):
        train_ms = [m for r in rounds[i - cfg.window_rounds : i] for m in r]
        val_ms = rounds[i]

        X_train, y_train = build_flat_tabular_arrays_for_matches(
            matches=train_ms,
            cat_maps=cat_maps,
            competition_names=competition_names,
            mode=cfg.mode,
            max_goals_class=(cfg.max_goals_class),
        )
        X_val, y_val = build_flat_tabular_arrays_for_matches(
            matches=val_ms,
            cat_maps=cat_maps,
            competition_names=competition_names,
            mode=cfg.mode,
            max_goals_class=(cfg.max_goals_class),
        )

        pred, score = _fit_predict_baseline(X_train, y_train, X_val, cfg)
        round_idx = int(i + 1)

        if cfg.mode == "binary_u25":
            metrics = _binary_summary(y_val, score)
            round_records.append(
                {
                    "round_idx": round_idx,
                    "train_size": len(train_ms),
                    "val_size": len(val_ms),
                    "positive_rate_val": float(np.mean(y_val)),
                    "val_accuracy": metrics["pooled_accuracy"],
                    "val_auc": metrics["pooled_auc"],
                    "val_brier": metrics["pooled_brier"],
                    "model_name": cfg.model_name,
                    "mode": cfg.mode,
                }
            )

            for m, yt, yp in zip(val_ms, y_val, score):
                oos_rows.append(
                    {
                        "round_idx": round_idx,
                        "match_id": m.id,
                        "season": m.season,
                        "competition": m.comp_name,
                        "y_true": float(yt),
                        "y_prob_under25": float(yp),
                        "model_name": cfg.model_name,
                        "mode": cfg.mode,
                    }
                )

        elif cfg.mode == "goals_reg":
            metrics = _reg_summary(y_val, pred)
            round_records.append(
                {
                    "round_idx": round_idx,
                    "train_size": len(train_ms),
                    "val_size": len(val_ms),
                    "val_mae": metrics["pooled_mae"],
                    "val_rmse": metrics["pooled_rmse"],
                    "model_name": cfg.model_name,
                    "mode": cfg.mode,
                }
            )

            for m, yt, yp in zip(val_ms, y_val, pred):
                oos_rows.append(
                    {
                        "round_idx": round_idx,
                        "match_id": m.id,
                        "season": m.season,
                        "competition": m.comp_name,
                        "y_true_goals": float(yt),
                        "y_pred_goals": float(yp),
                        "model_name": cfg.model_name,
                        "mode": cfg.mode,
                    }
                )

        elif cfg.mode == "goals_dist":
            metrics = _multiclass_summary(y_val, score, cfg.max_goals_class)
            round_records.append(
                {
                    "round_idx": round_idx,
                    "train_size": len(train_ms),
                    "val_size": len(val_ms),
                    "val_accuracy": metrics["pooled_accuracy"],
                    "val_log_loss": metrics["pooled_log_loss"],
                    "val_expected_goals_mae": metrics["pooled_expected_goals_mae"],
                    "model_name": cfg.model_name,
                    "mode": cfg.mode,
                }
            )

            expected = (score * np.arange(cfg.max_goals_class + 1)[None, :]).sum(axis=1)
            pred_cls = np.argmax(score, axis=1)

            for m, yt, yp, eg in zip(val_ms, y_val, pred_cls, expected):
                oos_rows.append(
                    {
                        "round_idx": round_idx,
                        "match_id": m.id,
                        "season": m.season,
                        "competition": m.comp_name,
                        "y_true_class": int(yt),
                        "y_pred_class": int(yp),
                        "y_pred_expected_goals": float(eg),
                        "model_name": cfg.model_name,
                        "mode": cfg.mode,
                    }
                )

    round_path = log_dir / "round_metrics.csv"
    if round_records:
        fieldnames = sorted({k for r in round_records for k in r.keys()})
        with round_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(round_records)

    pred_path = log_dir / "oos_predictions.csv"
    if oos_rows:
        fieldnames = sorted({k for r in oos_rows for k in r.keys()})
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(oos_rows)

    summary = {
        "run_name": run_name,
        "model_name": cfg.model_name,
        "mode": cfg.mode,
        "round_stats": round_info,
    }

    if cfg.mode == "binary_u25" and oos_rows:
        y_true = np.asarray([r["y_true"] for r in oos_rows], dtype=np.float32)
        y_prob = np.asarray([r["y_prob_under25"] for r in oos_rows], dtype=np.float32)
        summary.update(_binary_summary(y_true, y_prob))

    elif cfg.mode == "goals_reg" and oos_rows:
        y_true = np.asarray([r["y_true_goals"] for r in oos_rows], dtype=np.float32)
        y_pred = np.asarray([r["y_pred_goals"] for r in oos_rows], dtype=np.float32)
        summary.update(_reg_summary(y_true, y_pred))

    elif cfg.mode == "goals_dist" and oos_rows:
        y_true = np.asarray([r["y_true_class"] for r in oos_rows], dtype=np.int32)

        # Only expected-goals MAE and accuracy are available from saved compact rows.
        y_pred = np.asarray([r["y_pred_class"] for r in oos_rows], dtype=np.int32)
        y_exp = np.asarray([r["y_pred_expected_goals"] for r in oos_rows], dtype=np.float32)

        summary.update(
            {
                "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
                "pooled_expected_goals_mae": float(mean_absolute_error(y_true.astype(np.float32), y_exp)),
            }
        )

    with (log_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        **summary,
        "summary_path": str(log_dir / "summary.json"),
        "round_metrics_path": str(round_path),
        "oos_predictions_path": str(pred_path),
    }
