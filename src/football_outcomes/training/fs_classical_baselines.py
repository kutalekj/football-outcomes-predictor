from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import FSMatch
from football_outcomes.training.fs_training_utils import (
    CatMaps,
    build_flat_tabular_arrays_for_matches,
    distribute_matches_into_rounds,
)


@dataclass
class BaselineConfig:
    mode: str = "binary_u25"  # binary_u25 | goals_reg
    model_name: str = "logreg"  # logreg | ridge | rf
    window_rounds: int = 25
    max_goals_class: int = 10


def _make_model(cfg: BaselineConfig):
    if cfg.mode == "binary_u25" and cfg.model_name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

    if cfg.mode == "binary_u25" and cfg.model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    if cfg.mode == "goals_reg" and cfg.model_name == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    raise ValueError(f"Unsupported baseline setup: mode={cfg.mode}, model_name={cfg.model_name}")


def evaluate_baseline_rolling(
    matches_sorted: List[FSMatch],
    cat_maps: CatMaps,
    cfg: BaselineConfig,
) -> Path:
    rounds = distribute_matches_into_rounds(matches_sorted)

    log_root = Path(sett.DATA_DIR) / "baseline_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    out_path = log_root / f"{cfg.model_name}_{cfg.mode}_round_metrics.csv"
    pred_path = log_root / f"{cfg.model_name}_{cfg.mode}_oos_predictions.csv"

    round_rows = []
    pred_rows = []

    for i in range(cfg.window_rounds, len(rounds) - 1):
        train_ms = [m for r in rounds[i - cfg.window_rounds : i] for m in r]
        val_ms = rounds[i]

        X_train, y_train = build_flat_tabular_arrays_for_matches(train_ms, cat_maps, cfg.mode, cfg.max_goals_class)
        X_val, y_val = build_flat_tabular_arrays_for_matches(val_ms, cat_maps, cfg.mode, cfg.max_goals_class)

        model = _make_model(cfg)
        model.fit(X_train, y_train)

        if cfg.mode == "binary_u25":
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_val)[:, 1]
            else:
                p = model.decision_function(X_val)
                p = 1.0 / (1.0 + np.exp(-p))

            y_hat = (p >= 0.5).astype(np.float32)

            round_rows.append(
                {
                    "round_idx": i + 1,
                    "train_size": len(train_ms),
                    "val_size": len(val_ms),
                    "accuracy": float(accuracy_score(y_val, y_hat)),
                    "f1": float(f1_score(y_val, y_hat)),
                    "auc": float(roc_auc_score(y_val, p)) if len(np.unique(y_val)) > 1 else np.nan,
                    "positive_rate_val": float(np.mean(y_val)),
                }
            )

            for m, yt, yp in zip(val_ms, y_val, p):
                pred_rows.append(
                    {
                        "round_idx": i + 1,
                        "match_id": m.id,
                        "season": m.season,
                        "competition": m.comp_name,
                        "y_true": float(yt),
                        "y_prob_under25": float(yp),
                    }
                )

        elif cfg.mode == "goals_reg":
            pred = model.predict(X_val)

            round_rows.append(
                {
                    "round_idx": i + 1,
                    "train_size": len(train_ms),
                    "val_size": len(val_ms),
                    "mae": float(mean_absolute_error(y_val, pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
                }
            )

            for m, yt, yp in zip(val_ms, y_val, pred):
                pred_rows.append(
                    {
                        "round_idx": i + 1,
                        "match_id": m.id,
                        "season": m.season,
                        "competition": m.comp_name,
                        "y_true": float(yt),
                        "y_pred_goals": float(yp),
                    }
                )

    if round_rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(round_rows[0].keys()))
            writer.writeheader()
            writer.writerows(round_rows)

    if pred_rows:
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(pred_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pred_rows)

    print(f"[baseline] round metrics -> {out_path}")
    print(f"[baseline] oos predictions -> {pred_path}")
    return out_path
