from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

import football_outcomes.config.fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import load_avg_team_strength, load_sofifa_players, save_snapshot, try_load_snapshot
from football_outcomes.data.fs_models import FSDataBundle
from football_outcomes.data.fs_retrieve import fill_globals_with_cache, retrieve_new_data
from football_outcomes.training.fs_classical_baselines import BaselineConfig, evaluate_baseline_rolling
from football_outcomes.training.fs_training_utils import build_categorical_maps  # extract_numerical_features
from football_outcomes.training.train_mlp_rolling import (
    TrainConfig,
    train_rolling,
)
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu
from football_outcomes.utils.fs_player_skill_utils import match_fs_teams_to_sofifa_teams

# import matplotlib.pyplot as plt
# import tensorflow as tf


matplotlib.use("Agg")


def log_feature_error(msg: str) -> None:
    os.makedirs(sett.LOG_DIR, exist_ok=True)
    path = os.path.join("logs", "feature_errors.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip("\n") + "\n")


ut = utils
global_instance = Global.get_instance()

load_avg_team_strength()  # TODO: Remove this currently unused averaging?

cache = try_load_snapshot()
if sett.ALL_LOAD and cache is not None:
    fill_globals_with_cache(cache, update_leagues_list=False)

if sett.ALL_GET_NEW:
    bundle = retrieve_new_data()

# Only rebuild from CSV when explicitly requested (one-time migration/new data)
load_sofifa_players(rebuild=getattr(sett, "REBUILD_SOFIFA_FROM_CSV", False), debug_shifts=False)
print("[sofifa] snapshots:", len(global_instance.sofifa_snapshots))

ut.link_matches_to_comp_seasons()
if sett.VALIDATE_ROUND_IDS:
    ut.validate_league_valid_round_ids()

ut.ensure_comp_season_dates(force=sett.ALL_GET_NEW)
ut.initialize_league_tables(precompute_positions=True, force_rebuild=sett.ALL_GET_NEW)

# Match FS league teams to SOFIFA teams (one-time per run)
match_fs_teams_to_sofifa_teams(force=False)

# Relevant matches only (league comps)
all_matches_sorted = sorted(global_instance.all_matches, key=fu.match_sort_key)
league_matches_sorted = utils.filter_clean_league_matches(all_matches_sorted)
league_matches_sorted = utils.filter_valid_round_matches(league_matches_sorted)
league_matches_sorted = [
    m
    for m in league_matches_sorted
    if getattr(m, "season", None) is not None and sett.FIRST_SEASON <= m.season < sett.LAST_SEASON
]

team_index_all = fu.build_team_match_index(all_matches_sorted)
team_index_league = fu.build_team_match_index(league_matches_sorted)

last_progress_month = None  # type: tuple[int,int] | None  # (year, month)
processed = 0
total = len(league_matches_sorted)
skipped_matches = 0

for match in league_matches_sorted:
    processed += 1
    dt = match.datetime  # datetime at 00:00
    year = dt.year
    month = dt.month

    curr_month = (year, month)
    if curr_month != last_progress_month:
        last_progress_month = curr_month
        print(f"[features] {year:04d}-{month:02d}  (processed {processed}/{total})")

    try:
        match.features_before_match = match.calculate_match_features(
            team_index_league=team_index_league,
            team_index_all=team_index_all,
        )

        # if match.home_team.name == "KRC Genk" or match.away_team.name == "KRC Genk":
        #     fu.debug_print_match_and_features(match)  # DEBUG

    except ValueError as e:
        skipped_matches += 1
        log_feature_error(
            f"[SKIP] match_id={match.id} {match.comp_name} {match.season} "
            f"{match.datetime} h={match.hour_utc} "
            f"{match.home_team.name} vs {match.away_team.name} "
            f"error={repr(e)}"
        )
        continue

print(f"[features] Done. Skipped matches: {skipped_matches}")

league_matches_sorted = [
    m for m in league_matches_sorted if hasattr(m, "features_before_match") and m.features_before_match is not None
]
print(f"[features] usable matches for training: {len(league_matches_sorted)}")

cat_maps = build_categorical_maps(league_matches_sorted)

# ------------------------------------------------------------
# Simple baselines first
# ------------------------------------------------------------
"""
evaluate_baseline_rolling(
    league_matches_sorted,
    cat_maps,
    BaselineConfig(mode="binary_u25", model_name="logreg", window_rounds=25),
)

evaluate_baseline_rolling(
    league_matches_sorted,
    cat_maps,
    BaselineConfig(mode="binary_u25", model_name="rf", window_rounds=25),
)

evaluate_baseline_rolling(
    league_matches_sorted,
    cat_maps,
    BaselineConfig(mode="goals_reg", model_name="ridge", window_rounds=25),
)
"""

# ------------------------------------------------------------
# Baseline comparison across objectives
# ------------------------------------------------------------

EXPERIMENT_STAGE = "baseline_comparison_selected_model"

comparison_root = Path(sett.DATA_DIR) / "comparison" / EXPERIMENT_STAGE
comparison_root.mkdir(parents=True, exist_ok=True)

SELECTED_CFG = {
    "learning_rate": 8e-5,
    "lr_schedule": "exponential",
    "epochs_per_step": 2,
    "window_rounds": 25,
    "mlp_hidden_1": 128,
    "mlp_hidden_2": 64,
    "mlp_hidden_3": 32,
    "mlp_dropout_1": 0.30,
    "mlp_dropout_2": 0.20,
    "strength_emb_dim": 24,
    "seed": 123,
}

results = []

BASELINE_RUNS = [
    BaselineConfig(mode="binary_u25", model_name="majority", window_rounds=25, run_name="baseline_binary_majority"),
    BaselineConfig(mode="binary_u25", model_name="logreg", window_rounds=25, run_name="baseline_binary_logreg"),
    BaselineConfig(mode="binary_u25", model_name="rf", window_rounds=25, run_name="baseline_binary_rf"),
    BaselineConfig(mode="goals_reg", model_name="majority", window_rounds=25, run_name="baseline_reg_mean_goals"),
    BaselineConfig(mode="goals_reg", model_name="ridge", window_rounds=25, run_name="baseline_reg_ridge"),
    BaselineConfig(mode="goals_dist", model_name="majority", window_rounds=25, run_name="baseline_multiclass_majority"),
    BaselineConfig(
        mode="goals_dist", model_name="multinomial_logreg", window_rounds=25, run_name="baseline_multiclass_logreg"
    ),
]

for cfg in BASELINE_RUNS:
    print("\n" + "=" * 80)
    print(f"[BASELINE] {cfg.run_name}")
    print("=" * 80)

    summary = evaluate_baseline_rolling(league_matches_sorted, cat_maps, cfg)
    results.append(
        {
            "group": "baseline",
            "run_name": summary["run_name"],
            "model_name": summary["model_name"],
            "mode": summary["mode"],
            "pooled_accuracy": summary.get("pooled_accuracy"),
            "pooled_auc": summary.get("pooled_auc"),
            "pooled_brier": summary.get("pooled_brier"),
            "pooled_mae": summary.get("pooled_mae"),
            "pooled_rmse": summary.get("pooled_rmse"),
            "pooled_expected_goals_mae": summary.get("pooled_expected_goals_mae"),
            "pooled_log_loss": summary.get("pooled_log_loss"),
            "summary_path": summary.get("summary_path"),
            "round_metrics_path": summary.get("round_metrics_path"),
            "oos_predictions_path": summary.get("oos_predictions_path"),
        }
    )


MLP_MODES = [
    ("binary_u25", "selected_mlp_binary_u25"),
    ("goals_reg", "selected_mlp_goals_reg"),
    ("goals_dist", "selected_mlp_goals_dist"),
]

for mode, run_name in MLP_MODES:
    print("\n" + "=" * 80)
    print(f"[SELECTED MLP] {run_name}")
    print("=" * 80)

    cfg = TrainConfig(
        mode=mode,
        model_version="v1",
        representation="full",
        use_strength_masks=True,
        use_position_embedding=True,
        use_team_strength=True,
        use_team_ids=True,
        use_comp_embedding=True,
        learning_rate=SELECTED_CFG["learning_rate"],
        lr_schedule=SELECTED_CFG["lr_schedule"],
        lr_decay_rate=0.997,
        min_learning_rate=2e-5,
        batch_size=64,
        window_rounds=SELECTED_CFG["window_rounds"],
        epochs_per_step=SELECTED_CFG["epochs_per_step"],
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        team_emb_dim=8,
        comp_emb_dim=5,
        strength_emb_dim=SELECTED_CFG["strength_emb_dim"],
        position_emb_dim=3,
        mlp_hidden_1=SELECTED_CFG["mlp_hidden_1"],
        mlp_hidden_2=SELECTED_CFG["mlp_hidden_2"],
        mlp_hidden_3=SELECTED_CFG["mlp_hidden_3"],
        mlp_dropout_1=SELECTED_CFG["mlp_dropout_1"],
        mlp_dropout_2=SELECTED_CFG["mlp_dropout_2"],
        seed=SELECTED_CFG["seed"],
        run_name=run_name,
        enable_branch_diagnostics=False,
        save_oos_predictions=True,
    )

    _ = train_rolling(league_matches_sorted, cat_maps, cfg)

    summary_path = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name / "summary.json"
    round_metrics_path = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name / "round_metrics.csv"
    oos_path = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name / "oos_predictions.csv"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    results.append(
        {
            "group": "selected_mlp",
            "run_name": run_name,
            "model_name": "selected_mlp",
            "mode": mode,
            "pooled_accuracy": summary.get("pooled_accuracy"),
            "pooled_auc": summary.get("pooled_auc"),
            "pooled_brier": summary.get("pooled_brier"),
            "pooled_mae": summary.get("pooled_mae"),
            "pooled_rmse": summary.get("pooled_rmse"),
            "pooled_expected_goals_mae": summary.get("pooled_expected_goals_mae"),
            "pooled_log_loss": summary.get("pooled_log_loss"),
            "summary_path": str(summary_path),
            "round_metrics_path": str(round_metrics_path),
            "oos_predictions_path": str(oos_path),
        }
    )


csv_path = comparison_root / "baseline_comparison_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = sorted({k for r in results for k in r.keys()})
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

json_path = comparison_root / "baseline_comparison_results.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

txt_path = comparison_root / "baseline_comparison_summary.txt"
with txt_path.open("w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, indent=2) + "\n\n")

print(f"[COMPARISON] saved CSV to {csv_path}")
print(f"[COMPARISON] saved JSON to {json_path}")
print(f"[COMPARISON] saved text summary to {txt_path}")

if sett.ALL_STORE:
    save_snapshot(
        FSDataBundle(
            global_instance.all_comp_seasons,
            global_instance.all_teams,
            global_instance.all_players,
            global_instance.all_matches,
            global_instance.leagues_list,
            sofifa_snapshots=getattr(global_instance, "sofifa_snapshots", []),
            sofifa_player_occurrences=getattr(global_instance, "sofifa_player_occurrences", {}),
            sofifa_players_by_dob=getattr(global_instance, "sofifa_players_by_dob", {}),
            fs_to_sofifa_cache=getattr(global_instance, "fs_to_sofifa_cache", {}),
            sofifa_team_meta=getattr(global_instance, "sofifa_team_meta", {}),
            sofifa_players_by_team=getattr(global_instance, "sofifa_players_by_team", {}),
            sofifa_teams_by_league=getattr(global_instance, "sofifa_teams_by_league", {}),
            fs_team_to_sofifa_team=getattr(global_instance, "fs_team_to_sofifa_team", {}),
        )
    )
