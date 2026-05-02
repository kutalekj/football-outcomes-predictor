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

# from football_outcomes.training.fs_classical_baselines import BaselineConfig, evaluate_baseline_rolling
from football_outcomes.training.fs_training_utils import build_categorical_maps  # extract_numerical_features
from football_outcomes.training.train_mlp_rolling import (
    TrainConfig,
    train_rolling,
)

# StrengthPretrainConfig,; build_model,; set_global_seed,; train_strength_pretrain_rolling,;
# transfer_pretrained_strength_branch_weights,
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu
from football_outcomes.utils.fs_player_skill_utils import match_fs_teams_to_sofifa_teams

# import matplotlib.pyplot as plt

# import tensorflow as tf

# import matplotlib.pyplot as plt


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
# HPO confirmation: rerun top configs with seed 42 and another seed
# ------------------------------------------------------------

HPO_STAGE = "hpo_confirmation_v1_full_scratch"

CONFIRM_CONFIGS = [
    {
        "config_label": "best_auc_original",
        "learning_rate": 8e-5,
        "lr_schedule": "exponential",
        "epochs_per_step": 5,
        "window_rounds": 25,
        "mlp_hidden_1": 128,
        "mlp_hidden_2": 64,
        "mlp_hidden_3": 32,
        "mlp_dropout_1": 0.50,
        "mlp_dropout_2": 0.40,
        "strength_emb_dim": 16,
    },
    {
        "config_label": "best_modified_accuracy",
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
    },
    {
        "config_label": "best_long_window",
        "learning_rate": 8e-5,
        "lr_schedule": "exponential",
        "epochs_per_step": 2,
        "window_rounds": 35,
        "mlp_hidden_1": 128,
        "mlp_hidden_2": 64,
        "mlp_hidden_3": 32,
        "mlp_dropout_1": 0.50,
        "mlp_dropout_2": 0.40,
        "strength_emb_dim": 16,
    },
]

SEEDS = [42, 123]

hpo_root = Path(sett.DATA_DIR) / "hpo" / HPO_STAGE
hpo_root.mkdir(parents=True, exist_ok=True)

results = []

for cfg_base in CONFIRM_CONFIGS:
    for seed in SEEDS:
        lr_tag = f"{cfg_base['learning_rate']:.1e}".replace(".", "p").replace("-", "m")
        run_name = (
            f"confirm_{cfg_base['config_label']}_"
            f"lr{lr_tag}_ep{cfg_base['epochs_per_step']}_"
            f"w{cfg_base['window_rounds']}_seed{seed}"
        )

        print("\n" + "=" * 80)
        print(f"[CONFIRM] {run_name}")
        print("=" * 80)

        cfg = TrainConfig(
            mode="binary_u25",
            model_version="v1",
            representation="full",
            use_strength_masks=True,
            use_position_embedding=True,
            use_team_strength=True,
            use_team_ids=True,
            use_comp_embedding=True,
            learning_rate=cfg_base["learning_rate"],
            lr_schedule=cfg_base["lr_schedule"],
            lr_decay_rate=0.997,
            min_learning_rate=2e-5,
            batch_size=64,
            window_rounds=cfg_base["window_rounds"],
            epochs_per_step=cfg_base["epochs_per_step"],
            early_stopping_patience=1,
            early_stopping_min_delta=0.0,
            team_emb_dim=8,
            comp_emb_dim=5,
            strength_emb_dim=cfg_base["strength_emb_dim"],
            position_emb_dim=3,
            mlp_hidden_1=cfg_base["mlp_hidden_1"],
            mlp_hidden_2=cfg_base["mlp_hidden_2"],
            mlp_hidden_3=cfg_base["mlp_hidden_3"],
            mlp_dropout_1=cfg_base["mlp_dropout_1"],
            mlp_dropout_2=cfg_base["mlp_dropout_2"],
            seed=seed,
            run_name=run_name,
            enable_branch_diagnostics=False,
            save_oos_predictions=True,
        )

        _ = train_rolling(league_matches_sorted, cat_maps, cfg)

        summary_path = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name / "summary.json"
        config_path = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name / "train_config.json"
        round_metrics_path = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name / "round_metrics.csv"

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        results.append(
            {
                "rank": None,
                "run_name": run_name,
                "stage": HPO_STAGE,
                "config_label": cfg_base["config_label"],
                "seed": seed,
                "learning_rate": cfg_base["learning_rate"],
                "lr_schedule": cfg_base["lr_schedule"],
                "epochs_per_step": cfg_base["epochs_per_step"],
                "window_rounds": cfg_base["window_rounds"],
                "mlp_hidden_1": cfg_base["mlp_hidden_1"],
                "mlp_hidden_2": cfg_base["mlp_hidden_2"],
                "mlp_hidden_3": cfg_base["mlp_hidden_3"],
                "mlp_dropout_1": cfg_base["mlp_dropout_1"],
                "mlp_dropout_2": cfg_base["mlp_dropout_2"],
                "strength_emb_dim": cfg_base["strength_emb_dim"],
                "pooled_accuracy": summary.get("pooled_accuracy"),
                "pooled_auc": summary.get("pooled_auc"),
                "pooled_brier": summary.get("pooled_brier"),
                "summary_path": str(summary_path),
                "config_path": str(config_path),
                "round_metrics_path": str(round_metrics_path),
            }
        )

results.sort(
    key=lambda r: (
        -(r["pooled_auc"] if r["pooled_auc"] is not None else -999),
        (r["pooled_brier"] if r["pooled_brier"] is not None else 999),
        -(r["pooled_accuracy"] if r["pooled_accuracy"] is not None else -999),
    )
)

for i, row in enumerate(results, start=1):
    row["rank"] = i

csv_path = hpo_root / "hpo_confirmation_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

json_path = hpo_root / "hpo_confirmation_results.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

txt_path = hpo_root / "hpo_confirmation_ranking.txt"
with txt_path.open("w", encoding="utf-8") as f:
    for r in results:
        f.write(
            f"{r['rank']}. {r['run_name']} | "
            f"label={r['config_label']} | "
            f"seed={r['seed']} | "
            f"AUC={r['pooled_auc']:.6f} | "
            f"ACC={r['pooled_accuracy']:.6f} | "
            f"BRIER={r['pooled_brier']:.6f}\n"
        )

print(f"[CONFIRM] saved CSV to {csv_path}")
print(f"[CONFIRM] saved JSON to {json_path}")
print(f"[CONFIRM] saved ranking to {txt_path}")

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
