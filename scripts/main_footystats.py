from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

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
# HPO Stage 1: v1 full scratch training dynamics
# ------------------------------------------------------------

HPO_STAGE = "stage1_v1_full_scratch"

HPO_RUNS = []

learning_rates = [5e-5, 8e-5, 1e-4, 1.5e-4]
epochs_per_step_values = [2, 5]
lr_schedules = ["constant", "exponential"]
window_rounds_values = [25, 35]

for lr in learning_rates:
    for eps in epochs_per_step_values:
        for sched in lr_schedules:
            for window_rounds in window_rounds_values:
                lr_tag = f"{lr:.1e}".replace(".", "p").replace("-", "m")
                run_name = f"hpo1_v1_full_lr{lr_tag}_ep{eps}_{sched}_w{window_rounds}"

                HPO_RUNS.append(
                    {
                        "run_name": run_name,
                        "learning_rate": lr,
                        "epochs_per_step": eps,
                        "lr_schedule": sched,
                        "window_rounds": window_rounds,
                        "batch_size": 64,
                        "early_stopping_patience": 1,
                        "early_stopping_min_delta": 0.0,
                        "mlp_hidden_1": 128,
                        "mlp_hidden_2": 64,
                        "mlp_hidden_3": 32,
                        "mlp_dropout_1": 0.50,
                        "mlp_dropout_2": 0.40,
                    }
                )

hpo_root = Path(sett.DATA_DIR) / "hpo" / HPO_STAGE
hpo_root.mkdir(parents=True, exist_ok=True)

results = []

for idx, params in enumerate(HPO_RUNS, start=1):
    print("\n" + "=" * 80)
    print(f"[HPO {idx}/{len(HPO_RUNS)}] {params['run_name']}")
    print("=" * 80)

    cfg = TrainConfig(
        mode="binary_u25",
        model_version="v1",
        # v1 full scratch
        representation="full",
        use_strength_masks=True,
        use_position_embedding=True,
        use_team_strength=True,
        use_team_ids=True,
        use_comp_embedding=True,
        # training params
        learning_rate=params["learning_rate"],
        lr_schedule=params["lr_schedule"],
        lr_decay_rate=0.997,
        min_learning_rate=2e-5,
        batch_size=params["batch_size"],
        window_rounds=params["window_rounds"],
        epochs_per_step=params["epochs_per_step"],
        early_stopping_patience=params["early_stopping_patience"],
        early_stopping_min_delta=params["early_stopping_min_delta"],
        # current v1 architecture
        team_emb_dim=8,
        comp_emb_dim=5,
        strength_emb_dim=16,
        position_emb_dim=3,
        mlp_hidden_1=params["mlp_hidden_1"],
        mlp_hidden_2=params["mlp_hidden_2"],
        mlp_hidden_3=params["mlp_hidden_3"],
        mlp_dropout_1=params["mlp_dropout_1"],
        mlp_dropout_2=params["mlp_dropout_2"],
        seed=42,
        run_name=params["run_name"],
        enable_branch_diagnostics=False,
        save_oos_predictions=True,
    )

    _ = train_rolling(
        league_matches_sorted,
        cat_maps,
        cfg,
    )

    summary_path = Path(sett.DATA_DIR) / "tensorboard_logs" / params["run_name"] / "summary.json"
    config_path = Path(sett.DATA_DIR) / "tensorboard_logs" / params["run_name"] / "train_config.json"
    round_metrics_path = Path(sett.DATA_DIR) / "tensorboard_logs" / params["run_name"] / "round_metrics.csv"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    with config_path.open("r", encoding="utf-8") as f:
        saved_cfg = json.load(f)

    row = {
        "rank": None,
        "run_name": params["run_name"],
        "stage": HPO_STAGE,
        "model_version": saved_cfg["model_version"],
        "mode": saved_cfg["mode"],
        "learning_rate": saved_cfg["learning_rate"],
        "lr_schedule": saved_cfg["lr_schedule"],
        "lr_decay_rate": saved_cfg["lr_decay_rate"],
        "min_learning_rate": saved_cfg["min_learning_rate"],
        "epochs_per_step": saved_cfg["epochs_per_step"],
        "window_rounds": saved_cfg["window_rounds"],
        "batch_size": saved_cfg["batch_size"],
        "mlp_hidden_1": saved_cfg["mlp_hidden_1"],
        "mlp_hidden_2": saved_cfg["mlp_hidden_2"],
        "mlp_hidden_3": saved_cfg["mlp_hidden_3"],
        "mlp_dropout_1": saved_cfg["mlp_dropout_1"],
        "mlp_dropout_2": saved_cfg["mlp_dropout_2"],
        "pooled_accuracy": summary.get("pooled_accuracy"),
        "pooled_auc": summary.get("pooled_auc"),
        "pooled_brier": summary.get("pooled_brier"),
        "summary_path": str(summary_path),
        "config_path": str(config_path),
        "round_metrics_path": str(round_metrics_path),
    }

    results.append(row)

# Ranking priority:
# 1) AUC high
# 2) Brier low
# 3) Accuracy high
results.sort(
    key=lambda r: (
        -(r["pooled_auc"] if r["pooled_auc"] is not None else -999),
        (r["pooled_brier"] if r["pooled_brier"] is not None else 999),
        -(r["pooled_accuracy"] if r["pooled_accuracy"] is not None else -999),
    )
)

for i, row in enumerate(results, start=1):
    row["rank"] = i

csv_path = hpo_root / "hpo_stage1_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

json_path = hpo_root / "hpo_stage1_results.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

txt_path = hpo_root / "hpo_stage1_ranking.txt"
with txt_path.open("w", encoding="utf-8") as f:
    for r in results:
        f.write(
            f"{r['rank']}. {r['run_name']} | "
            f"lr={r['learning_rate']} | "
            f"sched={r['lr_schedule']} | "
            f"ep={r['epochs_per_step']} | "
            f"AUC={r['pooled_auc']:.6f} | "
            f"ACC={r['pooled_accuracy']:.6f} | "
            f"BRIER={r['pooled_brier']:.6f}\n"
        )

# Overview plot
top = results[: min(12, len(results))]
labels = [f"{r['rank']}. lr{r['learning_rate']:.0e}_{r['lr_schedule']}_ep{r['epochs_per_step']}" for r in top]

fig = plt.figure(figsize=(16, 10))

ax1 = fig.add_subplot(3, 1, 1)
ax1.bar(labels, [r["pooled_auc"] for r in top])
ax1.set_title("HPO Stage 1: pooled AUC")
ax1.tick_params(axis="x", rotation=45)

ax2 = fig.add_subplot(3, 1, 2)
ax2.bar(labels, [r["pooled_accuracy"] for r in top])
ax2.set_title("HPO Stage 1: pooled accuracy")
ax2.tick_params(axis="x", rotation=45)

ax3 = fig.add_subplot(3, 1, 3)
ax3.bar(labels, [r["pooled_brier"] for r in top])
ax3.set_title("HPO Stage 1: pooled Brier (lower is better)")
ax3.tick_params(axis="x", rotation=45)

plt.tight_layout()
plot_path = hpo_root / "hpo_stage1_top_runs_overview.png"
plt.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"[HPO] saved CSV to {csv_path}")
print(f"[HPO] saved JSON to {json_path}")
print(f"[HPO] saved ranking to {txt_path}")
print(f"[HPO] saved overview plot to {plot_path}")

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
