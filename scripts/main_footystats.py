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
from football_outcomes.training.fs_training_utils import build_categorical_maps
from football_outcomes.training.train_mlp_rolling import TrainConfig, train_rolling
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu
from football_outcomes.utils.fs_player_skill_utils import match_fs_teams_to_sofifa_teams

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

        if match.home_team.name == "KRC Genk" or match.away_team.name == "KRC Genk":
            fu.debug_print_match_and_features(match)  # DEBUG

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
# Diagnosable MLP run
# ------------------------------------------------------------
# ------------------------------------------------------------
# Focused sweep on training dynamics
# ------------------------------------------------------------

SWEEP_RUNS = [
    # ------------------------------------------------------------
    # v1 family: 15 runs
    # epochs fixed to 3, batch fixed to 64
    # sweep: learning_rate x window_rounds
    # ------------------------------------------------------------
    {"model_version": "v1", "learning_rate": 2e-5, "window_rounds": 20, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 2e-5, "window_rounds": 25, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 2e-5, "window_rounds": 30, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 3e-5, "window_rounds": 20, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 3e-5, "window_rounds": 25, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 3e-5, "window_rounds": 30, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 5e-5, "window_rounds": 20, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 5e-5, "window_rounds": 25, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 5e-5, "window_rounds": 30, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 7e-5, "window_rounds": 20, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 7e-5, "window_rounds": 25, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 7e-5, "window_rounds": 30, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 1e-4, "window_rounds": 20, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 1e-4, "window_rounds": 25, "epochs_per_step": 3, "batch_size": 64},
    {"model_version": "v1", "learning_rate": 1e-4, "window_rounds": 30, "epochs_per_step": 3, "batch_size": 64},
    # ------------------------------------------------------------
    # v2-lite family: 15 runs
    # epochs fixed to 2, batch fixed to 64
    # sweep: learning_rate x window_rounds x regularization profile
    # ------------------------------------------------------------
    # Standard regularization
    {"model_version": "v2", "learning_rate": 3e-5, "window_rounds": 20, "epochs_per_step": 2, "batch_size": 64},
    {"model_version": "v2", "learning_rate": 3e-5, "window_rounds": 25, "epochs_per_step": 2, "batch_size": 64},
    {"model_version": "v2", "learning_rate": 3e-5, "window_rounds": 30, "epochs_per_step": 2, "batch_size": 64},
    {"model_version": "v2", "learning_rate": 5e-5, "window_rounds": 20, "epochs_per_step": 2, "batch_size": 64},
    {"model_version": "v2", "learning_rate": 5e-5, "window_rounds": 25, "epochs_per_step": 2, "batch_size": 64},
    {"model_version": "v2", "learning_rate": 5e-5, "window_rounds": 30, "epochs_per_step": 2, "batch_size": 64},
    {"model_version": "v2", "learning_rate": 7e-5, "window_rounds": 20, "epochs_per_step": 2, "batch_size": 64},
    {"model_version": "v2", "learning_rate": 7e-5, "window_rounds": 25, "epochs_per_step": 2, "batch_size": 64},
    {"model_version": "v2", "learning_rate": 7e-5, "window_rounds": 30, "epochs_per_step": 2, "batch_size": 64},
    # Stronger regularization around the most plausible area
    {
        "model_version": "v2",
        "learning_rate": 3e-5,
        "window_rounds": 25,
        "epochs_per_step": 2,
        "batch_size": 64,
        "team_dropout": 0.30,
        "fusion_dropout_1": 0.50,
        "fusion_dropout_2": 0.35,
        "team_l2": 7e-5,
        "fusion_l2": 7e-5,
    },
    {
        "model_version": "v2",
        "learning_rate": 5e-5,
        "window_rounds": 20,
        "epochs_per_step": 2,
        "batch_size": 64,
        "team_dropout": 0.30,
        "fusion_dropout_1": 0.50,
        "fusion_dropout_2": 0.35,
        "team_l2": 7e-5,
        "fusion_l2": 7e-5,
    },
    {
        "model_version": "v2",
        "learning_rate": 5e-5,
        "window_rounds": 25,
        "epochs_per_step": 2,
        "batch_size": 64,
        "team_dropout": 0.30,
        "fusion_dropout_1": 0.50,
        "fusion_dropout_2": 0.35,
        "team_l2": 7e-5,
        "fusion_l2": 7e-5,
    },
    {
        "model_version": "v2",
        "learning_rate": 5e-5,
        "window_rounds": 30,
        "epochs_per_step": 2,
        "batch_size": 64,
        "team_dropout": 0.30,
        "fusion_dropout_1": 0.50,
        "fusion_dropout_2": 0.35,
        "team_l2": 7e-5,
        "fusion_l2": 7e-5,
    },
    {
        "model_version": "v2",
        "learning_rate": 7e-5,
        "window_rounds": 25,
        "epochs_per_step": 2,
        "batch_size": 64,
        "team_dropout": 0.30,
        "fusion_dropout_1": 0.50,
        "fusion_dropout_2": 0.35,
        "team_l2": 7e-5,
        "fusion_l2": 7e-5,
    },
    {
        "model_version": "v2",
        "learning_rate": 7e-5,
        "window_rounds": 30,
        "epochs_per_step": 2,
        "batch_size": 64,
        "team_dropout": 0.30,
        "fusion_dropout_1": 0.50,
        "fusion_dropout_2": 0.35,
        "team_l2": 7e-5,
        "fusion_l2": 7e-5,
    },
]

sweep_root = Path(sett.DATA_DIR) / "sweeps"
sweep_root.mkdir(parents=True, exist_ok=True)

results = []

for i, params in enumerate(SWEEP_RUNS, start=1):
    run_name = (
        f"{params['model_version']}"
        f"_u25"
        f"_lr{params['learning_rate']}"
        f"_wr{params['window_rounds']}"
        f"_ep{params['epochs_per_step']}"
        f"_bs{params['batch_size']}"
    ).replace(".", "p")

    if "team_dropout" in params:
        run_name += (
            f"_td{params['team_dropout']}"
            f"_fd1{params['fusion_dropout_1']}"
            f"_fd2{params['fusion_dropout_2']}"
            f"_tl2{params['team_l2']}"
            f"_fl2{params['fusion_l2']}"
        ).replace(".", "p")

    print("\n" + "=" * 80)
    print(f"[SWEEP {i}/{len(SWEEP_RUNS)}] {run_name}")
    print("=" * 80)

    base_cfg = {
        "mode": "binary_u25",
        "model_version": params["model_version"],
        "use_team_aux_head": False,
        "aux_task": None,
        "seed": 42,
        "run_name": run_name,
        "enable_branch_diagnostics": False,  # keep sweep fast
        "early_stopping_patience": 1,
        "early_stopping_min_delta": 0.0,
    }

    # allow sweep params to override TrainConfig fields
    cfg_kwargs = {**base_cfg, **params}

    cfg = TrainConfig(**cfg_kwargs)

    _ = train_rolling(league_matches_sorted, cat_maps, cfg)

    summary_path = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name / "summary.json"
    config_path = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name / "train_config.json"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    with config_path.open("r", encoding="utf-8") as f:
        saved_cfg = json.load(f)

    row = {
        "run_name": run_name,
        "model_version": saved_cfg["model_version"],
        "learning_rate": saved_cfg["learning_rate"],
        "epochs_per_step": saved_cfg["epochs_per_step"],
        "batch_size": saved_cfg["batch_size"],
        "pooled_accuracy": summary.get("pooled_accuracy"),
        "pooled_auc": summary.get("pooled_auc"),
        "pooled_brier": summary.get("pooled_brier"),
    }
    results.append(row)

# Sort: best AUC first, then best accuracy, then lowest Brier
results.sort(
    key=lambda r: (
        -(r["pooled_auc"] if r["pooled_auc"] is not None else -999),
        -(r["pooled_accuracy"] if r["pooled_accuracy"] is not None else -999),
        (r["pooled_brier"] if r["pooled_brier"] is not None else 999),
    )
)

# Save CSV
csv_path = sweep_root / "overnight_sweep_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"[SWEEP] Saved results CSV to {csv_path}")

# Save simple text ranking
txt_path = sweep_root / "overnight_sweep_ranking.txt"
with txt_path.open("w", encoding="utf-8") as f:
    for idx, r in enumerate(results, start=1):
        f.write(
            f"{idx}. {r['run_name']} | "
            f"AUC={r['pooled_auc']:.6f} | "
            f"ACC={r['pooled_accuracy']:.6f} | "
            f"BRIER={r['pooled_brier']:.6f}\n"
        )

print(f"[SWEEP] Saved ranking to {txt_path}")

# Make one compact comparison figure
labels = [r["run_name"] for r in results]
accs = [r["pooled_accuracy"] for r in results]
aucs = [r["pooled_auc"] for r in results]
briers = [r["pooled_brier"] for r in results]

fig = plt.figure(figsize=(16, 10))

ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(labels, accs, marker="o")
ax1.tick_params(axis="x", rotation=45)

ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(labels, aucs, marker="o")
ax2.tick_params(axis="x", rotation=45)

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(labels, briers, marker="o")
ax3.tick_params(axis="x", rotation=45)

ax1.set_title("Overnight sweep: pooled accuracy")
ax2.set_title("Overnight sweep: pooled AUC")
ax3.set_title("Overnight sweep: pooled Brier (lower is better)")

plt.tight_layout()
plot_path = sweep_root / "overnight_sweep_overview.png"
plt.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"[SWEEP] Saved overview plot to {plot_path}")

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
