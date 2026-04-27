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

# from football_outcomes.training.train_mlp_rolling import transfer_pretrained_strength_branch_weights
from football_outcomes.training.train_mlp_rolling import (  # TrainConfig; build_model; set_global_seed; train_rolling
    StrengthPretrainConfig,
    train_strength_pretrain_rolling,
)
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu
from football_outcomes.utils.fs_player_skill_utils import match_fs_teams_to_sofifa_teams

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
# Structured-branch representation sweep:
#   branch versions: v1, v2
#   representations: full, no_positions, no_masks, skills_only
#   learning rates: 5e-5, 8e-5
# ------------------------------------------------------------

REPRESENTATIONS = {
    "full": {
        "use_strength_masks": True,
        "use_position_embedding": True,
        "label": "skills_masks_positions",
    },
    "no_positions": {
        "use_strength_masks": True,
        "use_position_embedding": False,
        "label": "skills_masks",
    },
    "no_masks": {
        "use_strength_masks": False,
        "use_position_embedding": True,
        "label": "skills_positions",
    },
    "skills_only": {
        "use_strength_masks": False,
        "use_position_embedding": False,
        "label": "skills_only",
    },
}

LEARNING_RATES = [5e-5, 8e-5]
BRANCH_VERSIONS = ["v1", "v2"]

PRETRAIN_RUNS = []

for branch_version in BRANCH_VERSIONS:
    for rep_name, rep_cfg in REPRESENTATIONS.items():
        for lr in LEARNING_RATES:
            lr_tag = f"{lr:.0e}".replace("-", "m").replace("+", "").replace("e", "e")
            run_name = f"strength_pretrain_{branch_version}_u25_{rep_name}_lr{lr_tag}"

            base = {
                "branch_version": branch_version,
                "mode": "binary_u25",
                "window_rounds": 25,
                "learning_rate": lr,
                "batch_size": 64,
                "strength_emb_dim": 16,
                "position_emb_dim": 3,
                "representation": rep_name,
                "use_strength_masks": rep_cfg["use_strength_masks"],
                "use_position_embedding": rep_cfg["use_position_embedding"],
                "run_name": run_name,
            }

            if branch_version == "v1":
                base.update(
                    {
                        "epochs_per_step": 3,
                    }
                )
            else:
                base.update(
                    {
                        "epochs_per_step": 2,
                        "player_row_hidden_dim": 32,
                        "role_post_hidden_dim": 32,
                        "team_branch_dim": 32,
                        "team_dropout": 0.25,
                        "team_l2": 5e-5,
                        "compare_hidden_dim": 32,
                        "compare_dropout": 0.20,
                    }
                )

            PRETRAIN_RUNS.append(base)

pretrain_root = Path(sett.DATA_DIR) / "pretraining_representation_sweep"
pretrain_root.mkdir(parents=True, exist_ok=True)

results = []

for i, params in enumerate(PRETRAIN_RUNS, start=1):
    print("\n" + "=" * 80)
    print(f"[REP-SWEEP {i}/{len(PRETRAIN_RUNS)}] {params['run_name']}")
    print("=" * 80)

    cfg = StrengthPretrainConfig(**params)
    _ = train_strength_pretrain_rolling(league_matches_sorted, cfg)

    summary_path = Path(sett.DATA_DIR) / "tensorboard_logs" / params["run_name"] / "summary.json"
    config_path = Path(sett.DATA_DIR) / "tensorboard_logs" / params["run_name"] / "pretrain_config.json"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    with config_path.open("r", encoding="utf-8") as f:
        saved_cfg = json.load(f)

    row = {
        "run_name": params["run_name"],
        "branch_version": saved_cfg["branch_version"],
        "representation": saved_cfg["representation"],
        "use_strength_masks": saved_cfg["use_strength_masks"],
        "use_position_embedding": saved_cfg["use_position_embedding"],
        "learning_rate": saved_cfg["learning_rate"],
        "epochs_per_step": saved_cfg["epochs_per_step"],
        "window_rounds": saved_cfg["window_rounds"],
        "batch_size": saved_cfg["batch_size"],
        "strength_emb_dim": saved_cfg["strength_emb_dim"],
        "position_emb_dim": saved_cfg["position_emb_dim"],
        "pooled_accuracy": summary.get("pooled_accuracy"),
        "pooled_auc": summary.get("pooled_auc"),
        "pooled_brier": summary.get("pooled_brier"),
    }
    results.append(row)

results.sort(
    key=lambda r: (
        -(r["pooled_auc"] if r["pooled_auc"] is not None else -999),
        -(r["pooled_accuracy"] if r["pooled_accuracy"] is not None else -999),
        (r["pooled_brier"] if r["pooled_brier"] is not None else 999),
    )
)

csv_path = pretrain_root / "representation_pretraining_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

json_path = pretrain_root / "representation_pretraining_results.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

txt_path = pretrain_root / "representation_pretraining_ranking.txt"
with txt_path.open("w", encoding="utf-8") as f:
    for idx, r in enumerate(results, start=1):
        f.write(
            f"{idx}. {r['run_name']} | "
            f"branch={r['branch_version']} | "
            f"repr={r['representation']} | "
            f"lr={r['learning_rate']} | "
            f"AUC={r['pooled_auc']:.6f} | "
            f"ACC={r['pooled_accuracy']:.6f} | "
            f"BRIER={r['pooled_brier']:.6f}\n"
        )

print(f"[REP-SWEEP] saved CSV to {csv_path}")
print(f"[REP-SWEEP] saved JSON to {json_path}")
print(f"[REP-SWEEP] saved ranking to {txt_path}")

# Compact overview plot, sorted by AUC
labels = [f"{r['branch_version']}-{r['representation']}-lr{r['learning_rate']:.0e}" for r in results]
accs = [r["pooled_accuracy"] for r in results]
aucs = [r["pooled_auc"] for r in results]
briers = [r["pooled_brier"] for r in results]

fig = plt.figure(figsize=(18, 10))

ax1 = fig.add_subplot(3, 1, 1)
ax1.bar(labels, accs)
ax1.set_title("Representation sweep: pooled accuracy")
ax1.tick_params(axis="x", rotation=60)

ax2 = fig.add_subplot(3, 1, 2)
ax2.bar(labels, aucs)
ax2.set_title("Representation sweep: pooled AUC")
ax2.tick_params(axis="x", rotation=60)

ax3 = fig.add_subplot(3, 1, 3)
ax3.bar(labels, briers)
ax3.set_title("Representation sweep: pooled Brier (lower is better)")
ax3.tick_params(axis="x", rotation=60)

plt.tight_layout()
plot_path = pretrain_root / "representation_pretraining_overview.png"
plt.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"[REP-SWEEP] saved overview plot to {plot_path}")

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
