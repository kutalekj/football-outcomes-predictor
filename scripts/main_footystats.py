from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
import tensorflow as tf

import football_outcomes.config.fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import load_avg_team_strength, load_sofifa_players, save_snapshot, try_load_snapshot
from football_outcomes.data.fs_models import FSDataBundle
from football_outcomes.data.fs_retrieve import fill_globals_with_cache, retrieve_new_data

# from football_outcomes.training.fs_classical_baselines import BaselineConfig, evaluate_baseline_rolling
from football_outcomes.training.fs_training_utils import build_categorical_maps, extract_numerical_features
from football_outcomes.training.train_mlp_rolling import (
    StrengthPretrainConfig,
    TrainConfig,
    build_model,
    set_global_seed,
    train_rolling,
    train_strength_pretrain_rolling,
    transfer_pretrained_strength_branch_weights,
)
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu
from football_outcomes.utils.fs_player_skill_utils import match_fs_teams_to_sofifa_teams

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
# Transfer top-3 v1 pretrained branch representations
# ------------------------------------------------------------

TOP_REPRESENTATIONS = [
    {
        "representation": "no_masks",
        "use_strength_masks": False,
        "use_position_embedding": True,
        "pretrain_run_name": "strength_pretrain_v1_u25_no_masks_lr8em05",
        "use_pretrained": True,
    },
    {
        "representation": "full",
        "use_strength_masks": True,
        "use_position_embedding": True,
        "pretrain_run_name": "strength_pretrain_v1_u25_full_lr8em05",
        "use_pretrained": False,
    },
    {
        "representation": "full",
        "use_strength_masks": True,
        "use_position_embedding": True,
        "pretrain_run_name": "strength_pretrain_v1_u25_full_lr8em05",
        "use_pretrained": True,
    },
]

sample_feat = league_matches_sorted[0].features_before_match
num_num = extract_numerical_features(sample_feat).shape[0]
num_teams = len(cat_maps.team_id_map)
num_comps = len(cat_maps.comp_id_map)

transfer_root = Path(sett.DATA_DIR) / "transfer_top3_representations"
transfer_root.mkdir(parents=True, exist_ok=True)

results = []

for rep in TOP_REPRESENTATIONS:
    use_pretrained = rep["use_pretrained"]
    label = "pretrained" if use_pretrained else "scratch"
    run_name = f"mlp_v1_{rep['representation']}_{label}_lr8em05_diag"

    print("\n" + "=" * 80)
    print(f"[TRANSFER] {run_name}")
    print("=" * 80)

    cfg = TrainConfig(
        mode="binary_u25",
        model_version="v1",
        learning_rate=8e-5,
        batch_size=64,
        window_rounds=25,
        epochs_per_step=3,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        seed=42,
        run_name=run_name,
        enable_branch_diagnostics=True,
        representation=rep["representation"],
        use_strength_masks=rep["use_strength_masks"],
        use_position_embedding=rep["use_position_embedding"],
        strength_emb_dim=16,
        position_emb_dim=3,
    )

    if use_pretrained:
        pretrained_path = Path(sett.DATA_DIR) / "tensorboard_logs" / rep["pretrain_run_name"] / "pretrained_model.keras"

        if pretrained_path.exists():
            print(f"[transfer] loading pretrained model: {pretrained_path}")
            pretrained_model = tf.keras.models.load_model(pretrained_path)
        else:
            print(f"[transfer] pretrained model missing, rerunning: {rep['pretrain_run_name']}")
            pre_cfg = StrengthPretrainConfig(
                branch_version="v1",
                mode="binary_u25",
                window_rounds=25,
                epochs_per_step=3,
                learning_rate=8e-5,
                batch_size=64,
                strength_emb_dim=16,
                position_emb_dim=3,
                representation=rep["representation"],
                use_strength_masks=rep["use_strength_masks"],
                use_position_embedding=rep["use_position_embedding"],
                run_name=rep["pretrain_run_name"],
            )
            pretrained_model = train_strength_pretrain_rolling(league_matches_sorted, pre_cfg)

        if cfg.seed is not None:
            set_global_seed(cfg.seed)

        model = build_model(
            num_num=num_num,
            num_teams=num_teams,
            num_comps=num_comps,
            cfg=cfg,
        )

        transfer_pretrained_strength_branch_weights(
            pretrained_model=pretrained_model,
            full_model=model,
            branch_version="v1",
        )

        _ = train_rolling(
            league_matches_sorted,
            cat_maps,
            cfg,
            model=model,
            pretrained_branch_version="v1",
        )
    else:
        _ = train_rolling(
            league_matches_sorted,
            cat_maps,
            cfg,
        )

    summary_path = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    results.append(
        {
            "run_name": run_name,
            "representation": rep["representation"],
            "use_strength_masks": rep["use_strength_masks"],
            "use_position_embedding": rep["use_position_embedding"],
            "use_pretrained_init": use_pretrained,
            "pooled_accuracy": summary.get("pooled_accuracy"),
            "pooled_auc": summary.get("pooled_auc"),
            "pooled_brier": summary.get("pooled_brier"),
        }
    )

results.sort(
    key=lambda r: (
        -(r["pooled_auc"] if r["pooled_auc"] is not None else -999),
        -(r["pooled_accuracy"] if r["pooled_accuracy"] is not None else -999),
        (r["pooled_brier"] if r["pooled_brier"] is not None else 999),
    )
)

csv_path = transfer_root / "top3_transfer_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

json_path = transfer_root / "top3_transfer_results.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

txt_path = transfer_root / "top3_transfer_ranking.txt"
with txt_path.open("w", encoding="utf-8") as f:
    for idx, r in enumerate(results, start=1):
        f.write(
            f"{idx}. {r['run_name']} | "
            f"repr={r['representation']} | "
            f"pretrained={r['use_pretrained_init']} | "
            f"AUC={r['pooled_auc']:.6f} | "
            f"ACC={r['pooled_accuracy']:.6f} | "
            f"BRIER={r['pooled_brier']:.6f}\n"
        )

print(f"[TRANSFER] saved CSV to {csv_path}")
print(f"[TRANSFER] saved JSON to {json_path}")
print(f"[TRANSFER] saved ranking to {txt_path}")

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
