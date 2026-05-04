from __future__ import annotations

import csv
import json

# import os
from pathlib import Path
from typing import Any

import matplotlib

import football_outcomes.config.fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import load_avg_team_strength, load_sofifa_players, try_load_snapshot
from football_outcomes.data.fs_retrieve import fill_globals_with_cache
from football_outcomes.training.fs_training_utils import build_categorical_maps, distribute_matches_into_rounds
from football_outcomes.training.train_mlp_rolling import TrainConfig, train_rolling
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu
from football_outcomes.utils.fs_player_skill_utils import match_fs_teams_to_sofifa_teams

matplotlib.use("Agg")

OUT_ROOT = Path(sett.DATA_DIR) / "comparison" / "dataset_sensitivity"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SELECTED_MODEL_CFG = {
    "learning_rate": 8e-5,
    "lr_schedule": "exponential",
    "lr_decay_rate": 0.997,
    "min_learning_rate": 2e-5,
    "epochs_per_step": 2,
    "window_rounds": 25,
    "batch_size": 64,
    "team_emb_dim": 8,
    "comp_emb_dim": 5,
    "strength_emb_dim": 24,
    "position_emb_dim": 3,
    "mlp_hidden_1": 128,
    "mlp_hidden_2": 64,
    "mlp_hidden_3": 32,
    "mlp_dropout_1": 0.30,
    "mlp_dropout_2": 0.20,
    "early_stopping_patience": 1,
    "early_stopping_min_delta": 0.0,
    "seed": 123,
}


def log_feature_error(variant: str, msg: str) -> None:
    path = OUT_ROOT / f"feature_errors_{variant}.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip("\n") + "\n")


def load_common_state() -> None:
    g = Global.get_instance()

    load_avg_team_strength()

    cache = try_load_snapshot()
    if sett.ALL_LOAD and cache is not None:
        fill_globals_with_cache(cache, update_leagues_list=False)

    load_sofifa_players(
        rebuild=getattr(sett, "REBUILD_SOFIFA_FROM_CSV", False),
        debug_shifts=False,
    )

    print("[sofifa] snapshots:", len(g.sofifa_snapshots))

    utils.link_matches_to_comp_seasons()
    utils.ensure_comp_season_dates(force=False)
    match_fs_teams_to_sofifa_teams(force=False)


def patch_less_restricted_settings(all_matches_sorted: list) -> dict[str, Any]:
    """
    Temporarily modifies settings for this script process only.

    - Restores ignored offsides by setting IGNORED_MATCH_STATS to empty.
    - Adds conservative valid-round fallbacks for the four restored excluded seasons
      if they are missing from LEAGUE_VALID_ROUND_IDS_BY_SEASON.
    """
    original_ignored = set(getattr(sett, "IGNORED_MATCH_STATS", set()))
    original_valid_rounds = {key: set(value) for key, value in sett.LEAGUE_VALID_ROUND_IDS_BY_SEASON.items()}

    inferred = utils.infer_initial_phase_round_ids_for_missing_league_seasons(all_matches_sorted)

    for key, round_ids in inferred.items():
        if key in sett.EXCLUDED_COMP_SEASONS:
            sett.LEAGUE_VALID_ROUND_IDS_BY_SEASON[key] = set(round_ids)
            print(f"[less_restricted] inferred valid round ids for {key}: {sorted(round_ids)}")

    sett.IGNORED_MATCH_STATS = set()
    sett.INCLUDE_OFFSIDES_FEATURES = True

    return {
        "original_ignored": sorted(original_ignored),
        "original_valid_rounds_count": len(original_valid_rounds),
        "include_offsides_features": True,
        "offsides_norm_coefficient": getattr(sett, "OFFSIDES_NORM_COEFFICIENT", None),
        "inferred_valid_rounds": {
            f"{k[0]} {k[1]}": sorted(v) for k, v in inferred.items() if k in sett.EXCLUDED_COMP_SEASONS
        },
    }


def select_matches_for_variant(all_matches_sorted: list, variant: str) -> list:
    if variant == "clean":
        matches = utils.filter_clean_league_matches(all_matches_sorted)
    elif variant == "less_restricted":
        matches = utils.filter_league_matches_including_excluded_comp_seasons(all_matches_sorted)
    else:
        raise ValueError(f"Unknown dataset variant: {variant}")

    matches = utils.filter_valid_round_matches(matches)

    matches = [
        m
        for m in matches
        if getattr(m, "season", None) is not None and sett.FIRST_SEASON <= int(m.season) < sett.LAST_SEASON
    ]

    return sorted(matches, key=fu.match_sort_key)


def clear_features(matches: list) -> None:
    for m in matches:
        m.features_before_match = None
        m.home_elo_after_match_raw = None
        m.away_elo_after_match_raw = None


def compute_features_for_variant(variant: str, all_matches_sorted: list, variant_matches_sorted: list) -> list:
    clear_features(all_matches_sorted)

    # Rebuild league tables after possible valid-round setting changes.
    utils.initialize_league_tables(precompute_positions=True, force_rebuild=True)

    team_index_all = fu.build_team_match_index(all_matches_sorted)
    team_index_league = fu.build_team_match_index(variant_matches_sorted)

    usable = []
    skipped = 0
    last_progress_month = None

    total = len(variant_matches_sorted)

    for i, match in enumerate(variant_matches_sorted, start=1):
        dt = match.datetime
        curr_month = (dt.year, dt.month) if dt is not None else None

        if curr_month != last_progress_month:
            last_progress_month = curr_month
            if curr_month is not None:
                print(f"[{variant} features] {dt.year:04d}-{dt.month:02d} ({i}/{total})")

        try:
            match.features_before_match = match.calculate_match_features(
                team_index_league=team_index_league,
                team_index_all=team_index_all,
            )
            usable.append(match)

        except Exception as e:
            skipped += 1
            log_feature_error(
                variant,
                f"[SKIP] match_id={getattr(match, 'id', None)} "
                f"{getattr(match, 'comp_name', None)} {getattr(match, 'season', None)} "
                f"{getattr(match, 'datetime', None)} "
                f"{getattr(getattr(match, 'home_team', None), 'name', None)} vs "
                f"{getattr(getattr(match, 'away_team', None), 'name', None)} "
                f"error={repr(e)}",
            )

    print(f"[{variant} features] usable={len(usable)} skipped={skipped}")
    return usable


def dataset_summary(
    variant: str, matches: list, usable: list, patch_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    by_comp: dict[str, int] = {}
    by_comp_season: dict[str, int] = {}

    for m in usable:
        comp = getattr(m, "comp_name", "<unknown>")
        season = getattr(m, "season", "<unknown>")

        by_comp[comp] = by_comp.get(comp, 0) + 1
        by_comp_season[f"{comp} {season}"] = by_comp_season.get(f"{comp} {season}", 0) + 1

    rounds = distribute_matches_into_rounds(usable)

    return {
        "variant": variant,
        "selected_before_feature_calculation": len(matches),
        "usable_after_feature_calculation": len(usable),
        "num_rounds": len(rounds),
        "matches_by_competition": dict(sorted(by_comp.items())),
        "matches_by_competition_season": dict(sorted(by_comp_season.items())),
        "patch_info": patch_info or {},
    }


def train_selected_mlp(variant: str, matches: list):
    cat_maps = build_categorical_maps(matches)

    cfg = TrainConfig(
        mode="binary_u25",
        model_version="v1",
        representation="full",
        use_strength_masks=True,
        use_position_embedding=True,
        use_team_strength=True,
        use_team_ids=True,
        use_comp_embedding=True,
        learning_rate=SELECTED_MODEL_CFG["learning_rate"],
        lr_schedule=SELECTED_MODEL_CFG["lr_schedule"],
        lr_decay_rate=SELECTED_MODEL_CFG["lr_decay_rate"],
        min_learning_rate=SELECTED_MODEL_CFG["min_learning_rate"],
        epochs_per_step=SELECTED_MODEL_CFG["epochs_per_step"],
        window_rounds=SELECTED_MODEL_CFG["window_rounds"],
        batch_size=SELECTED_MODEL_CFG["batch_size"],
        team_emb_dim=SELECTED_MODEL_CFG["team_emb_dim"],
        comp_emb_dim=SELECTED_MODEL_CFG["comp_emb_dim"],
        strength_emb_dim=SELECTED_MODEL_CFG["strength_emb_dim"],
        position_emb_dim=SELECTED_MODEL_CFG["position_emb_dim"],
        mlp_hidden_1=SELECTED_MODEL_CFG["mlp_hidden_1"],
        mlp_hidden_2=SELECTED_MODEL_CFG["mlp_hidden_2"],
        mlp_hidden_3=SELECTED_MODEL_CFG["mlp_hidden_3"],
        mlp_dropout_1=SELECTED_MODEL_CFG["mlp_dropout_1"],
        mlp_dropout_2=SELECTED_MODEL_CFG["mlp_dropout_2"],
        early_stopping_patience=SELECTED_MODEL_CFG["early_stopping_patience"],
        early_stopping_min_delta=SELECTED_MODEL_CFG["early_stopping_min_delta"],
        seed=SELECTED_MODEL_CFG["seed"],
        run_name=f"dataset_sensitivity_{variant}_selected_mlp_binary_u25",
        enable_branch_diagnostics=False,
        save_oos_predictions=True,
    )

    train_rolling(matches, cat_maps, cfg)

    summary_path = Path(sett.DATA_DIR) / "tensorboard_logs" / cfg.run_name / "summary.json"
    round_metrics_path = Path(sett.DATA_DIR) / "tensorboard_logs" / cfg.run_name / "round_metrics.csv"
    oos_predictions_path = Path(sett.DATA_DIR) / "tensorboard_logs" / cfg.run_name / "oos_predictions.csv"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    return {
        "variant": variant,
        "run_name": cfg.run_name,
        "summary_path": str(summary_path),
        "round_metrics_path": str(round_metrics_path),
        "oos_predictions_path": str(oos_predictions_path),
        **summary,
    }


def write_outputs(dataset_summaries: list[dict], run_results: list[dict]) -> None:
    dataset_summary_path = OUT_ROOT / "dataset_sensitivity_dataset_summaries.json"
    with dataset_summary_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_summaries, f, indent=2)

    results_json_path = OUT_ROOT / "dataset_sensitivity_results.json"
    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump(run_results, f, indent=2)

    results_csv_path = OUT_ROOT / "dataset_sensitivity_results.csv"
    fieldnames = sorted({k for r in run_results for k in r.keys()})
    with results_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_results)

    ranking_path = OUT_ROOT / "dataset_sensitivity_ranking.txt"
    ranked = sorted(
        run_results,
        key=lambda r: (
            -(r.get("pooled_auc") or -999),
            r.get("pooled_brier") or 999,
            -(r.get("pooled_accuracy") or -999),
        ),
    )
    with ranking_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(ranked, start=1):
            f.write(
                f"{i}. {r['run_name']} | "
                f"AUC={r.get('pooled_auc'):.6f} | "
                f"ACC={r.get('pooled_accuracy'):.6f} | "
                f"BRIER={r.get('pooled_brier'):.6f}\n"
            )

    print(f"[saved] {dataset_summary_path}")
    print(f"[saved] {results_json_path}")
    print(f"[saved] {results_csv_path}")
    print(f"[saved] {ranking_path}")


def main() -> None:
    load_common_state()

    g = Global.get_instance()
    all_matches_sorted = sorted(g.all_matches, key=fu.match_sort_key)

    dataset_summaries = []
    run_results = []

    # ------------------------------------------------------------------
    # Variant 1: clean dataset
    # ------------------------------------------------------------------
    sett.INCLUDE_OFFSIDES_FEATURES = False

    clean_selected = select_matches_for_variant(all_matches_sorted, "clean")
    clean_usable = compute_features_for_variant("clean", all_matches_sorted, clean_selected)

    dataset_summaries.append(dataset_summary("clean", clean_selected, clean_usable))

    run_results.append(train_selected_mlp("clean", clean_usable))

    # ------------------------------------------------------------------
    # Variant 2: less-restricted dataset
    # ------------------------------------------------------------------
    patch_info = patch_less_restricted_settings(all_matches_sorted)

    less_selected = select_matches_for_variant(all_matches_sorted, "less_restricted")
    less_usable = compute_features_for_variant("less_restricted", all_matches_sorted, less_selected)

    dataset_summaries.append(dataset_summary("less_restricted", less_selected, less_usable, patch_info=patch_info))

    run_results.append(train_selected_mlp("less_restricted", less_usable))

    write_outputs(dataset_summaries, run_results)


if __name__ == "__main__":
    main()
