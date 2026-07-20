from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

import football_outcomes.config.fs_settings as sett
from football_outcomes.application.snapshot_selection import (
    resolve_snapshot_path,
)
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.snapshots import try_load_snapshot
from football_outcomes.data.sofifa_ingestion import load_avg_team_strength, load_sofifa_players
from football_outcomes.data.sofifa_team_matching import (
    match_fs_teams_to_sofifa_teams,
)
from football_outcomes.data.state import (
    apply_bundle_to_global,
)
from football_outcomes.training.fs_training_utils import build_categorical_maps, distribute_matches_into_rounds
from football_outcomes.training.train_mlp_rolling import TrainConfig, train_rolling
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu

matplotlib.use("Agg")

OUT_ROOT = Path(sett.DATA_DIR) / "comparison" / "missing_skill_sensitivity"
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


def log_feature_error(msg: str) -> None:
    os.makedirs(sett.LOG_DIR, exist_ok=True)
    path = OUT_ROOT / "feature_errors_missing_skill_sensitivity.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip("\n") + "\n")


def load_common_state() -> None:
    g = Global.get_instance()

    load_avg_team_strength()

    cache = try_load_snapshot(resolve_snapshot_path())

    if cache is None:
        raise RuntimeError("Could not load the explicitly " "selected snapshot.")

    apply_bundle_to_global(cache)

    load_sofifa_players(
        rebuild=getattr(sett, "REBUILD_SOFIFA_FROM_CSV", False),
        debug_shifts=False,
    )

    print("[sofifa] snapshots:", len(g.sofifa_snapshots))

    utils.link_matches_to_comp_seasons()
    utils.ensure_comp_season_dates(force=False)
    utils.initialize_league_tables(
        precompute_positions=True,
        force_rebuild=False,
    )
    match_fs_teams_to_sofifa_teams(force=False)


def prepare_clean_matches() -> list:
    g = Global.get_instance()
    all_matches_sorted = sorted(g.all_matches, key=fu.match_sort_key)

    league_matches_sorted = utils.filter_clean_league_matches(all_matches_sorted)
    league_matches_sorted = utils.filter_valid_round_matches(league_matches_sorted)
    league_matches_sorted = [
        m
        for m in league_matches_sorted
        if getattr(m, "season", None) is not None and sett.FIRST_SEASON <= int(m.season) < sett.LAST_SEASON
    ]
    league_matches_sorted = sorted(league_matches_sorted, key=fu.match_sort_key)

    team_index_all = fu.build_team_match_index(all_matches_sorted)
    team_index_league = fu.build_team_match_index(league_matches_sorted)

    usable = []
    skipped = 0
    last_progress_month = None

    for i, match in enumerate(league_matches_sorted, start=1):
        dt = match.datetime
        curr_month = (dt.year, dt.month) if dt is not None else None

        if curr_month != last_progress_month:
            last_progress_month = curr_month
            if dt is not None:
                print(f"[features] {dt.year:04d}-{dt.month:02d} ({i}/{len(league_matches_sorted)})")

        try:
            match.features_before_match = match.calculate_match_features(
                team_index_league=team_index_league,
                team_index_all=team_index_all,
            )
            usable.append(match)
        except Exception as e:
            skipped += 1
            log_feature_error(
                f"[SKIP] match_id={getattr(match, 'id', None)} "
                f"{getattr(match, 'comp_name', None)} {getattr(match, 'season', None)} "
                f"{getattr(match, 'datetime', None)} "
                f"error={repr(e)}"
            )

    print(f"[features] usable={len(usable)} skipped={skipped}")
    return usable


def missing_skill_summary(matches: list) -> dict:
    total_cells = 0
    missing_cells = 0
    total_player_rows = 0
    fully_missing_rows = 0

    by_comp = {}

    def consume(comp: str, mat):
        nonlocal total_cells, missing_cells, total_player_rows, fully_missing_rows

        if mat is None:
            return

        import numpy as np

        arr = np.asarray(mat, dtype=np.float32)
        if arr.shape != (sett.TEAM_STRENGTH_NUM_PLAYERS, sett.TEAM_STRENGTH_NUM_SKILLS):
            return

        comp_rec = by_comp.setdefault(
            comp,
            {
                "total_cells": 0,
                "missing_cells": 0,
                "total_player_rows": 0,
                "fully_missing_rows": 0,
            },
        )

        miss = arr < 0.0

        total_cells += int(arr.size)
        missing_cells += int(miss.sum())
        total_player_rows += int(arr.shape[0])
        fully_missing_rows += int(np.all(miss, axis=1).sum())

        comp_rec["total_cells"] += int(arr.size)
        comp_rec["missing_cells"] += int(miss.sum())
        comp_rec["total_player_rows"] += int(arr.shape[0])
        comp_rec["fully_missing_rows"] += int(np.all(miss, axis=1).sum())

    for m in matches:
        f = getattr(m, "features_before_match", None)
        if f is None:
            continue

        comp = getattr(m, "comp_name", "<unknown>")
        consume(comp, getattr(f, "home_team_strength", None))
        consume(comp, getattr(f, "away_team_strength", None))

    for comp, rec in by_comp.items():
        rec["missing_cell_rate"] = rec["missing_cells"] / max(1, rec["total_cells"])
        rec["fully_missing_row_rate"] = rec["fully_missing_rows"] / max(1, rec["total_player_rows"])

    return {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_cell_rate": missing_cells / max(1, total_cells),
        "total_player_rows": total_player_rows,
        "fully_missing_rows": fully_missing_rows,
        "fully_missing_row_rate": fully_missing_rows / max(1, total_player_rows),
        "by_competition": dict(sorted(by_comp.items())),
    }


def train_variant(matches: list, strategy: str) -> dict:
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
        missing_skill_strategy=strategy,
        run_name=f"missing_skill_{strategy}_selected_mlp_binary_u25",
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
        "strategy": strategy,
        "run_name": cfg.run_name,
        "summary_path": str(summary_path),
        "round_metrics_path": str(round_metrics_path),
        "oos_predictions_path": str(oos_predictions_path),
        **summary,
    }


def main() -> None:
    load_common_state()

    matches = prepare_clean_matches()
    rounds = distribute_matches_into_rounds(matches)

    dataset_summary = {
        "num_matches": len(matches),
        "num_rounds": len(rounds),
        "missing_skill_summary": missing_skill_summary(matches),
    }

    with (OUT_ROOT / "missing_skill_dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    results = []
    for strategy in ["zero_mask", "position_mean"]:
        print("\n" + "=" * 80)
        print(f"[MISSING SKILL SENSITIVITY] {strategy}")
        print("=" * 80)
        results.append(train_variant(matches, strategy))

    results_csv = OUT_ROOT / "missing_skill_sensitivity_results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for r in results for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    results_json = OUT_ROOT / "missing_skill_sensitivity_results.json"
    with results_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ranking = sorted(
        results,
        key=lambda r: (
            -(r.get("pooled_auc") or -999),
            r.get("pooled_brier") or 999,
            -(r.get("pooled_accuracy") or -999),
        ),
    )

    with (OUT_ROOT / "missing_skill_sensitivity_ranking.txt").open("w", encoding="utf-8") as f:
        for i, r in enumerate(ranking, start=1):
            f.write(
                f"{i}. {r['run_name']} | "
                f"AUC={r.get('pooled_auc'):.6f} | "
                f"ACC={r.get('pooled_accuracy'):.6f} | "
                f"BRIER={r.get('pooled_brier'):.6f}\n"
            )

    print(f"[saved] {results_csv}")
    print(f"[saved] {results_json}")


if __name__ == "__main__":
    main()
