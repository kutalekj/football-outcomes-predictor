from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

import football_outcomes.config.fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import load_avg_team_strength, load_sofifa_players, try_load_snapshot
from football_outcomes.data.fs_retrieve import fill_globals_with_cache, retrieve_new_data
from football_outcomes.training.fs_training_utils import build_categorical_maps
from football_outcomes.training.train_mlp_rolling import TrainConfig, train_rolling
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu
from football_outcomes.utils.fs_player_skill_utils import match_fs_teams_to_sofifa_teams

matplotlib.use("Agg")

OUT_ROOT = Path(sett.DATA_DIR) / "comparison" / "thesis_fig55_diagnostics"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def log_feature_error(msg: str) -> None:
    path = OUT_ROOT / "feature_errors_fig55.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip("\n") + "\n")


def prepare_matches():
    g = Global.get_instance()

    load_avg_team_strength()

    cache = try_load_snapshot()
    if sett.ALL_LOAD and cache is not None:
        fill_globals_with_cache(cache, update_leagues_list=False)

    if sett.ALL_GET_NEW:
        retrieve_new_data()

    load_sofifa_players(
        rebuild=getattr(sett, "REBUILD_SOFIFA_FROM_CSV", False),
        debug_shifts=False,
    )

    print("[sofifa] snapshots:", len(g.sofifa_snapshots))

    utils.link_matches_to_comp_seasons()

    if sett.VALIDATE_ROUND_IDS:
        utils.validate_league_valid_round_ids()

    utils.ensure_comp_season_dates(force=sett.ALL_GET_NEW)
    utils.initialize_league_tables(
        precompute_positions=True,
        force_rebuild=sett.ALL_GET_NEW,
    )

    match_fs_teams_to_sofifa_teams(force=False)

    all_matches_sorted = sorted(g.all_matches, key=fu.match_sort_key)

    league_matches_sorted = utils.filter_clean_league_matches(all_matches_sorted)
    league_matches_sorted = utils.filter_valid_round_matches(league_matches_sorted)
    league_matches_sorted = [
        m
        for m in league_matches_sorted
        if getattr(m, "season", None) is not None and sett.FIRST_SEASON <= int(m.season) < sett.LAST_SEASON
    ]

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
        except ValueError as e:
            skipped += 1
            log_feature_error(
                f"[SKIP] match_id={getattr(match, 'id', None)} "
                f"{getattr(match, 'comp_name', None)} {getattr(match, 'season', None)} "
                f"{getattr(match, 'datetime', None)} "
                f"{getattr(match.home_team, 'name', None)} vs {getattr(match.away_team, 'name', None)} "
                f"error={repr(e)}"
            )

    print(f"[features] usable={len(usable)} skipped={skipped}")
    return sorted(usable, key=fu.match_sort_key)


def build_fig55_config(run_name: str, model_version: str) -> TrainConfig:
    return TrainConfig(
        mode="binary_u25",
        window_rounds=25,
        epochs_per_step=4,
        learning_rate=1e-4,
        lr_schedule="constant",
        batch_size=64,
        team_emb_dim=8,
        comp_emb_dim=5,
        strength_emb_dim=16,
        position_emb_dim=3,
        max_goals_class=10,
        seed=42,
        model_version=model_version,
        use_team_aux_head=False,
        aux_task=None,
        aux_weight=0.15,
        num_branch_dim=48,
        cat_branch_dim=32,
        team_branch_dim=32,
        player_row_hidden_dim=32,
        role_post_hidden_dim=32,
        fusion_hidden_dim_1=64,
        fusion_hidden_dim_2=32,
        tabular_dropout=0.20,
        cat_dropout=0.15,
        team_dropout=0.25,
        fusion_dropout_1=0.45,
        fusion_dropout_2=0.30,
        num_l2=1e-5,
        cat_l2=1e-5,
        team_l2=5e-5,
        fusion_l2=5e-5,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        run_name=run_name,
        min_warning_val_size=20,
        save_oos_predictions=True,
        enable_branch_diagnostics=True,
        probe_matches=32,
        use_team_strength=True,
        use_team_ids=True,
        use_comp_embedding=True,
        use_position_embedding=True,
        representation="full",
        use_strength_masks=True,
    )


def main() -> None:
    matches = prepare_matches()
    cat_maps = build_categorical_maps(matches)

    runs = [
        {
            "label": "v1",
            "run_name": "thesis_fig55_diag_v1_full",
            "model_version": "v1",
        },
        {
            "label": "v2_lite",
            "run_name": "thesis_fig55_diag_v2_lite_full",
            "model_version": "v2",
        },
    ]

    results = []

    for r in runs:
        print("\n" + "=" * 80)
        print(f"[THESIS FIG 5.5 DIAGNOSTICS] {r['run_name']}")
        print("=" * 80)

        cfg = build_fig55_config(
            run_name=r["run_name"],
            model_version=r["model_version"],
        )

        train_rolling(matches, cat_maps, cfg)

        log_dir = Path(sett.DATA_DIR) / "tensorboard_logs" / cfg.run_name

        with (log_dir / "summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)

        result = {
            "label": r["label"],
            "run_name": cfg.run_name,
            "model_version": cfg.model_version,
            "summary_path": str(log_dir / "summary.json"),
            "round_metrics_path": str(log_dir / "round_metrics.csv"),
            "oos_predictions_path": str(log_dir / "oos_predictions.csv"),
            "diagnostics_path": str(log_dir / "diagnostics.csv"),
            **summary,
        }
        results.append(result)

    out_json = OUT_ROOT / "thesis_fig55_diagnostics_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_csv = OUT_ROOT / "thesis_fig55_diagnostics_results.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for row in results for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[saved] {out_json}")
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
