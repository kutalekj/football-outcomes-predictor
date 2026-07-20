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
from football_outcomes.training.fs_training_utils import build_categorical_maps
from football_outcomes.training.train_mlp_rolling import TrainConfig, train_rolling
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu

matplotlib.use("Agg")

OUT_ROOT = Path(sett.DATA_DIR) / "comparison" / "thesis_fig51_diagnostics"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def log_feature_error(msg: str) -> None:
    os.makedirs(sett.LOG_DIR, exist_ok=True)
    path = OUT_ROOT / "feature_errors_fig51.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip("\n") + "\n")


def prepare_matches():
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

    if sett.VALIDATE_ROUND_IDS:
        utils.validate_league_valid_round_ids()

    utils.ensure_comp_season_dates(force=False)
    utils.initialize_league_tables(
        precompute_positions=True,
        force_rebuild=False,
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
                f"[SKIP] match_id={match.id} {match.comp_name} {match.season} "
                f"{match.datetime} h={match.hour_utc} "
                f"{match.home_team.name} vs {match.away_team.name} "
                f"error={repr(e)}"
            )

    print(f"[features] usable={len(usable)} skipped={skipped}")
    return sorted(usable, key=fu.match_sort_key)


def main() -> None:
    matches = prepare_matches()
    cat_maps = build_categorical_maps(matches)

    runs = [
        {
            "label": "full",
            "run_name": "thesis_fig51_diag_full",
            "use_team_strength": True,
            "use_position_embedding": True,
        },
        {
            "label": "no_strength",
            "run_name": "thesis_fig51_diag_no_strength",
            "use_team_strength": False,
            "use_position_embedding": False,
        },
        {
            "label": "no_positions",
            "run_name": "thesis_fig51_diag_no_positions",
            "use_team_strength": True,
            "use_position_embedding": False,
        },
    ]

    results = []

    for r in runs:
        print("\n" + "=" * 80)
        print(f"[THESIS FIG 5.1 DIAGNOSTICS] {r['run_name']}")
        print("=" * 80)

        cfg = TrainConfig(
            mode="binary_u25",
            model_version="v1",
            run_name=r["run_name"],
            # Original diagnostic setting approximation.
            window_rounds=25,
            epochs_per_step=5,
            learning_rate=1e-4,
            lr_schedule="constant",
            batch_size=64,
            seed=42,
            team_emb_dim=8,
            comp_emb_dim=5,
            strength_emb_dim=24,
            position_emb_dim=3,
            mlp_hidden_1=128,
            mlp_hidden_2=64,
            mlp_hidden_3=32,
            mlp_dropout_1=0.50,
            mlp_dropout_2=0.40,
            use_team_strength=r["use_team_strength"],
            use_team_ids=True,
            use_comp_embedding=True,
            use_position_embedding=r["use_position_embedding"],
            use_strength_masks=True,
            representation="full",
            enable_branch_diagnostics=True,
            probe_matches=32,
            save_oos_predictions=True,
            min_warning_val_size=20,
        )

        train_rolling(matches, cat_maps, cfg)

        log_dir = Path(sett.DATA_DIR) / "tensorboard_logs" / cfg.run_name
        with (log_dir / "summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)

        results.append(
            {
                "label": r["label"],
                "run_name": cfg.run_name,
                "summary_path": str(log_dir / "summary.json"),
                "round_metrics_path": str(log_dir / "round_metrics.csv"),
                "oos_predictions_path": str(log_dir / "oos_predictions.csv"),
                "diagnostics_path": str(log_dir / "diagnostics.csv"),
                **summary,
            }
        )

    out_json = OUT_ROOT / "thesis_fig51_diagnostics_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_csv = OUT_ROOT / "thesis_fig51_diagnostics_results.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for row in results for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[saved] {out_json}")
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
