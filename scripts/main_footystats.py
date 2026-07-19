from __future__ import annotations

import json
from pathlib import Path

import matplotlib

import football_outcomes.config.fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import save_snapshot, try_load_snapshot
from football_outcomes.data.fs_retrieve import retrieve_new_data
from football_outcomes.data.state import (
    apply_bundle_to_global,
    bundle_from_global,
)
from football_outcomes.datasets.mappings import (
    build_categorical_maps,
)
from football_outcomes.datasets.rounds import (
    distribute_matches_into_rounds,
)
from football_outcomes.training.train_mlp_rolling import TrainConfig, train_rolling
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu
from football_outcomes.utils.fs_player_skill_utils import match_fs_teams_to_sofifa_teams

matplotlib.use("Agg")


def log_feature_error(msg: str) -> None:
    sett.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = sett.LOG_DIR / "feature_errors.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip("\n") + "\n")


def selected_model_config(run_name: str) -> TrainConfig:
    """
    Final selected v1-full scratch configuration used for the main binary U/O 2.5 model.
    In submission mode it is executed on the reduced EPL sample snapshot.
    """
    return TrainConfig(
        mode="binary_u25",
        model_version="v1",
        representation="full",
        use_strength_masks=True,
        use_position_embedding=True,
        use_team_strength=True,
        use_team_ids=True,
        use_comp_embedding=True,
        learning_rate=8e-5,
        lr_schedule="exponential",
        lr_decay_rate=0.997,
        min_learning_rate=2e-5,
        batch_size=64,
        window_rounds=25,
        epochs_per_step=2,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        team_emb_dim=8,
        comp_emb_dim=5,
        strength_emb_dim=24,
        position_emb_dim=3,
        mlp_hidden_1=128,
        mlp_hidden_2=64,
        mlp_hidden_3=32,
        mlp_dropout_1=0.30,
        mlp_dropout_2=0.20,
        seed=123,
        run_name=run_name,
        enable_branch_diagnostics=False,
        save_oos_predictions=True,
    )


def load_data_into_globals() -> None:
    g = Global.get_instance()

    cache = try_load_snapshot()
    if sett.ALL_LOAD and cache is not None:
        apply_bundle_to_global(cache)
    elif sett.ALL_GET_NEW:
        bundle = retrieve_new_data()
        apply_bundle_to_global(bundle)

    print(f"[data] matches loaded: {len(g.all_matches)}")
    print(f"[data] teams loaded: {len(g.all_teams)}")
    print(f"[data] players loaded: {len(g.all_players)}")
    print(f"[data] SOFIFA snapshots loaded: {len(getattr(g, 'sofifa_snapshots', []))}")


def prepare_matches():
    g = Global.get_instance()

    utils.link_matches_to_comp_seasons()

    if sett.VALIDATE_ROUND_IDS:
        utils.validate_league_valid_round_ids()

    utils.ensure_comp_season_dates(force=sett.ALL_GET_NEW)
    utils.initialize_league_tables(precompute_positions=True, force_rebuild=sett.ALL_GET_NEW)

    match_fs_teams_to_sofifa_teams(force=False)

    all_matches_sorted = sorted(g.all_matches, key=fu.match_sort_key)

    league_matches_sorted = utils.filter_clean_league_matches(all_matches_sorted)
    league_matches_sorted = utils.filter_valid_round_matches(league_matches_sorted)
    league_matches_sorted = [
        m
        for m in league_matches_sorted
        if getattr(m, "season", None) is not None and sett.FIRST_SEASON <= int(m.season) < sett.LAST_SEASON
    ]

    if getattr(sett, "SUBMISSION_MODE", False):
        league_matches_sorted = [
            m for m in league_matches_sorted if getattr(m, "comp_name", None) == "England Premier League"
        ]

    print(f"[matches] usable league matches before feature calculation: {len(league_matches_sorted)}")

    team_index_all = fu.build_team_match_index(all_matches_sorted)
    team_index_league = fu.build_team_match_index(league_matches_sorted)

    processed = 0
    skipped_matches = 0
    last_progress_month: tuple[int, int] | None = None

    for match in league_matches_sorted:
        processed += 1
        dt = match.datetime
        curr_month = (dt.year, dt.month)

        if curr_month != last_progress_month:
            last_progress_month = curr_month
            print(f"[features] {dt.year:04d}-{dt.month:02d}  (processed {processed}/{len(league_matches_sorted)})")

        try:
            match.features_before_match = match.calculate_match_features(
                team_index_league=team_index_league,
                team_index_all=team_index_all,
            )
        except ValueError as e:
            skipped_matches += 1
            log_feature_error(
                f"[SKIP] match_id={match.id} {match.comp_name} {match.season} "
                f"{match.datetime} h={match.hour_utc} "
                f"{match.home_team.name} vs {match.away_team.name} "
                f"error={repr(e)}"
            )

    print(f"[features] Done. Skipped matches: {skipped_matches}")

    league_matches_sorted = [
        m for m in league_matches_sorted if hasattr(m, "features_before_match") and m.features_before_match is not None
    ]

    print(f"[features] usable matches for training: {len(league_matches_sorted)}")

    if len(league_matches_sorted) == 0:
        raise RuntimeError("No matches with computed features are available for training.")

    num_with_strength = sum(
        1
        for m in league_matches_sorted
        if getattr(m.features_before_match, "home_team_strength", None) is not None
        and getattr(m.features_before_match, "away_team_strength", None) is not None
    )

    print(f"[features] matches with both team-strength tensors: {num_with_strength}/{len(league_matches_sorted)}")

    return league_matches_sorted


def write_dataset_summary(matches) -> None:
    out_dir = Path(sett.DATA_DIR) / "submission_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds = distribute_matches_into_rounds(matches)

    summary = {
        "submission_mode": bool(getattr(sett, "SUBMISSION_MODE", False)),
        "num_matches": len(matches),
        "num_rounds": len(rounds),
        "competitions": sorted({m.comp_name for m in matches}),
        "seasons": sorted({int(m.season) for m in matches if getattr(m, "season", None) is not None}),
        "matches_by_competition": {},
        "matches_by_season": {},
    }

    for m in matches:
        summary["matches_by_competition"][m.comp_name] = summary["matches_by_competition"].get(m.comp_name, 0) + 1
        summary["matches_by_season"][str(m.season)] = summary["matches_by_season"].get(str(m.season), 0) + 1

    path = out_dir / "submission_dataset_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[summary] saved dataset summary to {path}")


def maybe_store_snapshot() -> None:
    if not sett.ALL_STORE:
        return

    save_snapshot(bundle_from_global())


def main() -> None:
    print("=" * 80)
    print("[football-outcomes] submission/main pipeline")
    print("=" * 80)

    load_data_into_globals()
    league_matches_sorted = prepare_matches()
    write_dataset_summary(league_matches_sorted)

    cat_maps = build_categorical_maps(
        matches=league_matches_sorted,
        competition_names=(sett.COMPS_LEAGUE),
    )

    run_name = "submission_epl_selected_mlp_binary_u25"
    cfg = selected_model_config(run_name=run_name)

    print("=" * 80)
    print(f"[training] {run_name}")
    print("=" * 80)

    _ = train_rolling(
        matches_sorted=league_matches_sorted,
        cat_maps=cat_maps,
        cfg=cfg,
        competition_names=(sett.COMPS_LEAGUE),
    )

    print("[done] training completed")
    print(f"[summary] saved outputs under: {Path(sett.DATA_DIR) / 'tensorboard_logs' / run_name}")

    maybe_store_snapshot()


if __name__ == "__main__":
    main()
