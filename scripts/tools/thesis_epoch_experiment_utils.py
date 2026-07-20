from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import tensorflow as tf

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
from football_outcomes.training.fs_training_utils import (
    extract_numerical_features,
)
from football_outcomes.training.train_mlp_rolling import (
    TrainConfig,
    build_model,
    compile_model_for_cfg,
    transfer_pretrained_strength_branch_weights,
)
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu


def prepare_clean_matches():
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
            print(
                f"[SKIP] match_id={getattr(match, 'id', None)} "
                f"{getattr(match, 'comp_name', None)} "
                f"{getattr(match, 'season', None)} "
                f"error={repr(e)}"
            )

    print(f"[features] usable={len(usable)} skipped={skipped}")
    return sorted(usable, key=fu.match_sort_key)


def collect_run_summary(run_name: str, label: str) -> dict[str, Any]:
    log_dir = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name

    summary_path = log_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

    return {
        "label": label,
        "run_name": run_name,
        "log_dir": str(log_dir),
        "epoch_metrics_path": str(log_dir / "epoch_metrics.csv"),
        "round_metrics_path": str(log_dir / "round_metrics.csv"),
        "oos_predictions_path": str(log_dir / "oos_predictions.csv"),
        "summary_path": str(summary_path),
        **summary,
    }


def write_summary_files(out_root: Path, rows: list[dict[str, Any]], stem: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    with (out_root / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if rows:
        with (out_root / f"{stem}.csv").open("w", encoding="utf-8", newline="") as f:
            fieldnames = sorted({k for row in rows for k in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def build_base_v1_cfg(run_name: str) -> TrainConfig:
    return TrainConfig(
        mode="binary_u25",
        model_version="v1",
        run_name=run_name,
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
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        use_team_strength=True,
        use_team_ids=True,
        use_comp_embedding=True,
        use_position_embedding=True,
        use_strength_masks=True,
        representation="full",
        enable_branch_diagnostics=False,
        save_oos_predictions=True,
        min_warning_val_size=20,
    )


def build_fig54_v1_or_v2lite_cfg(run_name: str, model_version: str) -> TrainConfig:
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
        enable_branch_diagnostics=False,
        probe_matches=32,
        use_team_strength=True,
        use_team_ids=True,
        use_comp_embedding=True,
        use_position_embedding=True,
        representation="full",
        use_strength_masks=True,
    )


def build_fig54_v2_full_approx_cfg(run_name: str) -> TrainConfig:
    """
    Approximation of the earlier v2-full setting for slide 15.

    The exact original v2-full train_config was not preserved. This keeps
    the same rolling/training dynamics as the v1/v2-lite diagnostic comparison
    but uses a wider and less regularized v2 branch, so the resulting curves
    should reproduce the intended qualitative behavior: stronger fitting and
    worse generalization than v1.
    """
    return TrainConfig(
        mode="binary_u25",
        window_rounds=25,
        epochs_per_step=4,
        learning_rate=1e-4,
        lr_schedule="constant",
        batch_size=64,
        team_emb_dim=8,
        comp_emb_dim=5,
        strength_emb_dim=32,
        position_emb_dim=3,
        max_goals_class=10,
        seed=42,
        model_version="v2",
        use_team_aux_head=False,
        aux_task=None,
        aux_weight=0.15,
        num_branch_dim=64,
        cat_branch_dim=64,
        team_branch_dim=64,
        player_row_hidden_dim=64,
        role_post_hidden_dim=64,
        fusion_hidden_dim_1=128,
        fusion_hidden_dim_2=64,
        tabular_dropout=0.10,
        cat_dropout=0.05,
        team_dropout=0.10,
        fusion_dropout_1=0.30,
        fusion_dropout_2=0.20,
        num_l2=0.0,
        cat_l2=0.0,
        team_l2=0.0,
        fusion_l2=0.0,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        run_name=run_name,
        min_warning_val_size=20,
        save_oos_predictions=True,
        enable_branch_diagnostics=False,
        probe_matches=32,
        use_team_strength=True,
        use_team_ids=True,
        use_comp_embedding=True,
        use_position_embedding=True,
        representation="full",
        use_strength_masks=True,
    )


def build_fig56_v1_cfg(run_name: str, freeze_rounds: int) -> TrainConfig:
    return TrainConfig(
        mode="binary_u25",
        window_rounds=25,
        epochs_per_step=3,
        learning_rate=8e-5,
        lr_schedule="constant",
        batch_size=64,
        team_emb_dim=8,
        comp_emb_dim=5,
        strength_emb_dim=16,
        position_emb_dim=3,
        max_goals_class=10,
        seed=42,
        model_version="v1",
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
        freeze_pretrained_branch_rounds=freeze_rounds,
        run_name=run_name,
        min_warning_val_size=20,
        save_oos_predictions=True,
        enable_branch_diagnostics=False,
        probe_matches=32,
        use_team_strength=True,
        use_team_ids=True,
        use_comp_embedding=True,
        use_position_embedding=True,
        representation="full",
        use_strength_masks=True,
    )


def build_v1_model_with_optional_pretraining(matches, cat_maps, cfg: TrainConfig, pretrained_path: Path | None):
    if pretrained_path is None:
        return None

    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Pretrained branch model not found: {pretrained_path}\n"
            "Either run the standalone v1 branch pretraining first, or set PRETRAINED_V1_BRANCH_PATH."
        )

    sample_feat = matches[0].features_before_match
    num_num = extract_numerical_features(sample_feat).shape[0]

    full_model = build_model(
        num_num=num_num,
        num_teams=len(cat_maps.team_id_map),
        num_comps=len(cat_maps.comp_id_map),
        cfg=cfg,
    )

    pretrained_model = tf.keras.models.load_model(pretrained_path)
    transfer_pretrained_strength_branch_weights(
        pretrained_model=pretrained_model,
        full_model=full_model,
        branch_version="v1",
    )
    compile_model_for_cfg(full_model, cfg)
    return full_model
