from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import (
    Global,
)
from football_outcomes.data.fs_models import (
    FSMatch,
)
from football_outcomes.data.fs_retrieve import (
    retrieve_new_data,
)
from football_outcomes.data.snapshots import (
    load_snapshot,
    save_snapshot,
)
from football_outcomes.data.sofifa_strength import (
    PastOnlyStrengthConfig,
)
from football_outcomes.data.sofifa_team_matching import (
    match_fs_teams_to_sofifa_teams,
)
from football_outcomes.data.state import (
    apply_bundle_to_global,
    bundle_from_global,
)
from football_outcomes.datasets.imputed_strength import (
    StrengthImputationContext,
)
from football_outcomes.datasets.mappings import (
    build_categorical_maps,
)
from football_outcomes.datasets.rounds import (
    distribute_matches_into_rounds,
)
from football_outcomes.training.train_mlp_rolling import (
    TrainConfig,
    train_rolling,
)
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as feat_utils

RoundMap = Mapping[
    tuple[str, int],
    set[int] | frozenset[int],
]


@dataclass(frozen=True)
class FootyStatsPipelineConfig:
    snapshot_path: Path
    allow_network: bool
    save_snapshot_path: Path | None

    competitions: tuple[str, ...]
    first_season: int
    last_season_exclusive: int
    excluded_competition_seasons: frozenset[tuple[str, int]]
    valid_round_ids_by_season: RoundMap

    validate_round_ids: bool
    rebuild_derived_state: bool
    enable_strength_imputation: bool

    log_dir: Path
    summary_path: Path
    run_name: str


def default_pipeline_config() -> FootyStatsPipelineConfig:
    """
    Resolve active runtime choices without a hidden
    submission-mode switch.
    """

    default_snapshot = sett.PROJECT_ROOT / "data" / "cache" / "footystats_snapshot.pkl"

    snapshot_path = Path(
        os.getenv(
            "FOP_LOAD_SNAPSHOT_PATH",
            default_snapshot,
        )
    )

    save_path_text = os.getenv("FOP_SAVE_SNAPSHOT_PATH")
    save_snapshot_path = Path(save_path_text) if save_path_text else None

    allow_network = (
        os.getenv(
            "FOP_ALLOW_NETWORK",
            "0",
        )
        == "1"
    )

    enable_strength_imputation = (
        os.getenv(
            ("FOP_ENABLE_" "STRENGTH_IMPUTATION"),
            "0",
        )
        == "1"
    )

    output_dir = sett.DATA_DIR / "pipeline_outputs"

    return FootyStatsPipelineConfig(
        snapshot_path=snapshot_path,
        allow_network=allow_network,
        save_snapshot_path=(save_snapshot_path),
        competitions=tuple(sett.COMPS_LEAGUE),
        first_season=sett.FIRST_SEASON,
        last_season_exclusive=(sett.LAST_SEASON),
        excluded_competition_seasons=(frozenset(sett.EXCLUDED_COMP_SEASONS)),
        valid_round_ids_by_season=(sett.LEAGUE_VALID_ROUND_IDS_BY_SEASON),
        validate_round_ids=(sett.VALIDATE_ROUND_IDS),
        rebuild_derived_state=False,
        log_dir=sett.LOG_DIR,
        summary_path=(output_dir / "dataset_summary.json"),
        run_name=("selected_mlp_binary_u25"),
        enable_strength_imputation=(enable_strength_imputation),
    )


def load_data_into_globals(
    config: FootyStatsPipelineConfig,
) -> None:
    """
    Load one explicit offline snapshot, or perform
    explicitly permitted network retrieval.
    """

    if config.snapshot_path.is_file():
        bundle = load_snapshot(config.snapshot_path)
    elif config.allow_network:
        bundle = retrieve_new_data()
    else:
        raise RuntimeError(
            "Snapshot not found and network "
            "retrieval is disabled: "
            f"{config.snapshot_path}. "
            "Set FOP_LOAD_SNAPSHOT_PATH or "
            "explicitly set "
            "FOP_ALLOW_NETWORK=1."
        )

    global_instance = apply_bundle_to_global(bundle)

    print("[data] matches loaded: " f"{len(global_instance.all_matches)}")
    print("[data] teams loaded: " f"{len(global_instance.all_teams)}")
    print("[data] players loaded: " f"{len(global_instance.all_players)}")
    print("[data] SoFIFA snapshots loaded: " f"{len(global_instance.sofifa_snapshots)}")


def select_clean_league_matches(
    matches: Sequence[FSMatch],
    config: FootyStatsPipelineConfig,
) -> list[FSMatch]:
    """
    Apply the active league, season, exclusion and
    round selection rules explicitly.
    """

    selected = []

    for match in matches:
        competition = getattr(
            match,
            "comp_name",
            None,
        )
        season = getattr(
            match,
            "season",
            None,
        )
        round_id = getattr(
            match,
            "round_id",
            None,
        )

        if competition not in (config.competitions):
            continue

        if season is None:
            continue

        season = int(season)

        if not (config.first_season <= season < config.last_season_exclusive):
            continue

        key = (
            competition,
            season,
        )

        if key in config.excluded_competition_seasons:
            continue

        valid_round_ids = config.valid_round_ids_by_season.get(key)

        if valid_round_ids is None:
            continue

        if round_id not in valid_round_ids:
            continue

        selected.append(match)

    return selected


def validate_round_selection(
    config: FootyStatsPipelineConfig,
) -> None:
    global_instance = Global.get_instance()

    present_keys = {
        (
            comp_season.name,
            int(comp_season.season),
        )
        for comp_season in global_instance.all_comp_seasons.values()
        if (
            comp_season.name in config.competitions
            and comp_season.season is not None
            and config.first_season <= int(comp_season.season) < config.last_season_exclusive
            and (
                comp_season.name,
                int(comp_season.season),
            )
            not in config.excluded_competition_seasons
        )
    }

    missing = sorted(present_keys - set(config.valid_round_ids_by_season))

    if missing:
        formatted = "\n".join(f"  {key}" for key in missing)
        raise ValueError("Round whitelist is incomplete:\n" f"{formatted}")


def log_feature_error(
    config: FootyStatsPipelineConfig,
    message: str,
) -> None:
    config.log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = config.log_dir / "feature_errors.log"

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(message.rstrip("\n") + "\n")


def selected_model_config(
    run_name: str,
    *,
    enable_strength_imputation: bool = False,
) -> TrainConfig:
    """
    Final selected v1-full scratch configuration used for the main binary U/O 2.5 model.
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
        enable_strength_imputation=(enable_strength_imputation),
    )


def prepare_matches(
    config: FootyStatsPipelineConfig,
) -> list[FSMatch]:
    g = Global.get_instance()

    utils.link_matches_to_comp_seasons()

    if config.validate_round_ids:
        validate_round_selection(config)

    utils.ensure_comp_season_dates(force=config.rebuild_derived_state)
    utils.initialize_league_tables(precompute_positions=True, force_rebuild=(config.rebuild_derived_state))

    match_fs_teams_to_sofifa_teams(force=False)

    all_matches_sorted = sorted(g.all_matches, key=feat_utils.match_sort_key)

    league_matches_sorted = select_clean_league_matches(
        all_matches_sorted,
        config,
    )

    print(f"[matches] usable league matches before feature calculation: {len(league_matches_sorted)}")

    team_index_all = feat_utils.build_team_match_index(all_matches_sorted)
    team_index_league = feat_utils.build_team_match_index(league_matches_sorted)

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
                config,
                f"[SKIP] match_id={match.id} {match.comp_name} {match.season} "
                f"{match.datetime} h={match.hour_utc} "
                f"{match.home_team.name} vs {match.away_team.name} "
                f"error={repr(e)}",
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


def write_dataset_summary(
    matches: Sequence[FSMatch],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rounds = distribute_matches_into_rounds(list(matches))

    summary = {
        "num_matches": len(matches),
        "num_rounds": len(rounds),
        "competitions": sorted({match.comp_name for match in matches}),
        "seasons": sorted({int(match.season) for match in matches if match.season is not None}),
        "matches_by_competition": {},
        "matches_by_season": {},
    }

    for match in matches:
        competition = match.comp_name
        season = str(match.season)

        summary["matches_by_competition"][competition] = (
            summary["matches_by_competition"].get(
                competition,
                0,
            )
            + 1
        )

        summary["matches_by_season"][season] = (
            summary["matches_by_season"].get(
                season,
                0,
            )
            + 1
        )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("[summary] saved dataset " f"summary to {path}")


def maybe_store_snapshot(
    config: FootyStatsPipelineConfig,
) -> None:
    if config.save_snapshot_path is None:
        return

    save_snapshot(
        bundle_from_global(),
        config.save_snapshot_path,
    )


def build_strength_imputation_context() -> StrengthImputationContext:
    global_instance = Global.get_instance()

    return StrengthImputationContext(
        snapshots=(global_instance.sofifa_snapshots),
        player_occurrences=(global_instance.sofifa_player_occurrences),
        fs_to_sofifa_cache=(global_instance.fs_to_sofifa_cache),
        reconstruction_config=(
            PastOnlyStrengthConfig(
                player_count=(sett.TEAM_STRENGTH_NUM_PLAYERS),
                skill_count=len(sett.PLAYER_SKILLS),
                max_age_days=(sett.SF_MAX_TIMEDELTA_DAYS),
                max_snapshots=(sett.SF_MAX_SNAPSHOTS_TO_SCAN),
            )
        ),
    )


def run_pipeline(
    config: FootyStatsPipelineConfig,
) -> None:
    print("=" * 80)
    print("[football-outcomes] " "FootyStats training pipeline")
    print("=" * 80)

    load_data_into_globals(config)

    matches = prepare_matches(config)

    write_dataset_summary(
        matches,
        config.summary_path,
    )

    category_maps = build_categorical_maps(
        matches=matches,
        competition_names=(config.competitions),
    )

    training_config = selected_model_config(
        run_name=config.run_name,
        enable_strength_imputation=(config.enable_strength_imputation),
    )

    strength_context = build_strength_imputation_context() if (training_config.enable_strength_imputation) else None

    train_rolling(
        matches_sorted=matches,
        cat_maps=category_maps,
        cfg=training_config,
        competition_names=(config.competitions),
        strength_imputation_context=(strength_context),
    )

    maybe_store_snapshot(config)

    print("[done] training completed")
