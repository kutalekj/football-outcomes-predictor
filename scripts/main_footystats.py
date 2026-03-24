from __future__ import annotations

import os

import football_outcomes.config.fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import load_avg_team_strength, load_sofifa_players, save_snapshot, try_load_snapshot
from football_outcomes.data.fs_models import FSDataBundle
from football_outcomes.data.fs_retrieve import fill_globals_with_cache, retrieve_new_data
from football_outcomes.training.fs_training_utils import build_categorical_maps
from football_outcomes.training.train_mlp_rolling import TrainConfig, train_rolling
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu
from football_outcomes.utils.fs_player_skill_utils import match_fs_teams_to_sofifa_teams


def log_feature_error(msg: str) -> None:
    os.makedirs(sett.LOG_DIR, exist_ok=True)
    path = os.path.join("logs", "feature_errors.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip("\n") + "\n")


def debug_count_entities(matches, label: str) -> None:
    team_ids = set()
    player_ids = set()

    for m in matches:
        if getattr(m, "home_team", None) is not None:
            team_ids.add(m.home_team.id)
        if getattr(m, "away_team", None) is not None:
            team_ids.add(m.away_team.id)

        for p in getattr(m, "home_lineup", []) or []:
            if p is not None:
                player_ids.add(p.id)
        for p in getattr(m, "away_lineup", []) or []:
            if p is not None:
                player_ids.add(p.id)

    print(f"[{label}] matches={len(matches)}, " f"teams={len(team_ids)}, " f"players_in_lineups={len(player_ids)}")


def debug_sofifa_snapshot_player_coverage(matches) -> None:
    g = Global.get_instance()

    # Step 1: collect FS player IDs from matches
    fs_player_ids = set()
    for m in matches:
        for p in getattr(m, "home_lineup", []) or []:
            if p is not None:
                fs_player_ids.add(p.id)
        for p in getattr(m, "away_lineup", []) or []:
            if p is not None:
                fs_player_ids.add(p.id)

    print(f"[sofifa] unique FS players in matches: {len(fs_player_ids)}")

    # Step 2: map to SOFIFA player IDs
    sofifa_ids = set()
    unmapped = 0

    for fs_id in fs_player_ids:
        match = g.fs_to_sofifa_cache.get(fs_id)
        if match is None or match[0] is None:
            unmapped += 1
            continue
        sofifa_ids.add(match[0])

    print(f"[sofifa] mapped to SOFIFA players: {len(sofifa_ids)}")
    print(f"[sofifa] unmapped FS players: {unmapped}")

    # Step 3: count per snapshot
    print("\n[sofifa] approximate coverage per snapshot:")

    for snap_date, players_dict in g.sofifa_snapshots:
        snapshot_player_ids = set(players_dict.keys())

        overlap = len(snapshot_player_ids & sofifa_ids)

        print(f"{snap_date}: " f"{overlap} relevant players " f"(~{overlap/1000:.1f}k)")


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

league_matches_all_24 = [
    m
    for m in all_matches_sorted
    if getattr(m, "comp_name", None) in sett.COMPS_LEAGUE
    and getattr(m, "season", None) is not None
    and sett.FIRST_SEASON <= m.season < sett.LAST_SEASON
]

debug_count_entities(league_matches_all_24, "24 domestic leagues before filtering")
debug_sofifa_snapshot_player_coverage(league_matches_all_24)

league_matches_sorted = utils.filter_clean_league_matches(all_matches_sorted)
league_matches_sorted = utils.filter_valid_round_matches(league_matches_sorted)
league_matches_sorted = [
    m
    for m in league_matches_sorted
    if getattr(m, "season", None) is not None and sett.FIRST_SEASON <= m.season < sett.LAST_SEASON
]

debug_count_entities(league_matches_sorted, "24 domestic leagues after filtering")

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

cfg = TrainConfig(mode="binary_u25", window_rounds=25, epochs_per_step=5, batch_size=64, seed=42)

model = train_rolling(league_matches_sorted, cat_maps, cfg)
model.save("mlp_first_run.keras")
print("Model saved.")

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
