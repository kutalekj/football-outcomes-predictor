from __future__ import annotations

import football_outcomes.config.fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import load_avg_team_strength, load_sofifa_players, save_snapshot, try_load_snapshot
from football_outcomes.data.fs_models import FSDataBundle
from football_outcomes.data.fs_retrieve import fill_globals_with_cache, retrieve_new_data
from football_outcomes.utils import fs_common as utils

ut = utils
global_instance = Global.get_instance()

load_avg_team_strength()

cache = try_load_snapshot()
if sett.ALL_LOAD and cache is not None:
    fill_globals_with_cache(cache, update_leagues_list=False)
    ut.ensure_comp_season_dates(force=False)
    ut.initialize_league_tables(precompute_positions=True, force_rebuild=False)

if sett.ALL_GET_NEW:
    bundle = retrieve_new_data()
    ut.ensure_comp_season_dates(force=True)
    ut.initialize_league_tables(precompute_positions=True, force_rebuild=True)

load_sofifa_players()

if sett.ALL_STORE:
    save_snapshot(
        FSDataBundle(
            global_instance.all_comp_seasons,
            global_instance.all_teams,
            global_instance.all_players,
            global_instance.all_matches,
            global_instance.leagues_list,
        )
    )
