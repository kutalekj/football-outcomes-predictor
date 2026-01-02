from __future__ import annotations

import football_outcomes.config.fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import save_snapshot, try_load_snapshot
from football_outcomes.data.fs_models import FSCompSeason, FSDataBundle, FSMatch, FSPlayer, FSTeam
from football_outcomes.data.fs_retrieve import fill_globals_with_cache, retrieve_new_data
from football_outcomes.utils import common as utils

globals()["FSDataBundle"] = FSDataBundle
globals()["FSCompSeason"] = FSCompSeason
globals()["FSTeam"] = FSTeam
globals()["FSPlayer"] = FSPlayer
globals()["FSMatch"] = FSMatch


ut = utils
global_instance = Global.get_instance()


cache = try_load_snapshot()
if sett.ALL_LOAD and cache is not None:
    fill_globals_with_cache(cache, update_leagues_list=False)

if sett.ALL_GET_NEW:
    bundle = retrieve_new_data()

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
