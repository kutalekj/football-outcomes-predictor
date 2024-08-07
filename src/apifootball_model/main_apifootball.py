import http.client
import json
import datetime
import features_utils as feature_ut
import settings
from comp import Comp
from season_comp_table import SeasonCompTable
from match import Match
from feature import MatchFeatures
from globals import Global


global_instance = Global.get_instance()

# Init comps and their seasons and rounds
for comp in settings.COMPS:
    new_comp = Comp(comp['id'], comp['name'])
    new_comp.init_all_rounds()

    global_instance.all_comps.append(new_comp)

# Init tables for comp seasons
for comp in global_instance.all_comps:
    for season in [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]:
        new_table = SeasonCompTable(comp.id, comp.name, season)
        new_table.init_teams_in_season_comp()

        global_instance.all_tables.append(new_table)

# Get matches
existing_matches = Match.load_existing_matches()
new_matches = Match.get_matches_data_using_api(global_instance.all_comps)

global_instance.all_matches = existing_matches + new_matches
# TODO: Save matches

print("breakpoint")
