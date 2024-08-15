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
for comp in [{'id': 144, 'name': "Jupiler Pro League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round',
                                'Conference League Play-off Group']}]:
# for comp in settings.COMPS:  # TODO: Temporary
    new_comp = Comp(comp['id'], comp['name'], comp['regular_round_keywords'])
    print(f"Initializing comp [{new_comp.name}].")

    new_comp.init_teams_in_comp()
    new_comp.init_all_rounds()

    global_instance.all_comps.append(new_comp)
global_instance.all_teams = sorted(global_instance.all_teams, key=lambda team: team.id)

# Init tables for comp seasons
for comp in global_instance.all_comps:
    for season in [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]:
        new_table = SeasonCompTable(comp.id, comp.name, season)
        print(f"Initializing table for comp [{new_table.comp_name}].")

        new_table.init_teams_in_season_comp()

        global_instance.all_tables.append(new_table)

# Get matches
global_instance.all_matches = Match.load_existing_matches()
Match.get_new_matches_data_using_api()

# TODO: Save matches

print("breakpoint")
