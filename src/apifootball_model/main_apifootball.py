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
global_instance.all_teams = sorted(global_instance.all_teams, key=lambda team_: team_.id)

# Init tables for comp seasons
# TODO: Include also matches that do not belong to the predefined competitions - home cups, not UCL/UEL/UECL !!!
# TODO: Omit the table position information which is almost meaningless for them - set -1 for this feature
# TODO: Add-note...Or, if exclude EU cups for biased table pos/form calculation, maybe no need for -1
for comp in global_instance.all_comps:
    for season in [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]:
        new_table = SeasonCompTable(comp.id, comp.name, season)
        print(f"Initializing table for comp [{new_table.comp_name}].")

        new_table.init_teams_in_season_comp()

        global_instance.all_tables.append(new_table)

# Get matches
global_instance.all_matches = Match.load_existing_matches()
Match.get_new_matches_data_using_api()

# Sort matches by datetime played (asc.)
for team in global_instance.all_teams:
    team.matches = sorted(team.matches, key=lambda match_: match_.datetime)

# Calculate features
for team in global_instance.all_teams:
    for match in team.matches:
        match.features_before_match_played = match.calculate_match_features()
        match.feature_vector_before_match_played = MatchFeatures.match_features_to_vector(
            match.features_before_match_played)

# TODO: Save matches

print("breakpoint")
