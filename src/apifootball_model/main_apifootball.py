import http.client
import json
import datetime
import features_utils as feature_ut
import utils as ut
import settings
from comp import Comp
from season_comp_table import SeasonCompTable
from match import Match
from feature import MatchFeatures
from globals import Global

global_instance = Global.get_instance()

# Init comps and their seasons and rounds
for comp in [{'id': 144, 'name': "Jupiler Pro League", 'regular_round_keywords': ['Regular Season', 'Championship Round', 'Conference League Play-off Group']}]:  # TODO: Temporary
# for comp in settings.COMPS:
    new_comp = Comp(comp['id'], comp['name'], comp['regular_round_keywords'])
    print(f"Initializing comp [{new_comp.name}].")

    new_comp.init_teams_in_comp()
    new_comp.init_all_rounds()

    global_instance.all_comps.append(new_comp)
global_instance.all_teams = sorted(global_instance.all_teams, key=lambda team_: team_.id)

# Init tables for comp seasons
for comp in global_instance.all_comps:
    # Omit the cups - do not create tables for them
    if len(comp.regular_round_keywords) == 0:
        continue

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

    # Check team regularity (assume each team plays exactly in one regular comp each season!)
    team.correct_team_regularity()

# Exclude irregular teams from tables calculation
SeasonCompTable.exclude_irregular_teams_from_table_calculations()

# TODO: Add missing statistics by averaging the existing ones

# Calculate features for each match (must be done chronologically asc.!)
for match in global_instance.all_matches:
    if match.home_team_name == "Genk" or match.away_team_name == "Genk":
        stop_here = True

    match.features_before_match_played = match.calculate_match_features()
    match.feature_vector_before_match_played = MatchFeatures.match_features_to_vector(
        match.features_before_match_played)

# TODO: Save matches...Add-note: Maybe do this already after loading match data, before adding missing statistics?


print("breakpoint")
