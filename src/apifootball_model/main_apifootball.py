import http.client
import json
import datetime
import features_utils as feature_ut
import utils as ut
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import settings
from comp import Comp
from season_comp_table import SeasonCompTable
from match import Match
from feature import MatchFeatures
from globals import Global
import in_out

global_instance = Global.get_instance()

# 1. Init comps and their seasons and rounds
for comp in [{'id': 144, 'name': "Jupiler Pro League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Conference League Play-off Group']},
             {'id': 3, 'name': "UEFA Europa League", 'regular_round_keywords': []}]:
# for comp in settings.COMPS:
    new_comp = Comp(comp['id'], comp['name'], comp['regular_round_keywords'])
    print(f"Initializing comp [{new_comp.name}].")

    new_comp.init_teams_in_comp()
    new_comp.init_all_rounds()

    global_instance.all_comps.append(new_comp)
global_instance.all_teams = sorted(global_instance.all_teams, key=lambda team_: team_.id)

# Init one-ht encoder for comps
comp_ids = [[comp.id] for comp in global_instance.all_comps]
dummy_comp_ids = list(range(-1, -1 - (settings.ONE_HOT_ENCODED_VECTOR_LENGTH - len(comp_ids)), -1))
dummy_comp_ids = [[x] for x in dummy_comp_ids]  # Dummy IDs ensure unity of total lengths of one-hot encoded vectors

global_instance.one_hot_encoder_comps = OneHotEncoder(sparse_output=False)
global_instance.one_hot_encoder_comps.fit(np.array(comp_ids + dummy_comp_ids))

# 2. Init tables for comp seasons
for comp in global_instance.all_comps:
    # Omit the cups - do not create tables for them
    if len(comp.regular_round_keywords) == 0:
        continue

    for season in [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]:
        new_table = SeasonCompTable(comp.id, comp.name, season)
        print(f"Initializing table for comp [{new_table.comp_name}].")

        new_table.init_teams_in_season_comp()

        global_instance.all_tables.append(new_table)

# 3. Get matches (first existing locally saved, then new from API)
in_out.load_matches("tmp_csv_store5.csv")
all_loaded_comp_seasons = list(set([(x.comp.id, x.season) for x in global_instance.all_matches]))
Match.get_new_matches_data_using_api(existing=all_loaded_comp_seasons)

# Sort matches by datetime played (asc.)
for team in global_instance.all_teams:
    team.matches = sorted(team.matches, key=lambda match_: match_.datetime)

    # Check team regularity (assume each team plays exactly in one regular comp each season!)
    team.correct_team_regularity()

# Exclude irregular teams from tables calculation
SeasonCompTable.exclude_irregular_teams_from_table_calculations()

# Init one-hot encoder for tables (must be done after excluding irregular teams)
for table in global_instance.all_tables:
    team_ids = [[team.id] for team in table.teams]
    dummy_team_ids = list(range(-1, -1 - (settings.ONE_HOT_ENCODED_VECTOR_LENGTH - len(team_ids)), -1))
    dummy_team_ids = [[x] for x in dummy_team_ids]  # Dummy IDs ensure unity of total lengths of one-hot encoded vectors
    print(f"All team IDs for table {table.comp_name} {str(table.season)}: {team_ids + dummy_team_ids}")

    table.one_hot_encoder = OneHotEncoder(sparse_output=False)
    table.one_hot_encoder.fit(team_ids + dummy_team_ids)

# 4. Calculate features for each match (must be done chronologically asc.!)
global_instance.all_matches = sorted(global_instance.all_matches, key=lambda match_: match_.datetime)
for match in global_instance.all_matches:

    # DEBUG
    if match.home_team.name == "Genk" or match.away_team.name == "Genk":
        stop_here = True

    match.features_before_match_played = match.calculate_match_features()
    match.feature_vector_before_match_played = MatchFeatures.match_features_to_vector(
        match.features_before_match_played)

# 5. Store matches
in_out.store_matches("tmp_csv_store5_LS.csv")
print("breakpoint")
