import http.client
import json
import datetime
import features_utils as feature_ut
import utils as ut
import numpy as np
import settings
from comp import Comp
from season_comp_table import SeasonCompTable
from match import Match
from feature import MatchFeatures
from globals import Global
import in_out
from train import train

# import sys
# sys.stdout = open('C:\\Users\\kutalekj\\tmp_output.txt', 'w')

global_instance = Global.get_instance()

# 1. Init comps and their seasons and rounds
Comp.get_fs_leagues_list()

"""
for comp in [
    {'id': 144, 'name': "Jupiler Pro League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Conference League Play-off Group']},  # BEL
    {'id': 2, 'name': "UEFA Champions League", 'regular_round_keywords': []},
    {'id': 3, 'name': "UEFA Europa League", 'regular_round_keywords': []},
    {'id': 848, 'name': "UEFA Europa Conference League", 'regular_round_keywords': []},
    {'id': 147, 'name': "Cup", 'regular_round_keywords': []}  # BEL
]:
"""
for comp in [
    {'id': 94, 'name': "Primeira Liga", 'regular_round_keywords': ['Regular Season']},
    {'id': 96, 'name': "Taça de Portugal", 'regular_round_keywords': []},
    {'id': 97, 'name': "Taça da Liga", 'regular_round_keywords': []},
    {'id': 144, 'name': "Jupiler Pro League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Conference League Play-off Group']},  # BEL
    {'id': 2, 'name': "UEFA Champions League", 'regular_round_keywords': []},
    {'id': 3, 'name': "UEFA Europa League", 'regular_round_keywords': []},
    {'id': 848, 'name': "UEFA Europa Conference League", 'regular_round_keywords': []},
    {'id': 147, 'name': "Cup", 'regular_round_keywords': []}  # BEL
]:
# for comp in settings.COMPS_v2:
    new_comp = Comp(comp['id'], comp['name'], comp['regular_round_keywords'])
    print(f"Initializing comp [{new_comp.name}].")

    new_comp.init_teams_in_comp()
    new_comp.init_all_rounds()

    global_instance.all_comps.append(new_comp)
global_instance.all_teams = sorted(global_instance.all_teams, key=lambda team_: team_.id)

# 2. Init country start/end dates and tables for comp seasons
for comp in global_instance.all_comps:
    for season in [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]:

        # Country start/end dates
        if comp.country not in global_instance.start_end_dates_per_country_season:
            global_instance.start_end_dates_per_country_season[comp.country] = {}

        if season not in global_instance.start_end_dates_per_country_season[comp.country]:
            global_instance.start_end_dates_per_country_season[comp.country][season] = {
                'start': datetime.datetime.max, 'end': datetime.datetime.min}

        # Omit the cups - do not create tables for them
        if len(comp.regular_round_keywords) == 0:
            continue

        new_table = SeasonCompTable(comp.id, comp.name, season)
        print(f"Initializing table for comp [{new_table.comp_name}].")

        new_table.init_teams_in_season_comp()

        global_instance.all_tables.append(new_table)

# Country start/end dates
for comp in global_instance.all_comps:
    comp.init_country_start_end_dates_in_seasons()

# 3. Get matches (first existing locally saved, then new from API)
in_out.load_matches("tmp_csv_store10_BEL_POR.csv")
all_loaded_comp_seasons = list(set([(x.comp.id, x.season) for x in global_instance.all_matches]))
Match.get_new_matches_data_using_api(existing=all_loaded_comp_seasons)

global_instance.shots_on_goal = [x for x in global_instance.shots_on_goal if x != -1]
global_instance.total_shots = [x for x in global_instance.total_shots if x != -1]
global_instance.shots_inbox = [x for x in global_instance.shots_inbox if x != -1]
global_instance.corner_kicks = [x for x in global_instance.corner_kicks if x != -1]
global_instance.ball_possession = [x for x in global_instance.ball_possession if x != -1]
global_instance.pass_accuracy = [x for x in global_instance.pass_accuracy if x != -1]
print(f"Mean shots on goal = {np.mean(np.asarray(global_instance.shots_on_goal))}")
print(f"Variance shots on goal = {np.var(np.asarray(global_instance.shots_on_goal))}")
print(f"StdDev shots on goal = {np.std(np.asarray(global_instance.shots_on_goal))}")
print(f"Mean total shots = {np.mean(np.asarray(global_instance.total_shots))}")
print(f"Variance total shots = {np.var(np.asarray(global_instance.total_shots))}")
print(f"StdDev total shots = {np.std(np.asarray(global_instance.total_shots))}")
print(f"Mean shots inside box = {np.mean(np.asarray(global_instance.shots_inbox))}")
print(f"Variance shots inside box = {np.var(np.asarray(global_instance.shots_inbox))}")
print(f"StdDev shots inside box = {np.std(np.asarray(global_instance.shots_inbox))}")
print(f"Mean corner kicks = {np.mean(np.asarray(global_instance.corner_kicks))}")
print(f"Variance corner kicks = {np.var(np.asarray(global_instance.corner_kicks))}")
print(f"StdDev corner kicks = {np.std(np.asarray(global_instance.corner_kicks))}")
print(f"\nMean ball possession= {np.mean(np.asarray(global_instance.ball_possession))}")
print(f"Mean pass accuracy = {np.mean(np.asarray(global_instance.pass_accuracy))}")

# Sort matches by datetime played (asc.)
for team in global_instance.all_teams:
    team.matches = sorted(team.matches, key=lambda match_: match_.datetime)

    # Check team regularity (assume each team plays exactly in one regular comp each season!)
    team.correct_team_regularity_and_match_af_fs_teams()  # and match regular AF teams with FS teams !!!

# TODO: Add debug print check for number of matches (both all and just regular ones) for each comp season

# Exclude irregular teams from tables calculation
SeasonCompTable.exclude_irregular_teams_from_table_calculations()

# 4. Calculate features for each match (must be done chronologically asc.!)
global_instance.all_matches = sorted(global_instance.all_matches, key=lambda match_: match_.datetime)
for match in global_instance.all_matches:

    # DEBUG
    if match.home_team.name == "Genk" or match.away_team.name == "Genk":
        stop_here = True

    match.features_before_match_played = match.calculate_match_features()
    match.feature_vector_before_match_played = MatchFeatures.match_features_to_vector(
        match.features_before_match_played)

# sys.stdout.close()

# 5. Store matches
# in_out.store_matches("tmp_csv_store10_BEL_POR.csv")

"""

# 6. Distribute regular matches into rounds for training
regular_matches = [x for x in global_instance.all_matches if x.round.is_regular]
regular_matches = sorted(regular_matches, key=lambda match_: match_.datetime)

regular_matches_in_rounds = ut.distribute_matches_into_rounds(regular_matches)
for i, r in enumerate(regular_matches_in_rounds):
    print(f"{str(len(r))} matches found in round {str(i)}")
# TODO: Maybe ensure that there are at least N matches in each round?

# 7. Train
train(regular_matches_in_rounds)
print("breakpoint")

"""
