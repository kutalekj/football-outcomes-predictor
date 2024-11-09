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

global_instance = Global.get_instance()

"""
    {'id': 39, 'name': "Premier League", 'regular_round_keywords': ['Regular Season']},
    {'id': 40, 'name': "Championship", 'regular_round_keywords': ['Regular Season']},
    {'id': 41, 'name': "League One", 'regular_round_keywords': ['Regular Season']},
    {'id': 42, 'name': "League Two", 'regular_round_keywords': ['Regular Season']},
    {'id': 61, 'name': "Ligue 1", 'regular_round_keywords': ['Regular Season']},
    {'id': 62, 'name': "Ligue 2", 'regular_round_keywords': ['Regular Season']},
    {'id': 78, 'name': "Bundesliga", 'regular_round_keywords': ['Regular Season']},
    {'id': 79, 'name': "2. Bundesliga", 'regular_round_keywords': ['Regular Season']},
    {'id': 88, 'name': "Eredivisie", 'regular_round_keywords': ['Regular Season']},
    {'id': 94, 'name': "Primeira Liga", 'regular_round_keywords': ['Regular Season']},
    {'id': 106, 'name': "Ekstraklasa", 'regular_round_keywords': ['Regular Season']},  # POL
    {'id': 119, 'name': "Superliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round']},  # DEN
    {'id': 135, 'name': "Serie A", 'regular_round_keywords': ['Regular Season']},
    {'id': 136, 'name': "Serie B", 'regular_round_keywords': ['Regular Season']},
    {'id': 140, 'name': "La Liga", 'regular_round_keywords': ['Regular Season']},
    {'id': 141, 'name': "Segunda División", 'regular_round_keywords': ['Regular Season']},
    {'id': 144, 'name': "Jupiler Pro League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Conference League Play-off Group']},  # BEL
    {'id': 179, 'name': "Premiership",
     'regular_round_keywords': ['1st Phase', 'Championship Round', 'Relegation Round -']},  # SCO
    {'id': 188, 'name': "A-League",
     'regular_round_keywords': ['Regular Season', 'Elimination Finals', 'Semi-finals', 'Grand Final']},  # AUS
    {'id': 203, 'name': "Süper Lig", 'regular_round_keywords': ['Regular Season']},  # TUR
    {'id': 207, 'name': "Super League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -']},  # SUI
    {'id': 218, 'name': "Bundesliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -']},  # AUT
    {'id': 307, 'name': "Pro League", 'regular_round_keywords': ['Regular Season']},  # SA
    {'id': 323, 'name': "Super League",
     'regular_round_keywords': ['Regular Season', 'Qualifying Finals', 'Championship -']},  # IND
    {'id': 2, 'name': "UEFA Champions League", 'regular_round_keywords': []},
    {'id': 3, 'name': "UEFA Europa League", 'regular_round_keywords': []},
    {'id': 848, 'name': "UEFA Europa Conference League", 'regular_round_keywords': []},
    {'id': 45, 'name': "FA Cup", 'regular_round_keywords': []},
    {'id': 46, 'name': "EFL Trophy", 'regular_round_keywords': []},
    {'id': 81, 'name': "DFB Pokal", 'regular_round_keywords': []},
    {'id': 66, 'name': "Coupe de France", 'regular_round_keywords': []},
    {'id': 137, 'name': "Coppa Italia", 'regular_round_keywords': []},
    {'id': 143, 'name': "Copa del Rey", 'regular_round_keywords': []},
    {'id': 90, 'name': "KNVB Beker", 'regular_round_keywords': []},  # NED
    {'id': 96, 'name': "Taça de Portugal", 'regular_round_keywords': []},
    {'id': 97, 'name': "Taça da Liga", 'regular_round_keywords': []},
    {'id': 108, 'name': "Cup", 'regular_round_keywords': []},  # POL
    {'id': 209, 'name': "Schweizer Cup", 'regular_round_keywords': []},
    {'id': 206, 'name': "Cup", 'regular_round_keywords': []},  # TUR
    {'id': 121, 'name': "DBU Pokalen", 'regular_round_keywords': []},  # DEN
    {'id': 147, 'name': "Cup", 'regular_round_keywords': []},  # BEL
    {'id': 181, 'name': "FA Cup", 'regular_round_keywords': []},  # SCO
    {'id': 185, 'name': "League Cup", 'regular_round_keywords': []},  # SCO
    {'id': 220, 'name': "Cup", 'regular_round_keywords': []},  # AUT
    {'id': 504, 'name': "King's Cup", 'regular_round_keywords': []},  # SA
    {'id': 874, 'name': "Australia Cup", 'regular_round_keywords': []}  # AUS
"""

# 1. Init comps and their seasons and rounds
for comp in [
    {'id': 144, 'name': "Jupiler Pro League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Conference League Play-off Group']},  # BEL
    {'id': 2, 'name': "UEFA Champions League", 'regular_round_keywords': []},
    {'id': 3, 'name': "UEFA Europa League", 'regular_round_keywords': []},
    {'id': 848, 'name': "UEFA Europa Conference League", 'regular_round_keywords': []},
    {'id': 147, 'name': "Cup", 'regular_round_keywords': []}  # BEL
]:
    # for comp in settings.COMPS:
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
# in_out.load_matches("tmp_csv_store9_full.csv")
all_loaded_comp_seasons = list(set([(x.comp.id, x.season) for x in global_instance.all_matches]))
Match.get_new_matches_data_using_api(existing=all_loaded_comp_seasons)

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
    team.correct_team_regularity()

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

# 5. Store matches
in_out.store_matches("tmp_csv_store10_BEL.csv")

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
