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
"""
"""
for comp in [
    {'id': 39, 'name': "Premier League", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Premier League"},
    {'id': 40, 'name': "Championship", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Championship"},
    {'id': 41, 'name': "League One", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "EFL League One"},
    {'id': 42, 'name': "League Two", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "EFL League Two"},
    {'id': 61, 'name': "Ligue 1", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ligue 1"},
    {'id': 62, 'name': "Ligue 2", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ligue 2"},
    {'id': 78, 'name': "Bundesliga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Bundesliga"},
    {'id': 79, 'name': "2. Bundesliga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "2. Bundesliga"},
    {'id': 88, 'name': "Eredivisie", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Eredivisie"},
    {'id': 94, 'name': "Primeira Liga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Liga NOS"},
    {'id': 106, 'name': "Ekstraklasa", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ekstraklasa"},  # POL
    {'id': 119, 'name': "Superliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round'],
     'fs_alias': "Superliga"},  # DEN
    {'id': 135, 'name': "Serie A", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Serie A"},
    {'id': 136, 'name': "Serie B", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Serie B"},
    {'id': 140, 'name': "La Liga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "La Liga"},
    {'id': 141, 'name': "Segunda División", 'regular_round_keywords': ['Regular Season'],
     'fs_alias': "Segunda División"},
    {'id': 144, 'name': "Jupiler Pro League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Conference League Play-off Group'],
     'fs_alias': "Pro League"},  # BEL
    {'id': 179, 'name': "Premiership",
     'regular_round_keywords': ['1st Phase', 'Championship Round', 'Relegation Round -'],
     'fs_alias': "Premiership"},  # SCO
    {'id': 188, 'name': "A-League",
     'regular_round_keywords': ['Regular Season', 'Elimination Finals', 'Semi-finals', 'Grand Final'],
     'fs_alias': "A-League"},  # AUS
    {'id': 203, 'name': "Süper Lig", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Süper Lig"},  # TUR
    {'id': 207, 'name': "Super League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -'],
     'fs_alias': "Super League"},  # SUI
    {'id': 218, 'name': "Bundesliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -'],
     'fs_alias': "Bundesliga"},  # AUT
    {'id': 307, 'name': "Pro League", 'regular_round_keywords': ['Regular Season'],
     'fs_alias': "Professional League"},  # SA
    {'id': 323, 'name': "Indian Super League",
     'regular_round_keywords': ['Regular Season', 'Qualifying Finals', 'Championship -'],
     'fs_alias': "Indian Super League"},  # IND
    {'id': 96, 'name': "Taça de Portugal", 'regular_round_keywords': []},
    {'id': 97, 'name': "Taça da Liga", 'regular_round_keywords': []},
    {'id': 2, 'name': "UEFA Champions League", 'regular_round_keywords': []},
    {'id': 3, 'name': "UEFA Europa League", 'regular_round_keywords': []},
    {'id': 848, 'name': "UEFA Europa Conference League", 'regular_round_keywords': []},
    {'id': 147, 'name': "Cup", 'regular_round_keywords': []}  # BEL
]:
"""
"""
for comp in [
    {'id': 61, 'name': "Ligue 1", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ligue 1"},
    {'id': 62, 'name': "Ligue 2", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ligue 2"},
    {'id': 66, 'name': "Coupe de France", 'regular_round_keywords': []},
    {'id': 179, 'name': "Premiership",
     'regular_round_keywords': ['1st Phase', 'Championship Round', 'Relegation Round -'],
     'fs_alias': "Premiership"},  # SCO
    {'id': 181, 'name': "FA Cup", 'regular_round_keywords': []},  # SCO
    {'id': 185, 'name': "League Cup", 'regular_round_keywords': []},  # SCO
    {'id': 119, 'name': "Superliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round'],
     'fs_alias': "Superliga"},  # DEN
    {'id': 121, 'name': "DBU Pokalen", 'regular_round_keywords': []},  # DEN
    {'id': 88, 'name': "Eredivisie", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Eredivisie"},
    {'id': 90, 'name': "KNVB Beker", 'regular_round_keywords': []},  # NED
    {'id': 307, 'name': "Pro League", 'regular_round_keywords': ['Regular Season'],
     'fs_alias': "Professional League"},  # SA
    {'id': 504, 'name': "King's Cup", 'regular_round_keywords': []},  # SA
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
"""
"""
{'id': 39, 'name': "Premier League", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Premier League"},
    {'id': 40, 'name': "Championship", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Championship"},
    {'id': 41, 'name': "League One", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "EFL League One"},
    {'id': 42, 'name': "League Two", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "EFL League Two"},
    {'id': 61, 'name': "Ligue 1", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ligue 1"},
    {'id': 62, 'name': "Ligue 2", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ligue 2"},
    {'id': 45, 'name': "FA Cup", 'regular_round_keywords': []},
    {'id': 46, 'name': "EFL Trophy", 'regular_round_keywords': []},
    {'id': 66, 'name': "Coupe de France", 'regular_round_keywords': []},
    {'id': 179, 'name': "Premiership",
     'regular_round_keywords': ['1st Phase', 'Championship Round', 'Relegation Round -'],
     'fs_alias': "Premiership"},  # SCO
    {'id': 181, 'name': "FA Cup", 'regular_round_keywords': []},  # SCO
    {'id': 185, 'name': "League Cup", 'regular_round_keywords': []},  # SCO
    {'id': 119, 'name': "Superliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round'],
     'fs_alias': "Superliga"},  # DEN
    {'id': 121, 'name': "DBU Pokalen", 'regular_round_keywords': []},  # DEN
    {'id': 88, 'name': "Eredivisie", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Eredivisie"},
    {'id': 90, 'name': "KNVB Beker", 'regular_round_keywords': []},  # NED
    {'id': 307, 'name': "Pro League", 'regular_round_keywords': ['Regular Season'],
     'fs_alias': "Professional League"},  # SA
    {'id': 504, 'name': "King's Cup", 'regular_round_keywords': []},  # SA
    {'id': 94, 'name': "Primeira Liga", 'regular_round_keywords': ['Regular Season']},
    {'id': 96, 'name': "Taça de Portugal", 'regular_round_keywords': []},
    {'id': 97, 'name': "Taça da Liga", 'regular_round_keywords': []},
    {'id': 144, 'name': "Jupiler Pro League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Conference League Play-off Group']},  # BEL
    {'id': 2, 'name': "UEFA Champions League", 'regular_round_keywords': []},
    {'id': 3, 'name': "UEFA Europa League", 'regular_round_keywords': []},
    {'id': 848, 'name': "UEFA Europa Conference League", 'regular_round_keywords': []},
    {'id': 147, 'name': "Cup", 'regular_round_keywords': []}  # BEL
"""
for comp in [
    {'id': 78, 'name': "Bundesliga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Bundesliga"},
    {'id': 79, 'name': "2. Bundesliga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "2. Bundesliga"},
    {'id': 81, 'name': "DFB Pokal", 'regular_round_keywords': []},
    {'id': 135, 'name': "Serie A", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Serie A"},
    {'id': 136, 'name': "Serie B", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Serie B"},
    {'id': 140, 'name': "La Liga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "La Liga"},
    {'id': 141, 'name': "Segunda División", 'regular_round_keywords': ['Regular Season'],
     'fs_alias': "Segunda División"},
    {'id': 137, 'name': "Coppa Italia", 'regular_round_keywords': []},
    {'id': 143, 'name': "Copa del Rey", 'regular_round_keywords': []},
    {'id': 207, 'name': "Super League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -'],
     'fs_alias': "Super League"},  # SUI
    {'id': 209, 'name': "Schweizer Cup", 'regular_round_keywords': []},
    {'id': 106, 'name': "Ekstraklasa", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ekstraklasa"},  # POL
    {'id': 108, 'name': "Cup", 'regular_round_keywords': []},  # POL
    {'id': 46, 'name': "EFL Trophy", 'regular_round_keywords': []},
    {'id': 323, 'name': "Indian Super League",
     'regular_round_keywords': ['Regular Season', 'Qualifying Finals', 'Championship -'],
     'fs_alias': "Indian Super League"},  # IND
    {'id': 218, 'name': "Bundesliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -'],
     'fs_alias': "Bundesliga"},  # AUT
    {'id': 220, 'name': "Cup", 'regular_round_keywords': []},  # AUT
    {'id': 188, 'name': "A-League",
     'regular_round_keywords': ['Regular Season', 'Elimination Finals', 'Semi-finals', 'Grand Final'],
     'fs_alias': "A-League"},  # AUS
    {'id': 206, 'name': "Cup", 'regular_round_keywords': []},  # TUR
    {'id': 874, 'name': "Australia Cup", 'regular_round_keywords': []},  # AUS
    {'id': 203, 'name': "Süper Lig", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Süper Lig"},  # TUR
    {'id': 61, 'name': "Ligue 1", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ligue 1"},
    {'id': 62, 'name': "Ligue 2", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ligue 2"},
    {'id': 66, 'name': "Coupe de France", 'regular_round_keywords': []},
    {'id': 40, 'name': "Championship", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Championship"},
    {'id': 41, 'name': "League One", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "EFL League One"},
    {'id': 42, 'name': "League Two", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "EFL League Two"},
    {'id': 39, 'name': "Premier League", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Premier League"},
    {'id': 45, 'name': "FA Cup", 'regular_round_keywords': []},
    {'id': 179, 'name': "Premiership",
     'regular_round_keywords': ['1st Phase', 'Championship Round', 'Relegation Round -'],
     'fs_alias': "Premiership"},  # SCO
    {'id': 181, 'name': "FA Cup", 'regular_round_keywords': []},  # SCO
    {'id': 185, 'name': "League Cup", 'regular_round_keywords': []},  # SCO
    {'id': 119, 'name': "Superliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round'],
     'fs_alias': "Superliga"},  # DEN
    {'id': 121, 'name': "DBU Pokalen", 'regular_round_keywords': []},  # DEN
    {'id': 88, 'name': "Eredivisie", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Eredivisie"},
    {'id': 90, 'name': "KNVB Beker", 'regular_round_keywords': []},  # NED
    {'id': 307, 'name': "Pro League", 'regular_round_keywords': ['Regular Season'],
     'fs_alias': "Professional League"},  # SA
    {'id': 504, 'name': "King's Cup", 'regular_round_keywords': []},  # SA
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
in_out.load_matches("tmp_csv_store11_full_copy.csv")
all_loaded_comp_seasons = list(set([(x.comp.id, x.season) for x in global_instance.all_matches]))
Match.get_new_matches_data_using_api(existing=all_loaded_comp_seasons)

# Sort matches by datetime played (asc.)
for team in global_instance.all_teams:
    team.matches = sorted(team.matches, key=lambda match_: match_.datetime)

    # Check team regularity (assume each team plays exactly in one regular comp each season!)
    # And then match regular AF teams with FS teams !!!
    team.correct_team_regularity_and_match_af_fs_teams()

# TODO: Add debug print check for number of matches (both all and just regular ones) for each comp season

# Exclude irregular teams from tables calc. + get FS players from all teams for each comp season (represented by table)
SeasonCompTable.exclude_irregular_teams_from_table_calculations()
SeasonCompTable.init_players_lists_in_regular_comp_season_teams()

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
# in_out.store_matches("tmp_csv_store11_full.csv")

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
