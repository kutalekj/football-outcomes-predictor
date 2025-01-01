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
import in_out_mega
from train import train

global_instance = Global.get_instance()

if not settings.MEGA_LOAD:

    # 0. Load avg SOFIFA average goalkeeper skills values and average team strengths
    in_out.load_avg_gk_skills()
    in_out.load_avg_team_strength_scaled()

    # 1. Init FS comps and their seasons and rounds
    Comp.get_fs_leagues_list()

    for comp in [
        {'id': 94, 'name': "Primeira Liga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Liga NOS"},
        {'id': 106, 'name': "Ekstraklasa", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ekstraklasa"},
        # POL
        {'id': 207, 'name': "Super League",
         'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -'],
         'fs_alias': "Super League"},  # SUI
        {'id': 218, 'name': "Bundesliga",
         'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -'],
         'fs_alias': "Bundesliga"},  # AUT
        {'id': 323, 'name': "Indian Super League",
         'regular_round_keywords': ['Regular Season', 'Qualifying Finals', 'Championship -'],
         'fs_alias': "Indian Super League"},  # IND
        {'id': 96, 'name': "Taça de Portugal", 'regular_round_keywords': [], 'fs_alias': "Taça de Portugal"},
        {'id': 97, 'name': "Taça da Liga", 'regular_round_keywords': [], 'fs_alias': "Portuguese League Cup"},
        {'id': 108, 'name': "Cup", 'regular_round_keywords': [], 'fs_alias': "Polish Cup"},  # POL
        {'id': 209, 'name': "Schweizer Cup", 'regular_round_keywords': [], 'fs_alias': "Swiss Cup"},
        {'id': 220, 'name': "Cup", 'regular_round_keywords': [], 'fs_alias': "Austrian Cup"}  # AUT
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
    if settings.LOAD_MATCH_DATA_FROM_LOCAL_CSV:
        in_out.load_matches("tmp_csv_store14_BEL_TUR_SCO_SA_DEN_NED_FRA_GER.csv")
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
    SeasonCompTable.init_fs_players_lists_in_regular_comp_season_teams()

    # Load individual player stats from sofifa CSV files
    in_out.load_player_stats()

    # 4. Calculate features for each match (must be done chronologically asc.!)
    global_instance.all_matches = sorted(global_instance.all_matches, key=lambda match_: match_.datetime)
    for match in global_instance.all_matches:

        # DEBUG
        if match.home_team.name == "Genk" or match.away_team.name == "Genk":
            stop_here = True

        # Match AF/FS match lineups
        ut.get_fs_match_lineups(match)  # match players in AF match lineup with those in teams' FS comp season roster

        # Get xG match stats (FS)
        if match.round.is_regular:
            ut.get_fs_match_xg(match)

        # Feature calculation
        # TODO: Debug print (comment for FS/SF matching acc check)
        # print(f"\n\t\tProcessing match between [{match.home_team.name}] and [{match.away_team.name}] ({match.datetime}).")
        match.features_before_match_played = match.calculate_match_features()
        match.feature_vector_before_match_played = MatchFeatures.match_features_to_vector(
            match.features_before_match_played)

    # TODO: Remove these
    print(f"Mean home team xG: {np.mean(global_instance.home_team_xg)}")
    print(f"Var home team xG: {np.var(global_instance.home_team_xg)}")
    print(f"StdDev home team xG: {np.std(global_instance.home_team_xg)}")
    print(f"Mean away team xG: {np.mean(global_instance.away_team_xg)}")
    print(f"Var away team xG: {np.var(global_instance.away_team_xg)}")
    print(f"StdDev away team xG: {np.std(global_instance.away_team_xg)}")
    print(f"Mean total xG: {np.mean(global_instance.total_xg)}")
    print(f"Var total xG: {np.var(global_instance.total_xg)}")
    print(f"StdDev total xG: {np.std(global_instance.total_xg)}")
    print(f"Mean home team pre-match xG: {np.mean(global_instance.home_team_pre_match_xg)}")
    print(f"Var home team pre-match xG: {np.var(global_instance.home_team_pre_match_xg)}")
    print(f"StdDev home team pre-match xG: {np.std(global_instance.home_team_pre_match_xg)}")
    print(f"Mean away team pre-match xG: {np.mean(global_instance.away_team_pre_match_xg)}")
    print(f"Var away team pre-match xG: {np.var(global_instance.away_team_pre_match_xg)}")
    print(f"StdDev away team pre-match xG: {np.std(global_instance.away_team_pre_match_xg)}")
    print(f"Mean total pre-match xG: {np.mean(global_instance.total_pre_match_xg)}")
    print(f"Var total pre-match xG: {np.var(global_instance.total_pre_match_xg)}")
    print(f"StdDev total pre-match xG: {np.std(global_instance.total_pre_match_xg)}")
else:
    in_out_mega.load_all_matches_data()

# 5. Store matches
in_out.store_matches("tmp_csv_store14_POR_POL_SUI_AUT_IND.csv")

if settings.MEGA_STORE:
    in_out_mega.store_all_matches_data()

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
"""

print("breakpoint")
