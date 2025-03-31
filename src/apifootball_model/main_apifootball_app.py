import http.client
import json
import settings
from match import Match
from feature import MatchFeatures
from globals import Global
import in_out_mega
from datetime import datetime, timedelta
from dateutil.parser import parse
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import utils as ut
import in_out
from comp import Comp
from season_comp_table import SeasonCompTable
from train_ann import get_embedding_extractor, normalize_embeddings
import tzlocal


"""
conn = http.client.HTTPSConnection(settings.HOST)
conn.request("GET", "/timezone", headers=settings.HEADERS)
res = conn.getresponse()
data = res.read()
data_tz = json.loads(data)

yesterday = datetime.now() - timedelta(days=1)
tomorrow = datetime.now() + timedelta(days=1)

# API call
request_string = "/fixtures?season=" + str(settings.LAST_SEASON) + "&league=" + str(119) + \
                 "&from=" + yesterday.strftime("%Y-%m-%d") + "&to=" + tomorrow.strftime("%Y-%m-%d") + "&timezone=Europe/Amsterdam"

conn = http.client.HTTPSConnection(settings.HOST)
conn.request("GET", request_string, headers=settings.HEADERS)
res = conn.getresponse()
data = res.read()
data_fixtures = json.loads(data)
"""

WAIT_SECS = 420

global_instance = Global.get_instance()

# 0. Load average skills and team strengths (SF)
in_out.load_sf_avg_team_strength()

# 1. Init comps (seasons, teams, AF rounds, FS matches)
Comp.get_fs_leagues_list()

for comp in settings.COMPS_v2:
    new_comp = Comp(comp['id'], comp['name'], comp['regular_round_keywords'])
    new_comp.init_teams_in_comp()
    new_comp.init_all_rounds()

    global_instance.all_comps.append(new_comp)
global_instance.all_teams = sorted(global_instance.all_teams, key=lambda team_: team_.id)

# 2. Init country start/end dates, comp season tables and default GK skills
for comp in global_instance.all_comps:
    for season in [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]:

        # Country start/end dates
        if comp.country not in global_instance.start_end_dates_per_country_season:
            global_instance.start_end_dates_per_country_season[comp.country] = {}

        if season not in global_instance.start_end_dates_per_country_season[comp.country]:
            global_instance.start_end_dates_per_country_season[comp.country][season] = {
                'start': datetime.max, 'end': datetime.min}

        if len(comp.regular_round_keywords) == 0:
            continue  # omit the cups - do not create tables for them

        new_table = SeasonCompTable(comp.id, comp.name, season)
        new_table.init_teams_in_season_comp()

        global_instance.all_tables.append(new_table)

for comp in global_instance.all_comps:
    comp.init_country_start_end_dates_in_seasons()

# 3.
in_out_mega.load_all_matches_data()  # load matches data

# 3+.
for team in global_instance.all_teams:  # init teams matches (mega load only loads "global_instance.all_matches")
    team.matches = [x for x in global_instance.all_matches if x.home_team.id == team.id or x.away_team.id == team.id]
    team.matches = sorted(team.matches, key=lambda match_: match_.datetime)

# 4. Correct (set) team regularity and match AF teams with FS teams
for team in global_instance.all_teams:
    team.matches = sorted(team.matches, key=lambda match_: match_.datetime)  # sort team matches by datetime (asc.)
    team.correct_team_regularity_and_match_af_fs_teams()  # assume each team plays exactly in one reg. comp season!

# 5a. Exclude irregular teams from table calculations
SeasonCompTable.exclude_irregular_teams_from_table_calc()

# 5b. Get FS players from all teams for each comp season (represented by table)
SeasonCompTable.get_fs_player_rosters_per_regular_comp_season_team()

# 6. Load individual player stats from sofifa CSV files
in_out.load_player_stats()

model = load_model(settings.MAIN_MODEL_PATH)  # load prediction model

regular_matches = [x for x in global_instance.all_matches if x.round.is_regular]
regular_matches = sorted(regular_matches, key=lambda match_: match_.datetime)
team_id_map, comp_id_map = ut.get_categorical_features_maps(regular_matches)  # init categorical features mapping

comp_id_embedding_model = load_model(settings.COMP_ID_EMBEDDING_MODEL_PATH)
team_id_embedding_model = load_model(settings.TEAM_ID_EMBEDDING_MODEL_PATH)
team_strength_embedding_model = load_model(settings.TEAM_STRENGTH_EMBEDDING_MODEL_PATH)

comp_id_embedding_model = get_embedding_extractor(comp_id_embedding_model, 'competition_embedding')
team_id_embedding_model = get_embedding_extractor(team_id_embedding_model, 'team_embedding')

while True:
    matches_to_predict = []
    matches_to_predict_no_af_lineups = []
    lineup_estimation_logs = []

    # Collect upcoming regular matches data for prediction
    for comp in global_instance.all_comps:
        if len(comp.regular_round_keywords) > 0:
            yesterday = datetime.now() - timedelta(days=1)
            tomorrow = datetime.now() + timedelta(days=1)

            # API call
            request_string = "/fixtures?season=" + str(settings.LAST_SEASON) + "&league=" + str(comp.id) + \
                             "&from=" + yesterday.strftime("%Y-%m-%d") + "&to=" + tomorrow.strftime("%Y-%m-%d") + \
                             "&timezone=Europe/Amsterdam"

            conn = http.client.HTTPSConnection(settings.HOST)
            conn.request("GET", request_string, headers=settings.HEADERS)
            res = conn.getresponse()
            data = res.read()
            data_fixtures = json.loads(data)
            print(f"[3] {len(data_fixtures['response'])} matches were found in comp {comp.name}")

            for fixture in data_fixtures['response']:
                new_match_id = int(fixture['fixture']['id'])
                new_match = Match(new_match_id)

                new_match.status = fixture['fixture']['status']['short']

                if new_match.status not in ["NS"]:
                    print(
                        f"Found match between {fixture['teams']['home']['name']} and {fixture['teams']['away']['name']}"
                        f" played at {fixture['fixture']['date']} with the status of [{new_match.status}]. Skipping...")
                    continue

                new_match.datetime = parse(fixture['fixture']['date'])
                new_match.hour = int(new_match.datetime.hour)
                new_match.month = int(new_match.datetime.month)
                new_match.country = fixture['league']['country']

                new_match.comp = comp
                if int(fixture['league']['id']) != comp.id:
                    raise ValueError(
                        f"Comp ID found [{fixture['league']['id']}] not matching expected value {str(comp.id)}")
                if fixture['league']['name'] != comp.name:
                    raise ValueError(
                        f"Comp name found [{fixture['league']['name']}] not matching expected value {comp.name}")

                new_match.season = settings.LAST_SEASON
                if int(fixture['league']['season']) != settings.LAST_SEASON:
                    raise ValueError(
                        f"Found season [{fixture['league']['season']}], but [{str(settings.LAST_SEASON)}] was expected")

                new_match.round = comp.get_round_by_comp_season_round_name(
                    settings.LAST_SEASON, fixture['league']['round'])
                if new_match.round is None:
                    raise ValueError(f"Unable to get round for the match {str(new_match.id)}")

                # Home team
                home_team_id = int(fixture['teams']['home']['id'])
                home_team = ut.get_team_if_exists(home_team_id)
                if home_team is None:
                    print(f"\t\t\t\t\tFAILED to find a home team with ID {home_team_id} to assign a match.")
                    continue
                new_match.home_team = home_team
                home_team.matches.append(new_match)

                # Away team
                away_team_id = int(fixture['teams']['away']['id'])
                away_team = ut.get_team_if_exists(away_team_id)
                if away_team is None:
                    print(f"\t\t\t\t\tFAILED to find an away team with ID {home_team_id} to assign a match.")
                    continue
                new_match.away_team = away_team
                away_team.matches.append(new_match)

                # Future stats (non-existing)
                new_match.winner_team_id = -1
                new_match.home_team_goals = 0
                new_match.away_team_goals = 0
                new_match.home_team_points = 0
                new_match.away_team_points = 0

                # Lineups
                lineups_request_string = "/fixtures/lineups?fixture=" + str(new_match.id)
                conn.request("GET", lineups_request_string, headers=settings.HEADERS)
                res = conn.getresponse()
                data = res.read()
                data_lineups = json.loads(data)['response']

                if len(data_lineups) == 0:
                    print(f"\tLineups missing for both teams in a regular match between "
                          f"{new_match.home_team.name} and {new_match.away_team.name} ({new_match.datetime})")
                else:
                    if "startXI" in data_lineups[0]:
                        new_match.home_team_lineup = \
                            [(x['player']['id'], x['player']['name'], x['player']['pos'])
                             for x in data_lineups[0]['startXI']]
                        if len(new_match.home_team_lineup) > 11:  # if AF lineup is duplicated (API-football issue)
                            if len(new_match.home_team_lineup) == 22:
                                new_match.home_team_lineup = new_match.home_team_lineup[:11]  # fix
                            if len(new_match.home_team_lineup) != 11:  # check the fix
                                raise ValueError(f"AF match home team lineup: [{new_match.home_team_lineup}] "
                                                 f"should contain more than 11 players")
                    else:
                        new_match.home_team_lineup = []
                        print(f"\tLineups missing for a home team in match between {new_match.home_team.name} and "
                              f"{new_match.away_team.name} played at {new_match.datetime}")
                    if "startXI" in data_lineups[1]:
                        new_match.away_team_lineup = \
                            [(x['player']['id'], x['player']['name'], x['player']['pos'])
                             for x in data_lineups[1]['startXI']]
                        if len(new_match.away_team_lineup) > 11:  # if AF lineup is duplicated (API-football issue)
                            if len(new_match.away_team_lineup) == 22:
                                new_match.away_team_lineup = new_match.away_team_lineup[:11]  # fix
                            if len(new_match.away_team_lineup) != 11:  # check the fix
                                raise ValueError(f"AF match away team lineup: [{new_match.away_team_lineup}] "
                                                 f"should contain more than 11 players")
                    else:
                        new_match.away_team_lineup = []
                        print(f"\tLineups missing for an away team in match between {new_match.home_team.name} and "
                              f"{new_match.away_team.name} played at {new_match.datetime}")

                if new_match.home_team_lineup is not None and new_match.away_team_lineup is not None and \
                        len(new_match.home_team_lineup) > 0 and len(new_match.away_team_lineup) > 0:
                    matches_to_predict.append(new_match)

                # If no AF lineups, but match to be played within 30 minutes, take prev match lineups for both team
                # TODO: I think the time zones are still not handled correctly here...
                elif 0 < (new_match.datetime - datetime.now().replace(tzinfo=new_match.datetime.tzinfo)).total_seconds() <= 1800:
                    home_team_prev_match = ut.get_previous_match(new_match, new_match.home_team.id, same_comp=True,
                                                                 same_season=False, regular=True)
                    home_team_prev_af_lineup = home_team_prev_match.home_team_lineup if \
                        home_team_prev_match.home_team.id == new_match.home_team.id else \
                        home_team_prev_match.away_team_lineup

                    away_team_prev_match = ut.get_previous_match(new_match, new_match.away_team.id, same_comp=True,
                                                                 same_season=False, regular=True)
                    away_team_prev_af_lineup = away_team_prev_match.away_team_lineup if \
                        away_team_prev_match.away_team.id == new_match.away_team.id else \
                        away_team_prev_match.home_team_lineup

                    new_match.home_team_lineup = home_team_prev_af_lineup
                    new_match.away_team_lineup = away_team_prev_af_lineup

                    if len(new_match.home_team_lineup) != 11:  # check the fix
                        raise ValueError(f"AF match home team lineup: [{new_match.home_team_lineup}] "
                                         f"should contain 11 players")
                    if len(new_match.away_team_lineup) != 11:  # check the fix
                        raise ValueError(f"AF match away team lineup: [{new_match.away_team_lineup}] "
                                         f"should contain 11 players")

                    log_str = f"\t\t\t\t\tAF lineups taken from previous matches for teams " \
                              f"[{new_match.home_team.name}] and [{new_match.away_team.name}] " \
                              f"(match played at {new_match.datetime})"
                    print(log_str)
                    lineup_estimation_logs.append(log_str)

                    matches_to_predict.append(new_match)

    # ----------- PREDICTION -----------
    board_queue_entries = []
    for match in matches_to_predict:
        try:
            if match.home_fs_team_lineup is not None and match.away_fs_team_lineup is not None and \
                    len(match.home_fs_team_lineup) == 0 and len(match.away_fs_team_lineup) == 0:
                print("")
                ut.get_fs_match_lineups(match)  # FS lineups

            match.features_before_match_played = match.calculate_match_features()  # features
            match.feature_vector_before_match_played = MatchFeatures.match_features_to_vector(
                match.features_before_match_played)

            numerical_input = np.array([match.feature_vector_before_match_played])

            # Map raw categorical features to indices
            home_id_input = [team_id_map[match.home_team.id]]
            away_id_input = [team_id_map[match.away_team.id]]
            comp_id_input = [comp_id_map[match.comp.id]]

            # Scale team strength features skills to [0,1]
            home_strength = np.array(
                [[z / 100.0 for z in y] for y in match.features_before_match_played.home_team_strength])
            away_strength = np.array(
                [[z / 100.0 for z in y] for y in match.features_before_match_played.away_team_strength])

            # Normalize team strength features (expected shape is (11, 34) - add batch dimension, then remove it)
            home_strength_norm = np.squeeze(
                ut.separate_normalize_gk_and_outfield_skills(np.expand_dims(home_strength, axis=0)), axis=0)  # TODO: Error once occurred here - home_strength not having enough dimensions
            away_strength_norm = np.squeeze(
                ut.separate_normalize_gk_and_outfield_skills(np.expand_dims(away_strength, axis=0)), axis=0)

            home_strength_input = [home_strength_norm]
            away_strength_input = [away_strength_norm]

            # For categorical features add extra dimension so each input is shape (1,)
            home_id_array = np.expand_dims(np.array(home_id_input), axis=-1)  # shape: (num_samples = 1, 1)
            away_id_array = np.expand_dims(np.array(away_id_input), axis=-1)
            comp_id_array = np.expand_dims(np.array(comp_id_input), axis=-1)

            # For team strength assuming each normalized sample is (11, 34)
            home_strength_array = np.array(home_strength_input)  # shape: (num_samples = 1, 11, 34)
            away_strength_array = np.array(away_strength_input)

            # Batch predict embeddings
            home_id_embedding = np.squeeze(team_id_embedding_model.predict(home_id_array), axis=1)  # (num_samples = 1, 8)
            away_id_embedding = np.squeeze(team_id_embedding_model.predict(away_id_array), axis=1)
            comp_id_embedding = np.squeeze(comp_id_embedding_model.predict(comp_id_array), axis=1)
            home_strength_embedding = team_strength_embedding_model.predict(home_strength_array)
            away_strength_embedding = team_strength_embedding_model.predict(away_strength_array)

            # Normalize predicted embeddings to [0,1]
            home_id_embedding = normalize_embeddings(home_id_embedding)
            away_id_embedding = normalize_embeddings(away_id_embedding)
            comp_id_embedding = normalize_embeddings(comp_id_embedding)

            # Generate model prediction
            pred = model.predict([numerical_input, home_id_embedding, away_id_embedding, comp_id_embedding,
                                  home_strength_embedding, away_strength_embedding])
            pred_prob = float(pred[0][0])

            # Prepare the board queue entry
            board_entry = {
                "match_id": match.id,
                "country": match.country,
                "comp_name": match.comp.name,
                "season": match.season,
                "home_team": match.home_team.name,
                "away_team": match.away_team.name,
                "datetime": (match.datetime.astimezone(tzlocal.get_localzone())).isoformat(),
                "lineups": {
                    "home": [player[1] for player in match.home_team_lineup],
                    "away": [player[1] for player in match.away_team_lineup]
                },
                "prediction": pred_prob,
                "processed": False
            }
            board_queue_entries.append(board_entry)
        except:
            pass

    # Update "the board_queue.json" file with the new prediction entries
    with open(settings.BOARD_QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(board_queue_entries, f, indent=2)

    for log_s in lineup_estimation_logs:
        print(str(log_s))
    print(f"Saved {len(board_queue_entries)} entries to the board queue")

    time.sleep(WAIT_SECS)  # active waiting
