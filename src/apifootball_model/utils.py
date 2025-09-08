"""
utils.py
"""

import numpy as np
import re
import unicodedata
import difflib
from rapidfuzz import fuzz
from collections import Counter
import settings
import time
import requests
from globals import Global
from settings import MAX_MATCH_HISTORY_TO_CHECK_LOW
from datetime import timedelta, datetime


def get_previous_match(curr_match, team_id, same_comp=False, same_season=False, regular=False):
    team = get_team_if_exists(team_id)

    curr_match_idx = team.get_index_of_match_in_sorted_team_matches_list(curr_match)

    # A first match has no previous matches
    if curr_match_idx == 0 or curr_match_idx is None:
        return None

    prev_match = team.matches[curr_match_idx - 1]

    if prev_match is None:
        return None

    if same_comp:
        if curr_match.comp != prev_match.comp:
            return get_previous_match(prev_match, team.id, same_comp, same_season, regular)

    if same_season:
        if curr_match.season > prev_match.season:
            return None

    if regular:
        if not prev_match.round.is_regular:
            return get_previous_match(prev_match, team.id, same_comp, same_season, regular)

    return prev_match


def get_last_home_away_match(curr_match, team_id, home_away=None, same_comp=False, same_season=False, regular=False):
    curr_prev_match = get_previous_match(curr_match, team_id, same_comp, same_season, regular)

    if curr_prev_match is None:
        return None

    # Searching for a last home match of team "team_id"
    if home_away == "home":
        # Found and it is a home match
        if curr_prev_match.home_team.id == team_id:
            return curr_prev_match

        # Found away match - go back until home match is found
        elif curr_prev_match.away_team.id == team_id:
            new_curr_prev_match = curr_prev_match

            for i in range(0, MAX_MATCH_HISTORY_TO_CHECK_LOW):
                prev_prev_match = get_previous_match(new_curr_prev_match, team_id, same_comp, same_season, regular)

                # There might not be any previous matches left
                if prev_prev_match is None:
                    return None

                if prev_prev_match.home_team.id == team_id:
                    return prev_prev_match

                new_curr_prev_match = prev_prev_match

            raise Exception(
                f"Last home match not found even after checking last {MAX_MATCH_HISTORY_TO_CHECK_LOW} matches.")

        else:
            raise ValueError(f"Found team with ID={team_id} which is neither home or away in a match.")

    # Searching for a last away match of team "team_id"
    elif home_away == "away":

        # Found and it is an away match
        if curr_prev_match.away_team.id == team_id:
            return curr_prev_match

        # Found home match - go back until away match is found
        elif curr_prev_match.home_team.id == team_id:
            new_curr_prev_match = curr_prev_match

            for i in range(0, MAX_MATCH_HISTORY_TO_CHECK_LOW):
                prev_prev_match = get_previous_match(new_curr_prev_match, team_id, same_comp, same_season, regular)

                # There might not be any previous matches left
                if prev_prev_match is None:
                    return None

                if prev_prev_match.away_team.id == team_id:
                    return prev_prev_match

                new_curr_prev_match = prev_prev_match

            raise Exception(
                f"Last away match not found even after checking last {MAX_MATCH_HISTORY_TO_CHECK_LOW} matches.")

        else:
            raise ValueError(f"Found team with ID={team_id} which is neither home or away in a match.")

    else:
        raise ValueError("The \"home_away\" parameter set to a wrong value.")


def get_n_previous_matches(n, curr_match, team_id, home_away=None, same_comp=False, same_season=False, regular=False):
    # TODO: Remember that the previous match is in the first position in the list while the n-th previous is last...
    n_previous_matches = []

    # Get N last home matches
    if home_away == "home":
        curr_prev_match = curr_match

        for i in range(0, n):
            prev_home_match = get_last_home_away_match(curr_prev_match, team_id, "home", same_comp, same_season,
                                                       regular)
            n_previous_matches.append(prev_home_match)

            curr_prev_match = prev_home_match

    # Get N last away matches
    elif home_away == "away":
        curr_prev_match = curr_match

        for i in range(0, n):
            prev_away_match = get_last_home_away_match(curr_prev_match, team_id, "away", same_comp, same_season,
                                                       regular)
            n_previous_matches.append(prev_away_match)

            curr_prev_match = prev_away_match

    # Get N last matches
    else:
        curr_prev_match = curr_match

        for i in range(0, n):
            prev_match = get_previous_match(curr_prev_match, team_id, same_comp, same_season, regular)
            n_previous_matches.append(prev_match)

            curr_prev_match = prev_match

    return n_previous_matches


def get_all_regular_matches_in_season_table_up_to_date(curr_season_table, date):
    matches_up_to_date = []
    for team in curr_season_table.teams:
        # Get only regular matches up to the wanted date (from the current season)
        team_matches = [match for match in team.matches if
                        match.season == curr_season_table.season and match.datetime < date and match.round.is_regular]

        matches_up_to_date += team_matches

    # Remove duplicates (each match expected to appear twice)
    return list(set(matches_up_to_date))


def distribute_matches_into_rounds(sorted_matches):
    rounds = []
    current_round = []
    teams_in_current_round = set()

    for match in sorted_matches:
        home_team_id = match.home_team.id
        away_team_id = match.away_team.id

        # Ensure that no team appears multiple times in a round
        if home_team_id in teams_in_current_round or away_team_id in teams_in_current_round:

            # If possible second appearance, finish current round and start new one
            rounds.append(current_round)
            current_round = []
            teams_in_current_round = set()

        current_round.append(match)  # add match to current round
        teams_in_current_round.add(home_team_id)
        teams_in_current_round.add(away_team_id)

    # Append the last round if not empty
    if current_round:
        rounds.append(current_round)

    return rounds


def distribute_matches_into_rounds_uniformly(sorted_matches):
    rounds = []
    current_round = []

    # This implementation allows teams appearing in a single round multiple times
    for match in sorted_matches:
        if len(current_round) < settings.NUM_MATCHES_PER_ROUND_FOR_TRAINING:
            current_round.append(match)
        else:
            rounds.append(current_round)
            current_round = []

    # Append the last round if not empty
    if current_round:
        if len(current_round) < int(settings.NUM_MATCHES_PER_ROUND_FOR_TRAINING / 2):
            rounds[len(rounds) - 1] += current_round
        else:
            rounds.append(current_round)

    return rounds


def get_table_by_comp_season(comp_id, season):
    global_instance = Global.get_instance()

    for table in global_instance.all_tables:
        if table.comp_id == comp_id and table.season == season:
            return table

    return None


def get_team_if_exists(team_id):
    global_instance = Global.get_instance()

    for team in global_instance.all_teams:
        if team.id == team_id:
            return team

    return None


def get_comp_by_id(comp_id):
    global_instance = Global.get_instance()

    for comp in global_instance.all_comps:
        if comp.id == comp_id:
            return comp

    return None


def normalize_name(name):
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')  # to ASCII - remove accents
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s]', '', name)  # remove all non-alphanumeric characters
    name = ' '.join(name.split())  # remove whitespaces
    return name


def get_sf_player_data(match_datetime, sf_player_id, team_season_info, fs_position):
    global_instance = Global.get_instance()
    team_id, team_name, season = team_season_info

    # Get the latest available CSV file older than the match date
    sf_player_available_csv_files = global_instance.sofifa_player_index_dict[sf_player_id]
    if len(sf_player_available_csv_files) == 0:
        raise ValueError(f"If the FS player was matched to a SF player (id={sf_player_id}), "
                         f"there should be always at least one CSV file for him, not zero (even if the players "
                         f"were mismatched)! (match played at {match_datetime})")

    available_player_csvs_sorted_by_timedelta_to_match = sorted(sf_player_available_csv_files,
                                                                key=lambda x: abs(x[1].replace(tzinfo=match_datetime.tzinfo) - match_datetime))  # sort
    available_player_csvs_sorted_by_timedelta_to_match = [x for x in available_player_csvs_sorted_by_timedelta_to_match
                                                          if abs(x[1].replace(tzinfo=match_datetime.tzinfo) - match_datetime) < settings.MAX_TIMEDELTA_SF_PLAYER_SKILL]  # filter

    if len(available_player_csvs_sorted_by_timedelta_to_match) == 0:
        print(f"There are no available player CSV files within the timedelta range for player {sf_player_id}. "
              f"Imputing...")
        # TODO manual output check: count how many such players without CSV data are there
        return get_imitated_player_skills(season, team_id, fs_position)

    # DEBUG PRINT
    print(f"{len(available_player_csvs_sorted_by_timedelta_to_match)} available CSV files found...")

    skills_processed = set()  # keep track of already processed player skills
    collected_player_skills = {skill: -1 for skill in settings.PLAYER_SKILLS}

    # Loop because the first list element (closest timedelta to match) may not contain all skill values...
    all_player_positions = []
    while len(available_player_csvs_sorted_by_timedelta_to_match) > 0:
        best_available_player_csv = available_player_csvs_sorted_by_timedelta_to_match.pop(0)  # date closest to match

        index_to_player_data_list = best_available_player_csv[0]
        datetime_of_csv_data_file = best_available_player_csv[1]

        negative_value_found = False

        # Get the SOFIFA player data
        corresponding_player_data_dict = global_instance.sofifa_players_data[index_to_player_data_list]

        if corresponding_player_data_dict[0] != datetime_of_csv_data_file:
            raise ValueError(f"Datetime of the latest CSV file available for the player that is older than the match "
                             f"date does not match the expected datetime in the main SOFIFA players data list: "
                             f"({str(corresponding_player_data_dict[0])} vs. {str(datetime_of_csv_data_file)})")

        sf_player_data = corresponding_player_data_dict[1][sf_player_id]  # get SOFIFA player data dict

        # Determine player's position categories
        player_position_categories = map_player_positions_to_categories(sf_player_data['positions'])
        all_player_positions += player_position_categories

        # Get SOFIFA player skill values
        for skill in settings.PLAYER_SKILLS:
            if skill in skills_processed:  # skill already processed in the previous iteration
                continue

            value = sf_player_data.get(skill, -1)  # get skill value

            if value == -1:  # missing skill value
                negative_value_found = True
                continue

            collected_player_skills[skill] = float(value)
            skills_processed.add(skill)  # keep track of already processed skills for the player

        if not negative_value_found:
            break  # if no -1 values found, end getting skills

    # Check for missing skill values - impute
    for index, (skill_name, value) in enumerate(collected_player_skills.items()):
        if value == -1:
            player_pos = get_most_frequent_string(all_player_positions)  # get most probable player position
            collected_player_skills[skill_name] = \
                global_instance.sf_avg_team_strength[(season, team_id, player_pos)][index]  # impute a single value

    if len(collected_player_skills) != len(settings.PLAYER_SKILLS):
        raise ValueError(f"Found {len(collected_player_skills)} skill values for SF player (id={sf_player_id}), but "
                         f"{len(settings.PLAYER_SKILLS)} expected (match played at {match_datetime})")

    # Convert collected player skills dictionary to a list of 34 float values
    values_list = list(collected_player_skills.values())

    if not is_valid_n_float_list(values_list):
        raise ValueError(f"Unexpected shape of player skills variable: {values_list}")
    return values_list


def match_af_team_to_fs_team(af_team_name, fs_teams_in_comp_season):
    normalized_af_name = normalize_name(af_team_name)  # normalize AF team name

    best_fs_match = None
    highest_similarity = 0.0

    for fs_team in fs_teams_in_comp_season['fs_teams']:
        normalized_fs_clean_name = normalize_name(fs_team['cleanName'])  # normalize FS team name

        similarity = difflib.SequenceMatcher(None, normalized_af_name, normalized_fs_clean_name).ratio()  # similarity

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_fs_match = fs_team

    print(f"\t\t\t\t\tAF team matched to FS team: [{af_team_name}] [{best_fs_match['cleanName']}] "
          f"(similarity={str(highest_similarity)})")

    return best_fs_match['id'], best_fs_match['cleanName']


def get_fs_match_lineups(curr_match):  # for both home and away teams!
    if len(curr_match.comp.regular_round_keywords) == 0:  # skip for irregular matches
        return

    if len(curr_match.home_fs_team_lineup) > 0 and len(curr_match.away_fs_team_lineup) > 0:
        return  # case for both teams' FS lineups already loaded from CSV

    print(f"[7]\t\t Going to match AF players from match lineup [{curr_match.home_team.name}] vs. "
          f"[{curr_match.away_team.name}] ({curr_match.datetime}) with teams' FS players in comp season roster...")

    # Home team
    home_team_fs_players_in_comp_season = [x['fs_players'] for x in curr_match.home_team.players_in_regular_comp_season
                                           if curr_match.comp == x['comp'] and curr_match.season == x['season']]

    if len(home_team_fs_players_in_comp_season) > 1:  # error
        raise ValueError(f"Multiple FS home team comp season lineups found for match between "
                         f"{curr_match.home_team.name} and {curr_match.away_team.name} "
                         f"({curr_match.datetime}) - ERROR!")

    if len(curr_match.home_fs_team_lineup) > 0 or \
            len(home_team_fs_players_in_comp_season) == 0 or \
            (curr_match.home_team_lineup is None or len(curr_match.home_team_lineup) == 0):  # empty
        if len(curr_match.home_fs_team_lineup) > 0:
            pass  # case for only home team's FS lineup already loaded from CSV - do nothing
        elif len(home_team_fs_players_in_comp_season) == 0:
            print(f"Match between {curr_match.home_team.name} and {curr_match.away_team.name} "
                  f"({curr_match.datetime}) is regular, but no FS players found for the home team comp season "
                  f"- no FS team lineups")
        else:
            print(f"WARNING!!! - No AF home team lineup for match between {curr_match.home_team.name} and "
                  f"{curr_match.away_team.name} ({curr_match.datetime})")
        curr_match.home_fs_team_lineup = []

    else:  # ok
        for af_player in curr_match.home_team_lineup:
            matched_fs_player, similarity = match_af_player_to_fs_player_alternative(
                af_player, home_team_fs_players_in_comp_season[0])

            curr_match.home_fs_team_lineup.append(matched_fs_player)

        if len(curr_match.home_fs_team_lineup) != 11:
            raise ValueError(f"There were {len(curr_match.home_fs_team_lineup)} matched home team FS players, "
                             f"but the expected numbers of matches is 11")

    # Away team
    away_team_fs_players_in_comp_season = [x['fs_players'] for x in
                                           curr_match.away_team.players_in_regular_comp_season
                                           if curr_match.comp == x['comp'] and curr_match.season == x['season']]

    if len(away_team_fs_players_in_comp_season) > 1:  # error
        raise ValueError(f"Multiple FS away team comp season lineups found for match between "
                         f"{curr_match.home_team.name} and {curr_match.away_team.name} "
                         f"({curr_match.datetime}) - ERROR!")

    if len(curr_match.away_fs_team_lineup) > 0 or \
            len(away_team_fs_players_in_comp_season) == 0 or \
            (curr_match.away_team_lineup is None or len(curr_match.away_team_lineup) == 0):  # empty
        if len(curr_match.away_fs_team_lineup) > 0:
            pass  # case for only away team's FS lineup already loaded from CSV - do nothing
        elif len(away_team_fs_players_in_comp_season) == 0:
            print(f"Match between {curr_match.home_team.name} and {curr_match.away_team.name} "
                  f"({curr_match.datetime}) is regular, but no FS players found for the away team comp season "
                  f"- no FS team lineups")
        else:
            print(f"WARNING!!! - No AF away team lineup for match between {curr_match.home_team.name} and "
                  f"{curr_match.away_team.name} ({curr_match.datetime})")
        curr_match.away_fs_team_lineup = []

    else:  # ok
        for af_player in curr_match.away_team_lineup:
            matched_fs_player, similarity = match_af_player_to_fs_player_alternative(
                af_player, away_team_fs_players_in_comp_season[0])

            curr_match.away_fs_team_lineup.append(matched_fs_player)

        if len(curr_match.away_fs_team_lineup) != 11:
            raise ValueError(f"There were {len(curr_match.away_fs_team_lineup)} matched away team FS players, "
                             f"but the expected numbers of matches is 11")


def get_fs_match_xg(curr_match):
    global_instance = Global.get_instance()

    all_fs_matches_this_comp_season = global_instance.fs_leagues_matches[(curr_match.comp.id, curr_match.season)]
    fs_match_ids = [x["fs_match_id"] for x in all_fs_matches_this_comp_season if
                    x["fs_home_team_id"] == curr_match.home_team.fs_id and
                    x["fs_away_team_id"] == curr_match.away_team.fs_id and
                    x["season"] == curr_match.season and
                    x["datetime"].year == curr_match.datetime.year and
                    x["datetime"].month == curr_match.datetime.month and
                    x["datetime"].day == curr_match.datetime.day]

    if len(fs_match_ids) != 1:
        print(f"Found none, or multiple FS match IDs: [{fs_match_ids}] for a single match "
              f"({curr_match.home_team.id}: {curr_match.home_team.name} vs. "
              f"{curr_match.away_team.id}: {curr_match.away_team.name} - {curr_match.datetime})")
        curr_match.home_team_xg, curr_match.away_team_xg, curr_match.total_xg = -1, -1, -1
        curr_match.home_team_pre_match_xg, curr_match.away_team_pre_match_xg, curr_match.total_pre_match_xg = -1, -1, -1
        return

    fs_match_id = fs_match_ids[0]
    match_details_request_string_fs = settings.FS_HOST + "/match?key=" + settings.FS_KEY \
        + "&match_id=" + str(fs_match_id)
    res = requests.get(match_details_request_string_fs)
    data_match_details_fs = res.json()
    fs_match_details_dict_comp_season = data_match_details_fs['data']

    curr_match.home_team_xg = float(fs_match_details_dict_comp_season["team_a_xg"]) \
        if float(fs_match_details_dict_comp_season["team_a_xg"]) > 0.001 else -1

    curr_match.away_team_xg = float(fs_match_details_dict_comp_season["team_b_xg"]) \
        if float(fs_match_details_dict_comp_season["team_b_xg"]) > 0.001 else -1

    curr_match.total_xg = float(fs_match_details_dict_comp_season["total_xg"]) \
        if float(fs_match_details_dict_comp_season["total_xg"]) > 0.001 else -1

    curr_match.home_team_pre_match_xg = float(fs_match_details_dict_comp_season["team_a_xg_prematch"]) \
        if float(fs_match_details_dict_comp_season["team_a_xg_prematch"]) > 0.001 else -1

    curr_match.away_team_pre_match_xg = float(fs_match_details_dict_comp_season["team_b_xg_prematch"]) \
        if float(fs_match_details_dict_comp_season["team_b_xg_prematch"]) > 0.001 else -1

    curr_match.total_pre_match_xg = float(fs_match_details_dict_comp_season["total_xg_prematch"]) \
        if float(fs_match_details_dict_comp_season["total_xg_prematch"]) > 0.001 else -1

    time.sleep(1.7)


def match_af_player_to_fs_player_alternative(af_player, fs_players_in_comp_season):
    normalized_af_name = normalize_name(af_player[1])  # normalize AF player name (id, name, pos)

    best_fs_match = None
    highest_similarity = 0.0

    for fs_player in fs_players_in_comp_season:
        normalized_fs_known_as_name = normalize_name(fs_player['fs_known_as'])  # normalize FS player name

        similarity = fuzz.ratio(normalized_af_name, normalized_fs_known_as_name)  # similarity

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_fs_match = fs_player

    print(f"[{af_player[1]}][{best_fs_match['fs_known_as']}]", end='\t')
    # TODO manual output check: AF/FS players matching accuracy

    return best_fs_match, highest_similarity


def match_fs_player_to_sf_players(fs_player, sf_players_with_same_dob):
    normalized_fs_known_as = normalize_name(fs_player['fs_known_as'])
    normalized_fs_full_name = normalize_name(fs_player['fs_full_name'])

    similarities = []  # triples of (similarity_score, sf_player_id, some_sf_player_name)

    for sf_player in sf_players_with_same_dob:  # sf_player = (player_id, name, full_name)
        normalized_sf_player_name = normalize_name(sf_player[1])  # SF player name
        normalized_sf_player_full_name = normalize_name(sf_player[2])  # SF player full_name

        # Try matching with 'fs_known_as'
        similarity_1a = fuzz.ratio(normalized_fs_known_as, normalized_sf_player_name)
        similarity_2a = fuzz.ratio(normalized_fs_known_as, normalized_sf_player_full_name)

        # Try matching with 'fs_full_name'
        similarity_1b = fuzz.ratio(normalized_fs_full_name, normalized_sf_player_name)
        similarity_2b = fuzz.ratio(normalized_fs_full_name, normalized_sf_player_full_name)

        similarities.append((similarity_1a, sf_player[0], sf_player[1]))
        similarities.append((similarity_2a, sf_player[0], sf_player[2]))
        similarities.append((similarity_1b, sf_player[0], sf_player[1]))
        similarities.append((similarity_2b, sf_player[0], sf_player[2]))

    sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

    # Return SF player who belongs to the name that gave the highest similarity
    highest_similarity_sf_player_id = sorted_similarities[0][1]
    highest_similarity_score = sorted_similarities[0][0]
    highest_similarity_sf_player = [x for x in sf_players_with_same_dob if x[0] == highest_similarity_sf_player_id][0]

    print(f"Matched [{fs_player['fs_known_as']}] to [{highest_similarity_sf_player[2]}] "
          f"({highest_similarity_sf_player[1]}) with score \t\t{highest_similarity_score:.2f}")

    if highest_similarity_score < settings.SIMILARITY_THRESHOLD_FS_SOFIFA:
        print("(but rejected for too low similarity score :/)")
        return None, None, None

    return highest_similarity_sf_player


def map_player_positions_to_categories(sofifa_positions):
    categories = set()

    for pos in sofifa_positions:
        if pos == 'GK':
            categories.add('goalkeeper')
        elif pos in ['CB', 'LB', 'RB', 'RWB', 'LWB']:
            categories.add('defender')
        elif pos == 'CDM':
            categories.update(['midfielder', 'defender'])
        elif pos in ['CM', 'LM', 'RM']:
            categories.add('midfielder')
        elif pos == 'CAM':
            categories.update(['midfielder', 'attacker'])
        elif pos in ['ST', 'CF', 'LW', 'RW']:
            categories.add('attacker')
        else:
            raise ValueError(f"Unknown player position [{pos}] found.")

    return list(categories)


def get_imitated_team_strength(season, team_id):
    global_instance = Global.get_instance()

    # 1x goalkeeper (handle possible missing values)
    gk_skills = global_instance.sf_avg_team_strength[(season, team_id, "goalkeeper")]
    players_skills = [global_instance.sf_avg_team_strength[(-1, -1, "goalkeeper")]] \
        if all(x == 0 for x in gk_skills) else [gk_skills]

    for i in range(4):
        players_skills.append(global_instance.sf_avg_team_strength[(season, team_id, "defender")])  # 4x defender

    for i in range(4):
        players_skills.append(global_instance.sf_avg_team_strength[(season, team_id, "midfielder")])  # 4x midfielder

    for i in range(2):
        players_skills.append(global_instance.sf_avg_team_strength[(season, team_id, "attacker")])  # 4x attacker

    # TODO adj: currently assuming a default 4-4-2 formation - can add more complex logic
    if len(players_skills) != 11:
        raise ValueError(f"The imitated team strength was estimated for {len(players_skills)}, but 11 were expected")
    if not is_valid_mxn_float_list(players_skills):
        raise ValueError(f"Unexpected shape of team player skills variable: {players_skills}")
    return players_skills


def get_imitated_player_skills(season, team_id, fs_position):
    global_instance = Global.get_instance()

    if fs_position == "Goalkeeper":
        gk_skills = global_instance.sf_avg_team_strength[(season, team_id, "goalkeeper")]
        imitated_player_skills = global_instance.sf_avg_team_strength[(-1, -1, "goalkeeper")] \
            if all(x == 0 for x in gk_skills) else gk_skills  # handle possible missing values
    elif fs_position == "Defender":
        imitated_player_skills = global_instance.sf_avg_team_strength[(season, team_id, "defender")]
    elif fs_position == "Midfielder":
        imitated_player_skills = global_instance.sf_avg_team_strength[(season, team_id, "midfielder")]
    elif fs_position == "Forward":
        imitated_player_skills = global_instance.sf_avg_team_strength[(season, team_id, "attacker")]
    else:
        raise ValueError(f"Player FS position [{fs_position}] not a valid position for player skills imitation")

    if not is_valid_n_float_list(imitated_player_skills):
        raise ValueError(f"Unexpected shape of player skills variable: {imitated_player_skills}")
    return imitated_player_skills


def add_imitated_player_skills(season, team_id, team_sf_players_skills, lineup_fs_positions):
    lineup_length = len(lineup_fs_positions)

    possible_fs_positions = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
    expected_position_occurrences = {"Defender": 4, "Midfielder": 4, "Forward": 2}

    # Sort the possible positions to increase the chance the imputed positions will actually be the ones missing
    sorted_possible_fs_positions = sort_positions_by_missing_count(possible_fs_positions, expected_position_occurrences,
                                                                   lineup_fs_positions)
    if "Goalkeeper" in lineup_fs_positions:
        sorted_possible_fs_positions.remove("Goalkeeper")  # avoid imputing another goalkeeper skills if already present

    while lineup_length < 11:
        for position in sorted_possible_fs_positions:
            print(f"Adding imitated player skill (for position [{position}]), because not enough FS/SOFIFA matches")

            imitated_player_skills = get_imitated_player_skills(season, team_id, position)  # get imitated skills

            # Dependency of the insertion index on position - try to maintain the positions order for consistency
            if position == "Goalkeeper":
                team_sf_players_skills.insert(0, imitated_player_skills)  # goalkeeper first
            elif position == "Defender":
                team_sf_players_skills.insert(1, imitated_player_skills)  # then defenders
            elif position == "Midfielder":
                team_sf_players_skills.insert(6, imitated_player_skills)  # midfielders somewhere in the middle
            elif position == "Forward":
                team_sf_players_skills.insert(10, imitated_player_skills)  # forwards last
            else:
                raise ValueError(f"Unsupported FS player position [{position}] found when adding imitated player skill")

            lineup_length += 1
            if lineup_length == 11:
                break

    if lineup_length != 11:
        raise ValueError(f"Internal error occurred when adding imitated player skills: resulting lineup length equals "
                         f"to {lineup_length}, but 11 expected")

    if not is_valid_mxn_float_list(team_sf_players_skills):
        raise ValueError(f"Unexpected shape of team player skills variable: {team_sf_players_skills}")
    return team_sf_players_skills


def balance_goalkeeper_and_outfield_player_skills(season, team_id, team_skills):
    for idx, player in enumerate(team_skills):
        # Compute normalized skill means
        outfield_mean = sum(player[i] for i in range(29)) / 29  # avg of indices [0,28]
        goalkeeper_mean = sum(player[i] for i in range(29, 33)) / 5  # avg of indices [29,33]

        # Check and correct based on  relative mean strengths
        if idx == 0 and goalkeeper_mean < outfield_mean:  # replace probable outfield player with a goalkeeper
            imitated_gk_skills = get_imitated_player_skills(season, team_id, "Goalkeeper")
            print(f"Balancing player skills - replacing outfield player with goalkeeper: {team_skills[idx]}\n->\n{imitated_gk_skills}")
            team_skills[idx] = imitated_gk_skills

        elif idx > 0 and goalkeeper_mean > outfield_mean:  # replace probable goalkeeper with an outfield player
            if idx in [1, 2, 3, 4]:
                imitated_gk_skills = get_imitated_player_skills(season, team_id, "Defender")
                print(f"Balancing player skills - replacing goalkeeper with defender: {team_skills[idx]}\n->\n{imitated_gk_skills}")
                # TODO code: Results always in replacing goalkeeper with a defender, never with midfielders or forwards
                team_skills[idx] = imitated_gk_skills
            elif idx in [5, 6, 7, 8]:
                imitated_gk_skills = get_imitated_player_skills(season, team_id, "Midfielder")
                print(
                    f"Balancing player skills - replacing goalkeeper with midfielder: {team_skills[idx]}\n->\n{imitated_gk_skills}")
                team_skills[idx] = imitated_gk_skills
            elif idx in [9, 10]:
                imitated_gk_skills = get_imitated_player_skills(season, team_id, "Forward")
                print(
                    f"Balancing player skills - replacing goalkeeper with forward: {team_skills[idx]}\n->\n{imitated_gk_skills}")
                team_skills[idx] = imitated_gk_skills
            else:
                raise ValueError(f"Found player skills list containing more than 11 players (>={idx + 1}, concretely)")

    return team_skills


def sort_positions_by_missing_count(possible_fs_positions, expected_position_occurrences, lineup_fs_positions):
    # Count occurrences in the lineup
    actual_counts = Counter(lineup_fs_positions)

    # Compute the difference (expected - actual)
    position_diff = {
        pos: expected_position_occurrences.get(pos, 0) - actual_counts.get(pos, 0)
        for pos in possible_fs_positions
    }

    # Sort by missing count (desc.)
    sorted_positions = sorted(possible_fs_positions, key=lambda pos: position_diff[pos], reverse=True)

    return sorted_positions


def get_most_frequent_string(list_of_strings):
    if not list_of_strings:
        raise ValueError(f"Cannot determine most frequent string from empty list of strings")

    counter = Counter(list_of_strings)
    return counter.most_common(1)[0][0]


def is_match_within_days(curr_datetime, match_datetime, n):
    time_difference = curr_datetime - match_datetime

    return time_difference <= timedelta(days=n)


def is_valid_mxn_float_list(variable, m=11, n=34):
    if not isinstance(variable, list):  # check if list
        return False

    if len(variable) != m:  # check if "m" elements
        return False

    return all(
        isinstance(inner_list, list) and len(inner_list) == n and all(isinstance(x, float) for x in inner_list)
        for inner_list in variable
    )  # check if each "m" element is a list and only contains "n" floats (no nested lists, for instance)


def is_valid_n_float_list(variable, n=34):
    if not isinstance(variable, list) or len(variable) != n:  # check if list of exactly "n" elements
        return False

    return all(isinstance(x, float) for x in variable)  # check if each element is a single float value


# TODO: Minor adjustment possible: revise the normalization constants - from higher pool of competitions
def min_max_scaling_with_clipping(value, max_value):
    scaled_value = value / max_value
    return np.clip(scaled_value, settings.ALMOST_ZERO, settings.ALMOST_ONE)


def separate_normalize_gk_and_outfield_skills(data):
    gk_data = data[:, 0, :]  # shape: (num_samples, 34)
    outfield_data = data[:, 1:, :]  # shape: (num_samples, 10, 34)

    # 1. and 99. percentiles (used instead of min and max to avoid outlier influence - now should be already mitigated!)
    gk_p1 = np.array(settings.TEAM_STRENGTH_NORM_PERCENTILES["gk_p1"])
    gk_p99 = np.array(settings.TEAM_STRENGTH_NORM_PERCENTILES["gk_p99"])
    outfield_p1 = np.array(settings.TEAM_STRENGTH_NORM_PERCENTILES["outfield_p1"])
    outfield_p99 = np.array(settings.TEAM_STRENGTH_NORM_PERCENTILES["outfield_p99"])

    gk_iqr = gk_p99 - gk_p1
    outfield_iqr = outfield_p99 - outfield_p1

    # Robust min-max scaling (clipping extreme values to avoid outlier influence - implemented balancing mitigates it!)
    gk_data_norm = (np.clip(gk_data, gk_p1, gk_p99) - gk_p1) / (gk_iqr + 1e-6)
    outfield_data_norm = (np.clip(outfield_data, outfield_p1, outfield_p99) - outfield_p1) / (outfield_iqr + 1e-6)

    # Combine back normalized values
    normalized_data = np.zeros_like(data)
    normalized_data[:, 0, :] = gk_data_norm
    normalized_data[:, 1:, :] = outfield_data_norm

    return normalized_data


def get_categorical_features_maps(regular_matches_sorted):
    # Home team ID + Away team ID
    team_id_to_name = {}
    for m in regular_matches_sorted:
        home_team_name = f"{m.comp.name}_{m.home_team.name}"  # team-id -> "comp_team-name"
        team_id_to_name[m.home_team.id] = home_team_name
        away_team_name = f"{m.comp.name}_{m.away_team.name}"
        team_id_to_name[m.away_team.id] = away_team_name
    unique_regular_team_ids = sorted(team_id_to_name.keys())  # sorting ensures consistent indexing
    team_id_map = {team_id: idx for idx, team_id in enumerate(unique_regular_team_ids)}  # map to [0, 518)

    # Comp ID
    comp_id_to_name = {}
    for m in regular_matches_sorted:
        comp_name = "Bundesliga AUT" if m.comp.country == "Austria" and m.comp.name == "Bundesliga" else m.comp.name
        comp_id_to_name[m.comp.id] = comp_name
    unique_comp_ids = sorted(comp_id_to_name.keys())  # sorting ensures consistent indexing
    comp_id_map = {comp_id: idx for idx, comp_id in enumerate(unique_comp_ids)}  # map to [0, 24)

    return team_id_map, comp_id_map
