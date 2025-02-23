"""
utils.py
"""

import numpy as np
from sklearn.decomposition import PCA
import re
import unicodedata
import difflib
from rapidfuzz import fuzz
import settings
import time
import requests
from globals import Global
from settings import MAX_MATCH_HISTORY_TO_CHECK_LOW, CSV_CATEGORIES
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


def get_sf_player_data(match_datetime, sf_player_id, output_team_skills_dict, team_season_info):
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
        print(f"There are no available player CSV files within the timedelta range for player {sf_player_id}. Skip...")
        return output_team_skills_dict

    # TODO: Debug print
    print(f"{len(available_player_csvs_sorted_by_timedelta_to_match)} available CSV files found...")

    skills_processed = set()  # keep track of already processed player skills

    # Loop because the first list element (closest timedelta to match) may not contain all skill values...
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

        # Get SOFIFA player skill values
        for skill in settings.PLAYER_SKILLS:
            if skill in skills_processed:  # skill already processed in the previous iteration
                continue

            value = sf_player_data.get(skill, -1)  # get skill value

            if value == -1:  # missing skill value
                negative_value_found = True
                continue

            # Assign skill to skill category ("attacking", "power", "mentality", ...)
            skill_category = settings.SKILL_TO_CATEGORY.get(skill, None)
            if skill_category is None:
                raise ValueError(f"Skill [{skill}] not found in any skill category - should never happen!")

            # Check relevancy between player's position categories and the current skill's category
            relevant_positions = settings.PLAYER_CATEGORY_RELEVANCE[skill_category]
            if any(pos_cat in relevant_positions for pos_cat in player_position_categories):
                output_team_skills_dict[skill].append(value)  # player position is relevant for this category - append
                skills_processed.add(skill)  # keep track of already processed skills for the player
            else:
                pass  # player position not relevant for this category - skip (e.g. "CB" player and "goalkeeping" cat.)

        if not negative_value_found:
            break  # if no -1 values found, end getting skills

    # Handle possibly missing GK skills data - check if at least one value for each skill in the "goalkeeping" category
    gk_skills = CSV_CATEGORIES['goalkeeping']
    have_gk_data = any(output_team_skills_dict[sk] for sk in gk_skills if len(output_team_skills_dict[sk]) > 0)

    if not have_gk_data:
        for gk_skill in gk_skills:
            print(f"Imputing value for missing GK skill [{gk_skill}]...")
            team = get_team_if_exists(team_id)

            # If no GK skill value, impute it
            if gk_skill == "gk_diving":
                imputed_value = team.avg_gk_diving[season]
            elif gk_skill == "gk_handling":
                imputed_value = team.avg_gk_handling[season]
            elif gk_skill == "gk_kicking":
                imputed_value = team.avg_gk_kicking[season]
            elif gk_skill == "gk_positioning":
                imputed_value = team.avg_gk_positioning[season]
            elif gk_skill == "gk_reflexes":
                imputed_value = team.avg_gk_reflexes[season]
            else:
                raise ValueError(f"Found non-existing GK skill name [{gk_skill}]")

            output_team_skills_dict[gk_skill].append(imputed_value)

    return output_team_skills_dict


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


def match_af_team_to_fs_team_alternative(af_team_name, fs_teams_in_comp_season):
    normalized_af_name = normalize_name(af_team_name)  # normalize AF team name

    best_fs_match = None
    highest_similarity = 0.0

    for fs_team in fs_teams_in_comp_season['fs_teams']:
        normalized_fs_clean_name = normalize_name(fs_team.name)  # normalize FS team name

        similarity = fuzz.ratio(normalized_af_name, normalized_fs_clean_name)  # similarity

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_fs_match = fs_team

    print(
        f"AF team [{af_team_name}] matched to FS team [{best_fs_match['name']}] (similarity={str(highest_similarity)})")

    return best_fs_match['id'], best_fs_match['name']


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

            if similarity > settings.SIMILARITY_THRESHOLD_AF_FS:
                curr_match.home_fs_team_lineup.append(matched_fs_player)
            else:
                print(f"WARNING! - Found AF player [{af_player[1]}] match below threshold ({curr_match.home_team.name} "
                      f"vs. {curr_match.away_team.name}, {curr_match.datetime})")

        if len(curr_match.home_fs_team_lineup) < settings.MINIMUM_MATCHED_LINEUP_PLAYERS:
            raise ValueError(f"There were only {len(curr_match.home_fs_team_lineup)} matched home team FS player, "
                             f"but the minimum required is {settings.MINIMUM_MATCHED_LINEUP_PLAYERS}")
        if len(curr_match.home_fs_team_lineup) > 11:
            raise ValueError(f"Home team FS lineup of length {len(curr_match.home_fs_team_lineup)}:"
                             f" [{curr_match.home_fs_team_lineup}] should not contain more than "
                             f"11 players (match between {curr_match.home_team.name} and {curr_match.away_team.name} "
                             f"played at {curr_match.datetime})")

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

            if similarity > settings.SIMILARITY_THRESHOLD_AF_FS:
                curr_match.away_fs_team_lineup.append(matched_fs_player)

        if len(curr_match.away_fs_team_lineup) < settings.MINIMUM_MATCHED_LINEUP_PLAYERS:
            raise ValueError(f"There were only {len(curr_match.away_fs_team_lineup)} matched away team FS player, "
                             f"but the minimum required is {settings.MINIMUM_MATCHED_LINEUP_PLAYERS}")
        if len(curr_match.away_fs_team_lineup) > 11:
            raise ValueError(f"Away team FS lineup of length {len(curr_match.away_fs_team_lineup)}:"
                             f" [{curr_match.away_fs_team_lineup}] should not contain more than "
                             f"11 players (match between {curr_match.home_team.name} and {curr_match.away_team.name} "
                             f"played at {curr_match.datetime})")

    # Summary print
    print(f"Successfully matched {len(curr_match.home_fs_team_lineup)} home team players and "
          f"{len(curr_match.away_fs_team_lineup)} away team players!")
    # TODO: Minor adj.: note that there might be duplicates


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

    time.sleep(2.0)


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

    if highest_similarity > settings.SIMILARITY_THRESHOLD_AF_FS:
        # print(f"AF player [{af_player[1]}] matched to FS player 'known as' name [{best_fs_match['fs_known_as']}] (similarity={str(highest_similarity)})", end='\t')
        print(f"[{af_player[1]}][{best_fs_match['fs_known_as']}]", end='\t')

    return best_fs_match, highest_similarity


def match_fs_player_to_sf_players_alternative(fs_player, sf_players_with_same_dob):
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


def calculate_team_strength_scaled(sf_players_stats, team_season_info):
    global_instance = Global.get_instance()

    team_id, team_name, season = team_season_info

    # If no integer skill values for any of the skills, return average/default vector instead
    for skill in settings.PLAYER_SKILLS:
        if len(sf_players_stats.get(skill, [])) == 0:

            if (team_id, season) in global_instance.sf_avg_team_strength:
                avg_team_strength = global_instance.sf_avg_team_strength[(team_id, season)]
            else:
                avg_team_strength = global_instance.sf_default_team_strength

            print(f"No value found for skill [{skill}] - returning default team strength vector {avg_team_strength}")
            return avg_team_strength

    team_strength_vector = []

    skill_min_val = settings.ALMOST_ZERO
    skill_max_val = 99
    skill_range = skill_max_val - skill_min_val

    for category, skills in CSV_CATEGORIES.items():
        category_values = []
        for skill in skills:
            category_values.extend(sf_players_stats.get(skill, []))

        if len(category_values) == 0:
            raise ValueError(f"Unexpected error occurred in team strength scaled calculation for {team_name} ({season})"
                             f" - no category values found for category {category}")

        mean_val = np.mean(category_values)
        std_val = np.std(category_values) if len(category_values) > 1 else 0.0
        min_val = min(category_values)
        max_val = max(category_values)

        # Normalize values to [0,1]
        mean_val_scaled = (mean_val - skill_min_val) / skill_range
        min_val_scaled = (min_val - skill_min_val) / skill_range
        max_val_scaled = (max_val - skill_min_val) / skill_range
        std_val_scaled = (std_val / skill_range) * 3  # stddev values are usually below 0.2, not in [0,1] (enlarge them)

        team_strength_vector.extend([mean_val_scaled, std_val_scaled, min_val_scaled, max_val_scaled])

    if len(team_strength_vector) != 4 * len(settings.CSV_CATEGORIES):
        raise ValueError("Incomplete team strength vector found.")

    for idx, val in enumerate(team_strength_vector):
        if idx % 4 == 0:  # print only mean values (skip min, max and stddev)
            print(f"{val:.3f}", end='\t')
    print("\n")

    return team_strength_vector


def calculate_team_strength_pca(sf_players_stats, n_components=5, default_vector=None):
    # TODO: Was not tested yet...

    if default_vector is None:  # TODO: After get all API-football data and compute team strength for them, estimate it
        default_vector = [0.0] * n_components

    # Convert dict of skills->list into a matrix: rows=players, cols=skills
    # First find the max number of players across all skills
    player_counts = [len(vals) for vals in sf_players_stats.values()]
    max_players = max(player_counts) if player_counts else 0
    if max_players < 8:  # If fewer than 8 players, data might be incomplete
        return default_vector

    # Build the matrix
    # If any skill is shorter, we can fill missing with np.nan or skip those players
    skill_matrix = []
    for skill in settings.PLAYER_SKILLS:
        vals = sf_players_stats.get(skill, [])
        if len(vals) < max_players:
            # Pad with NaN for missing values
            vals = vals + [np.nan]*(max_players - len(vals))
        skill_matrix.append(vals)

    skill_matrix = np.array(skill_matrix).T  # shape: (max_players, 34)

    # Drop rows with NaN (players missing some skill)
    # Alternatively, you could do imputation here
    mask = ~np.isnan(skill_matrix).any(axis=1)
    skill_matrix = skill_matrix[mask]

    # If after filtering, not enough players remain, return default
    if skill_matrix.shape[0] < 8:
        return default_vector

    # Apply PCA
    pca = PCA(n_components=n_components)
    # If skill values range up to 99, consider scaling them to [0,1] before PCA
    scaled_matrix = skill_matrix / 99.0
    pca_features = pca.fit_transform(scaled_matrix)  # shape: (num_players_filtered, n_components)

    # Aggregate the PCA components across the players (e.g., mean)
    team_pca_mean = np.mean(pca_features, axis=0)

    # Normalize PCA results to [0,1]
    # PCA scores can be negative. One simple approach: shift and scale based on min/max in your dataset.
    # For simplicity, assume team_pca_mean are roughly in [-1,1] after scaling. Adjust as needed.
    # Let's do a min-max scaling based on a guessed range [-2, 2]
    pca_min, pca_max = -2, 2
    pca_range = pca_max - pca_min
    team_strength_pca_scaled = ((team_pca_mean - pca_min) / pca_range).tolist()

    return team_strength_pca_scaled


def get_avg_gk_skill_value(skill_name, team_id, season):
    global_instance = Global.get_instance()

    skill_name = skill_name.lower()

    if skill_name not in global_instance.sf_avg_gk_skills:
        raise ValueError(f"Skill '{skill_name}' not found.")

    # Get value for the team and season
    key = (team_id, season)
    if key in global_instance.sf_avg_gk_skills[skill_name]:
        return global_instance.sf_avg_gk_skills[skill_name][key]

    # Return default value if the team is not found
    if skill_name in global_instance.sf_default_gk_skills:
        return global_instance.sf_default_gk_skills[skill_name]

    raise ValueError(f"Default value for skill '{skill_name}' not found.")


def get_avg_team_strength_vector_scaled(team_id, season):
    global_instance = Global.get_instance()

    # Return the team strength vector if it exists
    if (team_id, season) in global_instance.sf_avg_team_strength:
        return global_instance.sf_avg_team_strength[(team_id, season)]

    # Return the default vector if team_id and season are not found
    return global_instance.sf_default_team_strength


def combine_players_stats_in_team_strength(team_players_individual_stats, player_ratings, player_positions, mode):
    # team_players_individual_stats: list of 11 dicts, each containing 7 categories, each category list of int values
    # player_ratings: list of 11 float values from 0.0 to 10.0
    # player_positions: list of 11 string values ({'G', 'D', 'M', 'F'})
    # TODO: Debug that all these values and list and sorted similarly - that they refer to the same player always

    # Basic: statistical aggregation taking into account player positions
    if mode == "basic":
        # Initialize team strength dictionary
        team_strength = {}

        # For each category, compute mean and standard deviation across all players
        for category in CSV_CATEGORIES.keys():
            # Collect all values from all players for this category
            all_values = []
            for player_stats in team_players_individual_stats:
                values = player_stats.get(category, [])
                # Filter out invalid or missing values (-1)
                valid_values = [v for v in values if v != -1]
                if valid_values:
                    player_mean = np.mean(valid_values)
                    all_values.append(player_mean)
            if all_values:
                category_mean = np.mean(all_values)
                category_std = np.std(all_values)
            else:
                category_mean = -1
                category_std = -1
            team_strength[f"{category}_mean"] = category_mean
            team_strength[f"{category}_std"] = category_std

        # Positional categories mapping
        position_categories = {
            'G': ['goalkeeping'],
            'D': ['defending', 'mentality'],
            'M': ['skill', 'movement', 'mentality'],
            'F': ['attacking', 'skill', 'movement']
        }

        # For each position, compute mean of relevant categories
        for pos, relevant_categories in position_categories.items():
            for category in relevant_categories:
                pos_values = []
                for i, player_stats in enumerate(team_players_individual_stats):
                    if player_positions[i] == pos:
                        values = player_stats.get(category, [])
                        valid_values = [v for v in values if v != -1]
                        if valid_values:
                            player_mean = np.mean(valid_values)
                            pos_values.append(player_mean)
                if pos_values:
                    pos_category_mean = np.mean(pos_values)
                else:
                    pos_category_mean = -1
                team_strength[f"{pos}_{category}_mean"] = pos_category_mean

        return team_strength

    # Weighted: incorporates overall player ratings
    elif mode == "weighted":
        # Normalize player ratings to sum to 1
        ratings = np.array(player_ratings)
        ratings_sum = np.sum(ratings)
        if ratings_sum > 0:
            normalized_ratings = ratings / ratings_sum
        else:
            normalized_ratings = np.ones(len(player_ratings)) / len(player_ratings)

        team_strength = {}

        # For each category, compute weighted mean across all players
        for category in CSV_CATEGORIES.keys():
            weighted_values = []
            weights = []
            for i, player_stats in enumerate(team_players_individual_stats):
                values = player_stats.get(category, [])
                valid_values = [v for v in values if v != -1]
                if valid_values:
                    player_mean = np.mean(valid_values)
                    weight = normalized_ratings[i]
                    weighted_values.append(player_mean * weight)
                    weights.append(weight)
            if weighted_values and weights:
                category_weighted_mean = np.sum(weighted_values)
            else:
                category_weighted_mean = -1
            team_strength[f"{category}_weighted_mean"] = category_weighted_mean

        # Positional categories mapping
        position_categories = {
            'G': ['goalkeeping'],
            'D': ['defending', 'mentality'],
            'M': ['skill', 'movement', 'mentality'],
            'F': ['attacking', 'skill', 'movement']
        }

        # For each position, compute weighted mean of relevant categories
        for pos, relevant_categories in position_categories.items():
            for category in relevant_categories:
                weighted_values = []
                weights = []
                for i, player_stats in enumerate(team_players_individual_stats):
                    if player_positions[i] == pos:
                        values = player_stats.get(category, [])
                        valid_values = [v for v in values if v != -1]
                        if valid_values:
                            player_mean = np.mean(valid_values)
                            weight = normalized_ratings[i]
                            weighted_values.append(player_mean * weight)
                            weights.append(weight)
                if weighted_values and weights:
                    pos_category_weighted_mean = np.sum(weighted_values)
                else:
                    pos_category_weighted_mean = -1
                team_strength[f"{pos}_{category}_weighted_mean"] = pos_category_weighted_mean

        return team_strength

    # PCA: dimensionality reduction
    elif mode == "pca":
        # Create a matrix of player stats
        player_vectors = []
        for player_stats in team_players_individual_stats:
            player_vector = []
            for category in CSV_CATEGORIES.keys():
                values = player_stats.get(category, [])
                # Replace missing values (-1) with np.nan
                valid_values = [v if v != -1 else np.nan for v in values]
                player_vector.extend(valid_values)
            player_vectors.append(player_vector)

        # Convert to numpy array
        player_matrix = np.array(player_vectors, dtype=np.float64)

        # Impute missing values with column means
        col_means = np.nanmean(player_matrix, axis=0)
        inds = np.where(np.isnan(player_matrix))
        player_matrix[inds] = np.take(col_means, inds[1])

        # Apply PCA
        n_components = min(10, player_matrix.shape[1])  # Adjust number of components as needed
        pca = PCA(n_components=n_components)
        pca.fit(player_matrix)
        team_pca_features = pca.transform(player_matrix)

        # Aggregate PCA features across players (e.g., take mean)
        team_strength_vector = np.mean(team_pca_features, axis=0)

        # Create a dictionary to return
        team_strength = {f"pca_component_{i + 1}": team_strength_vector[i] for i in range(len(team_strength_vector))}

        return team_strength

    else:
        raise ValueError(f"Unrecognized mode [{mode}]")


def is_match_within_days(curr_datetime, match_datetime, n):
    time_difference = curr_datetime - match_datetime

    return time_difference <= timedelta(days=n)


# TODO: Minor adjustment possible: revise the normalization constants - from higher pool of competitions
def min_max_scaling_with_clipping(value, max_value):
    scaled_value = value / max_value
    return np.clip(scaled_value, settings.ALMOST_ZERO, settings.ALMOST_ONE)
