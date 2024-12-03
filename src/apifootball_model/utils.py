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


def distribute_matches_into_rounds(matches):
    rounds = []
    current_round = []
    teams_in_current_round = set()

    for match in matches:
        home_team_id = match.home_team.id
        away_team_id = match.away_team.id

        if home_team_id in teams_in_current_round or away_team_id in teams_in_current_round:

            # Finish current round and start new one
            rounds.append(current_round)
            current_round = []
            teams_in_current_round = set()

        # Add match to current round
        current_round.append(match)
        teams_in_current_round.add(home_team_id)
        teams_in_current_round.add(away_team_id)

    # Append the last round if not empty
    if current_round:
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


def normalize_name(name):
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')  # to ASCII - remove accents
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s]', '', name)  # remove all non-alphanumeric characters
    name = ' '.join(name.split())  # remove whitespaces
    return name


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


def get_fs_match_lineups(curr_match):
    home_team_fs_lineup = away_team_fs_lineup = [], []

    if len(curr_match.comp.regular_round_keywords) == 0:  # skip for irregular matches
        print(f"Match between {curr_match.home_team.name} and {curr_match.away_team.name} ({curr_match.datetime})"
              f" is irregular - no FS team lineup")
        return home_team_fs_lineup, away_team_fs_lineup

    # Home team
    home_team_fs_players_in_comp_season = [x['fs_players'] for x in curr_match.home_team.players_in_regular_comp_season
                                           if curr_match.comp == x['comp'] and curr_match.season == x['season']]
    if len(home_team_fs_players_in_comp_season) == 0:
        print(f"Match between {curr_match.home_team.name} and {curr_match.away_team.name} ({curr_match.datetime})"
              f" is regular, but no AF team lineup was found for the home team - no FS team lineups")
        return home_team_fs_lineup, away_team_fs_lineup

    if len(home_team_fs_players_in_comp_season) > 1:
        raise ValueError(f"Multiple FS home team lineups found for match between {curr_match.home_team.name} and "
                         f"{curr_match.away_team.name} ({curr_match.datetime}) - ERROR!")

    if len(curr_match.home_team_lineup) == 0:
        print(f"WARNING!!! - No AF home team lineup for match between {curr_match.home_team.name} and "
              f"{curr_match.away_team.name} ({curr_match.datetime})")

    else:
        for af_player in curr_match.home_team_lineup:
            matched_fs_player, similarity = match_af_player_to_fs_player_alternative(af_player,
                                                                                     home_team_fs_players_in_comp_season[0])
            if similarity > settings.SIMILARITY_THRESHOLD:
                curr_match.home_fs_team_lineup.append(matched_fs_player)

        if len(curr_match.home_fs_team_lineup) < settings.MINIMUM_MATCHED_PLAYERS:
            raise ValueError(f"There were only {len(curr_match.home_fs_team_lineup)} matched home team FS player, "
                             f"but the minimum required is {settings.MINIMUM_MATCHED_PLAYERS}")

    # Away team
    away_team_fs_players_in_comp_season = [x['fs_players'] for x in
                                           curr_match.away_team.players_in_regular_comp_season
                                           if curr_match.comp == x['comp'] and curr_match.season == x['season']]

    if len(away_team_fs_players_in_comp_season) == 0:  # skip for irregular matches
        print(f"Match between {curr_match.home_team.name} and {curr_match.away_team.name} ({curr_match.datetime})"
              f" is regular, but no AF team lineup was found for the away team - no FS team lineups")
        return home_team_fs_lineup, away_team_fs_lineup

    if len(away_team_fs_players_in_comp_season) > 1:
        raise ValueError(f"Multiple FS away team lineups found for match between {curr_match.home_team.name} and "
                         f"{curr_match.away_team.name} ({curr_match.datetime}) - ERROR!")

    if len(curr_match.away_team_lineup) == 0:
        print(f"WARNING!!! - No AF away team lineup for match between {curr_match.home_team.name} and "
              f"{curr_match.away_team.name} ({curr_match.datetime})")

    else:
        for af_player in curr_match.away_team_lineup:
            matched_fs_player, similarity = match_af_player_to_fs_player_alternative(af_player,
                                                                                     away_team_fs_players_in_comp_season[0])

            if similarity > settings.SIMILARITY_THRESHOLD:
                curr_match.away_fs_team_lineup.append(matched_fs_player)

        if len(curr_match.away_fs_team_lineup) < settings.MINIMUM_MATCHED_PLAYERS:
            raise ValueError(f"There were only {len(curr_match.away_fs_team_lineup)} matched away team FS player, "
                             f"but the minimum required is {settings.MINIMUM_MATCHED_PLAYERS}")

    # Summary print
    print(f"Successfully matched {len(curr_match.home_fs_team_lineup)} home team players and "
          f"{len(curr_match.away_fs_team_lineup)} away team players!")
    # TODO: Minor adj.: note that there might be duplicates


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

    if highest_similarity > settings.SIMILARITY_THRESHOLD:
        print(
            f"AF player [{af_player[1]}] matched to FS player 'known as' name [{best_fs_match['fs_known_as']}] "
            f"(similarity={str(highest_similarity)})")

    return best_fs_match, highest_similarity


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
