"""
feature_utils.py
"""

import numpy as np
import utils as ut
from globals import Global
from settings import INIT_ELO, WINNER_TEAM_ID_CODE_FOR_DRAW, FIRST_SEASON, LAST_SEASON, SOG_NORM_COEFFICIENT, \
    GOALS_NORM_COEFFICIENT, TOTAL_SHOTS_NORM_COEFFICIENT, SHOTS_IN_BOX_NORM_COEFFICIENT, CORNER_KICKS_NORM_COEFFICIENT,\
    MATCH_LOAD_NORM_COEFFICIENT, ALMOST_ZERO, ALMOST_ONE, CSV_PLAYERS_PATH
from player_stats_loader import get_player_stats_for_team, tmp_try_find_team_players_in_so_fifa_csvs_by_lineup_name

ELO_C = 10.0
ELO_D = 400.0
ELO_K = 32.0


# home_match_load_per_day_last_10_days, home_match_load_per_day_last_25_days
# away_match_load_per_day_last_10_days, away_match_load_per_day_last_25_days
def get_match_load_per_day_last_n(curr_match, n, home_away):
    new_curr_match = curr_match

    num_matches = 0

    if home_away == "home":
        while True:
            prev_match = ut.get_previous_match(new_curr_match, curr_match.home_team.id, same_comp=False,
                                               same_season=False, regular=False)

            if prev_match is None:
                break

            if ut.is_match_within_days(curr_match.datetime, prev_match.datetime, n):
                num_matches += 1
                new_curr_match = prev_match
            else:
                break

    elif home_away == "away":
        while True:
            prev_match = ut.get_previous_match(new_curr_match, curr_match.away_team.id, same_comp=False,
                                               same_season=False, regular=False)

            if prev_match is None:
                break

            if ut.is_match_within_days(curr_match.datetime, prev_match.datetime, n):
                num_matches += 1
                new_curr_match = prev_match
            else:
                break
    else:
        raise Exception("The \"home_away\" parameter set to a wrong value.")

    match_load = float(num_matches) / n
    return normalize_match_loads(match_load)


# home_team_points_avg_last_5, home_team_points_avg_last_20
# away_team_points_avg_last_5, away_team_points_avg_last_20
def get_avg_points_last_n(curr_match, n, home_away):  # "N" 5 or 20 probably
    # Check if wanted for currently HOME or currently AWAY team
    if home_away == "home":
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.home_team.id, same_comp=False,
                                                   same_season=False, regular=False)
        team_id = curr_match.home_team.id
    elif home_away == "away":
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.away_team.id, same_comp=False,
                                                   same_season=False, regular=False)
        team_id = curr_match.away_team.id
    else:
        raise Exception("The \"home_away\" parameter set to a wrong value.")

    total_points = 0
    total_none_values = 0
    for match in last_n_matches:
        if match is None:
            total_none_values += 1

        # In each match the wanted team was either HOME or AWAY
        else:
            if team_id == match.home_team.id:
                total_points += match.home_team_points
            elif team_id == match.away_team.id:
                total_points += match.away_team_points
            else:
                raise Exception(
                    "The \"team_id\" parameter equals neither to home or away team in one of the previous matches.")

    # Avoid division by zero
    if n - total_none_values == 0:
        return 0

    avg_points = float(total_points / (n - total_none_values))
    return normalize_points(avg_points)


# home_avg_goals_last_5, home_avg_goals_last_20
# away_avg_goals_last_5, away_avg_goals_last_20
def get_avg_goals_last_n(curr_match, n, home_away):  # "N" 5 or 20 probably
    # Check if wanted for currently HOME or currently AWAY team
    if home_away == "home":

        # Last N matches of a home team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.home_team.id, same_comp=False,
                                                   same_season=False, regular=False)
        team_id = curr_match.home_team.id

    elif home_away == "away":

        # Last N matches of an away team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.away_team.id, same_comp=False,
                                                   same_season=False, regular=False)
        team_id = curr_match.away_team.id

    else:
        raise Exception("The \"home_away\" parameter set to a wrong value.")

    total_goals = 0
    total_none_values = 0
    for match in last_n_matches:
        if match is None:
            total_none_values += 1

        # In each match the wanted team was either HOME or AWAY
        else:
            if team_id == match.home_team.id:
                total_goals += match.home_team_goals
            elif team_id == match.away_team.id:
                total_goals += match.away_team_goals
            else:
                raise Exception(
                    "The \"team_id\" parameter equals neither to home or away team in one of the previous matches.")

    # Avoid division by zero
    if n - total_none_values == 0:
        return 0

    avg_goals = float(total_goals / (n - total_none_values))
    return normalize_goals(avg_goals)


# home_avg_shots_on_target_last_5, home_avg_shots_on_target_last_20
# away_avg_shots_on_target_last_5, away_avg_shots_on_target_last_20
# ...
def get_avg_stat_value_last_n(curr_match, n, home_away, stat_name):  # "N" 5 or 20 probably
    # For stats, irregular matches allowed too, but if there are no stats (-1) for any of them, ...
    # ...this match value is excluded from the avg_values calculation

    # Check if wanted for currently HOME or currently AWAY team
    if home_away == "home":

        # Last N matches of a home team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.home_team.id, same_comp=False,
                                                   same_season=False, regular=False)
        team_id = curr_match.home_team.id

    elif home_away == "away":

        # Last N matches of an away team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.away_team.id, same_comp=False,
                                                   same_season=False, regular=False)
        team_id = curr_match.away_team.id

    else:
        raise Exception("The \"home_away\" parameter set to a wrong value.")

    total_value = 0
    total_none_values = 0
    for match in last_n_matches:
        if match is None:
            total_none_values += 1

        # In each match the wanted team was either HOME or AWAY
        else:
            new_value = 0
            if team_id == match.home_team.id:

                if stat_name == "Shots on Goal":
                    new_value += match.home_team_shots_on_target
                elif stat_name == "Total Shots":
                    new_value += match.home_team_total_shots
                elif stat_name == "Shots insidebox":
                    new_value += match.home_team_shots_inside_box
                elif stat_name == "Corner Kicks":
                    new_value += match.home_team_corner_kicks
                elif stat_name == "Ball Possession":
                    new_value += match.home_team_ball_possession
                elif stat_name == "Passes %":
                    new_value += match.home_team_passes_acc

            elif team_id == match.away_team.id:

                if stat_name == "Shots on Goal":
                    new_value += match.away_team_shots_on_target
                elif stat_name == "Total Shots":
                    new_value += match.away_team_total_shots
                elif stat_name == "Shots insidebox":
                    new_value += match.away_team_shots_inside_box
                elif stat_name == "Corner Kicks":
                    new_value += match.away_team_corner_kicks
                elif stat_name == "Ball Possession":
                    new_value += match.away_team_ball_possession
                elif stat_name == "Passes %":
                    new_value += match.away_team_passes_acc

            else:
                raise Exception(
                    "The \"team_id\" parameter equals neither to home or away team in one of the previous matches.")

            # This is the correction of case an irregular match misses a stats value (-1)
            if new_value != -1:
                # TODO: Check this functionality via debug
                total_value += new_value
            else:
                total_none_values += 1

    # Avoid division by zero
    if n - total_none_values == 0:
        return 0

    avg_value = float(total_value / (n - total_none_values))

    if stat_name == "Shots on Goal":
        return normalize_sog(avg_value)
    elif stat_name == "Total Shots":
        return normalize_total_shots(avg_value)
    elif stat_name == "Shots insidebox":
        return normalize_shots_in_box(avg_value)
    elif stat_name == "Corner Kicks":
        return normalize_corner_kicks(avg_value)
    elif stat_name == "Ball Possession":
        return avg_value
    elif stat_name == "Passes %":
        return avg_value


# home_avg_goals_scored_home_last_5, home_avg_goals_scored_home_last_20
# away_avg_goals_scored_away_last_5, away_avg_goals_scored_away_last_20
# home_avg_goals_conceded_home_last_5, home_avg_goals_conceded_home_last_20
# away_avg_goals_conceded_away_last_5, away_avg_goals_conceded_away_last_20
def get_avg_goals_scored_conceded_home_or_away_last_n(curr_match, n, home_away, scored_conceded):
    # Check if wanted for currently HOME or currently AWAY team
    if home_away == "home":

        # Last N home matches of a home team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.home_team.id, "home", same_comp=False,
                                                   same_season=False, regular=False)
        team_id = curr_match.home_team.id

    elif home_away == "away":

        # Last N away matches of an away team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.away_team.id, "away", same_comp=False,
                                                   same_season=False, regular=False)
        team_id = curr_match.away_team.id

    else:
        raise Exception("The \"home_away\" parameter set to a wrong value.")

    total_goals = 0
    total_none_values = 0
    for match in last_n_matches:
        if match is None:
            total_none_values += 1

        else:
            # Counting number of scored goals of a team
            if scored_conceded == "scored":

                # In each match the wanted team was either HOME or AWAY - if HOME then count HOME team goals etc.
                if team_id == match.home_team.id:
                    total_goals += match.home_team_goals
                elif team_id == match.away_team.id:
                    total_goals += match.away_team_goals
                else:
                    raise Exception(
                        "The \"team_id\" param equals neither to home or away team in one of the previous matches.")

            # Counting number of conceded goals of a team
            elif scored_conceded == "conceded":

                # In each match the wanted team was either HOME or AWAY - if HOME then count AWAY team goals etc.
                if team_id == match.home_team.id:
                    total_goals += match.away_team_goals
                elif team_id == match.away_team.id:
                    total_goals += match.home_team_goals
                else:
                    raise Exception(
                        "The \"team_id\" param equals neither to home or away team in one of the previous matches.")

            else:
                raise ValueError("The \"scored_conceded\" parameter equals neither to \"scored\" or \"conceded\".")

    # Avoid division by zero
    if n - total_none_values == 0:
        return 0

    avg_goals_scored_conceded = float(total_goals / (n - total_none_values))
    return normalize_goals(avg_goals_scored_conceded)


# home_elo_rating, away_elo_rating
def calculate_elo_for_both_teams(curr_match):
    # Get previous match of currently HOME team and find out if it was home or away team in that previous match
    # Then get its ELO in the previous match
    home_team_prev_match = ut.get_previous_match(curr_match, curr_match.home_team.id, same_comp=False,
                                                 same_season=False, regular=False)
    if home_team_prev_match is None:
        home_team_prev_match_elo = INIT_ELO
    else:
        if home_team_prev_match.home_team.id == curr_match.home_team.id:
            home_team_prev_match_elo = home_team_prev_match.home_elo_before_match_not_normalized
        elif home_team_prev_match.away_team.id == curr_match.home_team.id:
            home_team_prev_match_elo = home_team_prev_match.away_elo_before_match_not_normalized
        else:
            raise Exception("Current home team not found in its previous match. This should never happen.")

    # Get previous match of currently AWAY team and find out if it was home or away team in that previous match
    # Then get its ELO in the previous match
    away_team_prev_match = ut.get_previous_match(curr_match, curr_match.away_team.id, same_comp=False,
                                                 same_season=False, regular=False)
    if away_team_prev_match is None:
        away_team_prev_match_elo = INIT_ELO
    else:
        if away_team_prev_match.home_team.id == curr_match.away_team.id:
            away_team_prev_match_elo = away_team_prev_match.home_elo_before_match_not_normalized
        elif away_team_prev_match.away_team.id == curr_match.away_team.id:
            away_team_prev_match_elo = away_team_prev_match.away_elo_before_match_not_normalized
        else:
            raise Exception("Current away team not found in its previous match. This should never happen.")

    expected_score_home_team = 1.0 / (1.0 + (ELO_C ** ((away_team_prev_match_elo - home_team_prev_match_elo) / ELO_D)))
    expected_score_away_team = 1.0 - expected_score_home_team

    if home_team_prev_match is None or home_team_prev_match.winner_team_id == WINNER_TEAM_ID_CODE_FOR_DRAW:
        alpha_home = 0.5
    elif home_team_prev_match.winner_team_id == curr_match.home_team.id:
        alpha_home = 1
    else:
        alpha_home = 0

    if away_team_prev_match is None or away_team_prev_match.winner_team_id == WINNER_TEAM_ID_CODE_FOR_DRAW:
        alpha_away = 0.5
    elif away_team_prev_match.winner_team_id == curr_match.away_team.id:
        alpha_away = 1
    else:
        alpha_away = 0

    home_team_new_elo = home_team_prev_match_elo + ELO_K * (alpha_home - expected_score_home_team)
    away_team_new_elo = away_team_prev_match_elo + ELO_K * (alpha_away - expected_score_away_team)

    curr_match.home_elo_before_match_not_normalized = home_team_new_elo
    curr_match.away_elo_before_match_not_normalized = away_team_new_elo

    return normalize_elo(home_team_new_elo), normalize_elo(away_team_new_elo)


# home_team_strength, away_team_strength
def calculate_team_strength(curr_match, team_id):
    global_instance = Global.get_instance()

    # Get FS lineup
    if team_id == curr_match.home_team.id:
        team_fs_lineup = curr_match.home_fs_team_lineup
    elif team_id == curr_match.away_team.id:
        team_fs_lineup = curr_match.away_fs_team_lineup
    else:
        raise ValueError(f"Team ID {team_id} matches neither the home team {curr_match.home_team.name} "
                         f"({curr_match.home_team.id}) or the away team {curr_match.away_team.name} "
                         f"({curr_match.away_team.id})")

    # No complete lineup found
    if team_fs_lineup is None or len(team_fs_lineup) == 0:
        print(f"None team FS lineup list found, but 11 expected (match "
              f"{curr_match.home_team.name} - {curr_match.away_team.name} played at {curr_match.datetime})")
        return []
    if len(team_fs_lineup) != 11:
        raise ValueError(f"Team FS lineup list of length {len(team_fs_lineup)} found, but 11 expected (match "
                         f"{curr_match.home_team.name} - {curr_match.away_team.name} played at {curr_match.datetime})")
        # print(f"Team lineup list of length {len(team_lineup)} found, but 11 expected (match "
        #       f"{curr_match.home_team.name} - {curr_match.away_team.name} played at {curr_match.datetime})")
        # return []

    # Iterate over FS lineup players
    for fs_player in team_fs_lineup:
        fs_player['fs_birthday'] = fs_player['fs_birthday'].replace(hour=0)  # set hours=0 to match SF datetime formats

        if fs_player['fs_birthday'] in global_instance.sofifa_players_by_dob:
            # Get sofifa player matching the FS player's date of birth
            sf_players_with_same_dob = global_instance.sofifa_players_by_dob[fs_player['fs_birthday']]
            if len(sf_players_with_same_dob) == 0:
                raise ValueError(f"No sofifa players were found for the birth date {fs_player['fs_birthday']} "
                                 f"of FS player {fs_player['fs_known_as']}")
        else:
            print(f"\t\tWarning! FS player {fs_player['fs_known_as']} not found in SOFIFA dob dict. Skipping...")
            continue

        sf_player_id, sf_player_name, sf_player_full_name = ut.match_fs_player_to_sf_players_alternative(
            fs_player, sf_players_with_same_dob)  # TODO: Continue here

        # TODO: Store successfully matched SF players in list and print out how many of them we finally got for the team strength calculation...

    """
    # Get players in current comp season team roster
    team = ut.get_team_if_exists(team_id)
    
    # TODO: Note that not every team has rating in a comp season
    team_rating_in_comp_season = team.rating_comp_season[curr_match.comp.name][str(curr_match.season)]

    team_players_stats_in_comp_season = team.player_stats_comp_season[curr_match.comp.name][str(curr_match.season)]
    print(team_players_stats_in_comp_season)

    # Get stats about those players which are in the current match lineup
    team_lineup_info = []
    for player in team_lineup:
        print(player)
        p_id, p_name, _ = player  # (ID, name, position)

        # Match player ID from match lineups with the ID in team player stats in comp season
        # TODO: What if a player from the match lineup (p_id) is not in the players list for team's comp season?
        player_stats_in_comp_season = [x for x in team_players_stats_in_comp_season if x['id'] == p_id]

        if len(player_stats_in_comp_season) == 0:
            print(f"Player {str(p_id)}:{p_name} was not found in the {team.name} team's comp season players list...")
            continue
        elif len(player_stats_in_comp_season) > 1:
            raise ValueError(f"Found more players matching {str(p_id)}:{p_name}. This should not happen.")
        else:
            player_stats_in_comp_season = player_stats_in_comp_season[0]

        # Get (full_name, dob, rating)
        team_lineup_info.append((
            player_stats_in_comp_season['firstname'] + " " + player_stats_in_comp_season['lastname'],
            player_stats_in_comp_season['birth_date'],
            player_stats_in_comp_season['rating'],
            player_stats_in_comp_season['position']
        ))

    print(f"{len(team_lineup_info)} players matched for team strength calculation for team {team.name} "
          f"(match {str(curr_match.id)})")  # TODO: Check how many players are actually matched here...
    # if len(team_lineup_info) != 11:
        # raise ValueError(f"Team lineup info list of length {len(team_lineup)}, but 11 expected")

    player_ratings = [z for (x, y, z, _) in team_lineup_info]
    player_positions = [z for (x, y, z) in team_lineup]
    # TODO: Note that might end up e.g. with 11 positions, but only 3 ratings...

    # Get player stats from CSV
    team_players_individual_stats = get_player_stats_for_team(team_lineup_info, team_rating_in_comp_season,
                                                              curr_match, CSV_PLAYERS_PATH)

    # Calculate team strength vector
    team_strength_vector = ut.combine_players_stats_in_team_strength(team_players_individual_stats, player_ratings,
                                                                     player_positions, mode="basic")
    return team_strength_vector
    """
    # player_names = [p_name for (p_id, p_name, _) in team_lineup]
    # tmp_try_find_team_players_in_so_fifa_csvs_by_lineup_name(curr_match, player_names, CSV_PLAYERS_PATH)
    # TODO: Uncomment for Phase 1 testing (and finish the implementation inside; and modify it - CSV caching)

    return []


def normalize_season(season):
    normalized_season = (season - FIRST_SEASON) / (LAST_SEASON - FIRST_SEASON)
    return float(max(0, min(1, normalized_season)))


def normalize_elo(elo, min_elo=1000, max_elo=2000):
    normalized_elo = (elo - min_elo) / (max_elo - min_elo)
    return float(max(0, min(1, normalized_elo)))


def normalize_points(points):
    normalized_points = points / 3.0

    if normalized_points == 0.0:
        return ALMOST_ZERO
    if normalized_points == 1.0:
        return ALMOST_ONE
    return normalized_points


def normalize_goals(goals):
    return ut.min_max_scaling_with_clipping(goals, GOALS_NORM_COEFFICIENT)


def normalize_sog(sog):
    return ut.min_max_scaling_with_clipping(sog, SOG_NORM_COEFFICIENT)


def normalize_total_shots(total_shots):
    return ut.min_max_scaling_with_clipping(total_shots, TOTAL_SHOTS_NORM_COEFFICIENT)


def normalize_shots_in_box(shots_in_box):
    return ut.min_max_scaling_with_clipping(shots_in_box, SHOTS_IN_BOX_NORM_COEFFICIENT)


def normalize_corner_kicks(corner_kicks):
    return ut.min_max_scaling_with_clipping(corner_kicks, CORNER_KICKS_NORM_COEFFICIENT)


def normalize_match_loads(match_loads):
    return ut.min_max_scaling_with_clipping(match_loads, MATCH_LOAD_NORM_COEFFICIENT)


def normalized_hour_month_cyclic(cyclic_value):
    return (cyclic_value + 1) / 2
