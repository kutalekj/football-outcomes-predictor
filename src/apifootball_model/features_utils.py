"""
feature_utils.py
"""


import utils as ut
from settings import INIT_ELO


ELO_C = 10.0
ELO_D = 400.0
ELO_K = 32.0


# home_team_points_avg_last_5, home_team_points_avg_last_20
# away_team_points_avg_last_5, away_team_points_avg_last_20
def get_avg_points_last_n(curr_match, n, home_away):  # "N" 5 or 20 probably
    # Check if wanted for currently HOME or currently AWAY team
    if home_away == "home":
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.home_team_id)
        team_id = curr_match.home_team_id
    elif home_away == "away":
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.away_team_id)
        team_id = curr_match.away_team_id
    else:
        raise Exception("The \"home_away\" parameter set to a wrong value.")

    total_points = 0
    total_none_values = 0
    for match in last_n_matches:
        if match is None:
            total_none_values += 1

        # In each match the wanted team was either HOME or AWAY
        else:
            if team_id == match.home_team_id:
                total_points += match.home_team_points
            elif team_id == match.away_team_id:
                total_points += match.away_team_points
            else:
                raise Exception(
                    "The \"team_id\" parameter equals neither to home or away team in one of the previous matches.")

    # Avoid division by zero
    if n - total_none_values == 0:
        return 0

    return total_points / (n - total_none_values)


# home_avg_goals_last_5, home_avg_goals_last_20
# away_avg_goals_last_5, away_avg_goals_last_20
def get_avg_goals_last_n(curr_match, n, home_away):  # "N" 5 or 20 probably
    # Check if wanted for currently HOME or currently AWAY team
    if home_away == "home":

        # Last N matches of a home team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.home_team_id)
        team_id = curr_match.home_team_id

    elif home_away == "away":

        # Last N matches of an away team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.away_team_id)
        team_id = curr_match.away_team_id

    else:
        raise Exception("The \"home_away\" parameter set to a wrong value.")

    total_goals = 0
    total_none_values = 0
    for match in last_n_matches:
        if match is None:
            total_none_values += 1

        # In each match the wanted team was either HOME or AWAY
        else:
            if team_id == match.home_team_id:
                total_goals += match.home_team_goals
            elif team_id == match.away_team_id:
                total_goals += match.away_team_goals
            else:
                raise Exception(
                    "The \"team_id\" parameter equals neither to home or away team in one of the previous matches.")

    # Avoid division by zero
    if n - total_none_values == 0:
        return 0

    return total_goals / (n - total_none_values)


# home_avg_shots_on_target_last_5, home_avg_shots_on_target_last_20
# away_avg_shots_on_target_last_5, away_avg_shots_on_target_last_20
def get_avg_shots_on_target_last_n(curr_match, n, home_away):  # "N" 5 or 20 probably
    # Check if wanted for currently HOME or currently AWAY team
    if home_away == "home":

        # Last N matches of a home team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.home_team_id)
        team_id = curr_match.home_team_id

    elif home_away == "away":

        # Last N matches of an away team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.away_team_id)
        team_id = curr_match.away_team_id

    else:
        raise Exception("The \"home_away\" parameter set to a wrong value.")

    total_shots_on_target = 0
    total_none_values = 0
    for match in last_n_matches:
        if match is None:
            total_none_values += 1

        # In each match the wanted team was either HOME or AWAY
        else:
            if team_id == match.home_team_id:
                total_shots_on_target += match.home_team_shots_on_target
            elif team_id == match.away_team_id:
                total_shots_on_target += match.away_team_shots_on_target
            else:
                raise Exception(
                    "The \"team_id\" parameter equals neither to home or away team in one of the previous matches.")

    # Avoid division by zero
    if n - total_none_values == 0:
        return 0

    return total_shots_on_target / (n - total_none_values)


# home_avg_goals_scored_home_last_5, home_avg_goals_scored_home_last_20
# away_avg_goals_scored_away_last_5, away_avg_goals_scored_away_last_20
# home_avg_goals_conceded_home_last_5, home_avg_goals_conceded_home_last_20
# away_avg_goals_conceded_away_last_5, away_avg_goals_conceded_away_last_20
def get_avg_goals_scored_conceded_home_or_away_last_n(curr_match, n, home_away, scored_conceded):
    # Check if wanted for currently HOME or currently AWAY team
    if home_away == "home":

        # Last N home matches of a home team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.home_team_id, "home")
        team_id = curr_match.home_team_id

    elif home_away == "away":

        # Last N away matches of an away team
        last_n_matches = ut.get_n_previous_matches(n, curr_match, curr_match.away_team_id, "away")
        team_id = curr_match.away_team_id

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
                if team_id == match.home_team_id:
                    total_goals += match.home_team_goals
                elif team_id == match.away_team_id:
                    total_goals += match.away_team_goals
                else:
                    raise Exception(
                        "The \"team_id\" param equals neither to home or away team in one of the previous matches.")

            # Counting number of conceded goals of a team
            elif scored_conceded == "conceded":

                # In each match the wanted team was either HOME or AWAY - if HOME then count AWAY team goals etc.
                if team_id == match.home_team_id:
                    total_goals += match.away_team_goals
                elif team_id == match.away_team_id:
                    total_goals += match.home_team_goals
                else:
                    raise Exception(
                        "The \"team_id\" param equals neither to home or away team in one of the previous matches.")

            else:
                raise ValueError("The \"scored_conceded\" parameter equals neither to \"scored\" or \"conceded\".")

    # Avoid division by zero
    if n - total_none_values == 0:
        return 0

    return total_goals / (n - total_none_values)


# home_elo_rating, away_elo_rating
def update_elo_for_both_teams(curr_match):
    # Get previous match of currently HOME team and find out if it was home or away team in that previous match
    # Then get its ELO and goals scored in the previous match
    home_team_prev_match = ut.get_previous_match(curr_match, curr_match.home_team)
    if home_team_prev_match is None:
        home_team_prev_match_elo = INIT_ELO
        home_team_prev_match_goals = 0
    else:
        if home_team_prev_match.home_team == curr_match.home_team:
            home_team_prev_match_elo = home_team_prev_match.home_team_elo
            home_team_prev_match_goals = home_team_prev_match.home_team_goals
        elif home_team_prev_match.away_team == curr_match.home_team:
            home_team_prev_match_elo = home_team_prev_match.away_team_elo
            home_team_prev_match_goals = home_team_prev_match.away_team_goals
        else:
            raise Exception("Current home team not found in its previous match. This should never happen.")

    # Get previous match of currently AWAY team and find out if it was home or away team in that previous match
    # Then get its ELO and goals scored in the previous match
    away_team_prev_match = ut.get_previous_match(curr_match, curr_match.away_team)
    if away_team_prev_match is None:
        away_team_prev_match_elo = INIT_ELO
        away_team_prev_match_goals = 0
    else:
        if away_team_prev_match.home_team == curr_match.away_team:
            away_team_prev_match_elo = away_team_prev_match.home_team_elo
            away_team_prev_match_goals = away_team_prev_match.home_team_goals
        elif away_team_prev_match.away_team == curr_match.away_team:
            away_team_prev_match_elo = away_team_prev_match.away_team_elo
            away_team_prev_match_goals = away_team_prev_match.away_team_goals
        else:
            raise Exception("Current away team not found in its previous match. This should never happen.")

    expected_score_home_team = 1.0 / (1.0 + (ELO_C ** ((away_team_prev_match_elo - home_team_prev_match_elo) / ELO_D)))
    expected_score_away_team = 1.0 - expected_score_home_team

    if home_team_prev_match_goals > away_team_prev_match_goals:
        alpha_home = 1
        alpha_away = 0
    elif home_team_prev_match_goals < away_team_prev_match_goals:
        alpha_home = 0
        alpha_away = 1
    else:
        alpha_home = 0.5
        alpha_away = 0.5

    home_team_new_elo = home_team_prev_match_elo + ELO_K * (alpha_home - expected_score_home_team)
    away_team_new_elo = away_team_prev_match_elo + ELO_K * (alpha_away - expected_score_away_team)

    curr_match.home_team_elo = home_team_new_elo
    curr_match.away_team_elo = away_team_new_elo


# home_position, away_position
def get_team_position_at_round(comp, season, round_, team_id):
    # Get the appropriate table
    table = ut.get_table_by_comp_season(comp, season)

    # Calculate table position up to the specified round
    sorted_teams_positions = table.calculate_and_get_teams_positions_at_round(round_)

    return sorted_teams_positions.index(team_id) + 1
