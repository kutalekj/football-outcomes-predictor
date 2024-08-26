"""
utils.py
"""

import numpy as np
import settings
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
            return get_previous_match(prev_match, team, same_comp, same_season, regular)

    if same_season:
        if curr_match.season > prev_match.season:
            return None

    if regular:
        if not prev_match.round.is_regular:
            return get_previous_match(prev_match, team, same_comp, same_season, regular)

    return prev_match


def get_last_home_away_match(curr_match, team_id, home_away=None, same_comp=False, same_season=False, regular=False):
    curr_prev_match = get_previous_match(curr_match, team_id, same_comp, same_season, regular)

    if curr_prev_match is None:
        return None

    # Searching for a last home match of team "team_id"
    if home_away == "home":
        # Found and it is a home match
        if curr_prev_match.home_team_id == team_id:
            return curr_prev_match

        # Found away match - go back until home match is found
        elif curr_prev_match.away_team_id == team_id:
            new_curr_prev_match = curr_prev_match

            for i in range(0, MAX_MATCH_HISTORY_TO_CHECK_LOW):
                prev_prev_match = get_previous_match(new_curr_prev_match, team_id, same_comp, same_season, regular)

                # There might not be any previous matches left
                if prev_prev_match is None:
                    return None

                if prev_prev_match.home_team_id == team_id:
                    return prev_prev_match

                new_curr_prev_match = prev_prev_match

            raise Exception(
                f"Last home match not found even after checking last {MAX_MATCH_HISTORY_TO_CHECK_LOW} matches.")

        else:
            raise ValueError(f"Found team with ID={team_id} which is neither home or away in a match.")

    # Searching for a last away match of team "team_id"
    elif home_away == "away":

        # Found and it is an away match
        if curr_prev_match.away_team_id == team_id:
            return curr_prev_match

        # Found home match - go back until away match is found
        elif curr_prev_match.home_team_id == team_id:
            new_curr_prev_match = curr_prev_match

            for i in range(0, MAX_MATCH_HISTORY_TO_CHECK_LOW):
                prev_prev_match = get_previous_match(new_curr_prev_match, team_id, same_comp, same_season, regular)

                # There might not be any previous matches left
                if prev_prev_match is None:
                    return None

                if prev_prev_match.away_team_id == team_id:
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


def is_match_within_days(curr_datetime, match_datetime, n):
    time_difference = curr_datetime - match_datetime

    return time_difference <= timedelta(days=n)


def min_max_scaling_with_clipping(value, max_value):
    scaled_value = value / max_value
    return np.clip(scaled_value, 0, 1)
