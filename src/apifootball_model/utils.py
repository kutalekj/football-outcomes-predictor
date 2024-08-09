"""
utils.py
"""

import settings
from globals import Global


def get_last_round_of_previous_season(comp, curr_season):
    prev_season = curr_season - 1

    if prev_season < settings.FIRST_SEASON:
        return None

    # Last round of previous season equals to the number of rounds up to the last one of the previous season
    for season_rounds in comp.rounds_per_season:
        if comp.rounds_per_season['season'] == prev_season:
            return len(season_rounds['rounds'])


def get_match_by_comp_season_round_team(comp_id, season, total_round_rank_in_season, team_id):
    global_instance = Global.get_instance()

    # Get match
    for match in global_instance.all_matches:
        if match.comp.id == comp_id and match.season == season and \
                match.round.total_rank_in_season == total_round_rank_in_season and \
                (match.home_team_id == team_id or match.away_team_id == team_id):
            return match

    return None


def get_previous_match(curr_match, team_id):
    # Might happen that a match has no previous matches
    if curr_match is None:
        return None

    curr_season = curr_match.season
    curr_round_rank_in_season = curr_match.round.regular_rank_in_season

    # Same season
    if curr_round_rank_in_season > 1:
        prev_round_rank_in_season = curr_round_rank_in_season - 1
        prev_season = curr_season

    # Previous season, last round
    else:
        prev_round_rank_in_season = get_last_round_of_previous_season(curr_match.comp, curr_season)
        prev_season = curr_season - 1

        # Check if not older than the initial season
        if prev_season < settings.FIRST_SEASON:
            return None

    prev_match = get_match_by_comp_season_round_team(curr_match.comp.id, prev_season, prev_round_rank_in_season,
                                                     team_id)
    if prev_match is None:
        return None
    else:
        return prev_match


def get_last_home_away_match(curr_match, team_id, home_away=None):
    MAX_MATCH_HISTORY_TO_CHECK = 10

    curr_prev_match = get_previous_match(curr_match, team_id)

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

            for i in range(0, MAX_MATCH_HISTORY_TO_CHECK):
                prev_prev_match = get_previous_match(new_curr_prev_match, team_id)

                # There might not be any previous matches left
                if prev_prev_match is None:
                    return None

                if prev_prev_match.home_team_id == team_id:
                    return prev_prev_match

                new_curr_prev_match = prev_prev_match  # TODO: Check the copying/deep copying functionality via debug

            raise Exception(
                f"Last home match not found even after checking last {MAX_MATCH_HISTORY_TO_CHECK} matches.")

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

            for i in range(0, MAX_MATCH_HISTORY_TO_CHECK):
                prev_prev_match = get_previous_match(new_curr_prev_match, team_id)

                # There might not be any previous matches left
                if prev_prev_match is None:
                    return None

                if prev_prev_match.away_team_id == team_id:
                    return prev_prev_match

                new_curr_prev_match = prev_prev_match  # TODO: Check the copying/deep copying functionality via debug

            raise Exception(
                f"Last away match not found even after checking last {MAX_MATCH_HISTORY_TO_CHECK} matches.")

        else:
            raise ValueError(f"Found team with ID={team_id} which is neither home or away in a match.")

    else:
        raise ValueError("The \"home_away\" parameter set to a wrong value.")


def get_n_previous_matches(n, curr_match, team_id, home_away=None):
    # TODO: Remember that the previous match is in the first position in the list while the n-th previous is last...
    n_previous_matches = []

    """
    curr_season = curr_match.season
    curr_round = curr_match.round

    for i in range(1, n + 1):

        # Same season
        if curr_round - i > 1:
            prev_round = curr_round - i
            prev_season = curr_season

        # Previous season
        else:
            prev_round = curr_match.rounds_per_season - i + curr_round
            prev_season = curr_season - 1

            # Check if not older than the initial season
            if prev_season < FIRST_SEASON:
                return None

        # NOTE THAT "None" CAN BE RETURNED FROM "get_match_by_comp_season_round_team"
    """

    # Get N last home matches
    if home_away == "home":
        curr_prev_match = curr_match

        for i in range(0, n):
            prev_home_match = get_last_home_away_match(curr_prev_match, team_id, "home")
            n_previous_matches.append(prev_home_match)

            curr_prev_match = prev_home_match  # TODO: Check the copying/deep copying functionality via debug

    # Get N last away matches
    elif home_away == "away":
        curr_prev_match = curr_match

        for i in range(0, n):
            prev_away_match = get_last_home_away_match(curr_prev_match, team_id, "away")
            n_previous_matches.append(prev_away_match)

            curr_prev_match = prev_away_match  # TODO: Check the copying/deep copying functionality via debug

    # Get N last matches
    else:
        curr_prev_match = curr_match

        for i in range(0, n):
            prev_match = get_previous_match(curr_prev_match, team_id)
            n_previous_matches.append(prev_match)

            curr_prev_match = prev_match  # TODO: Check the copying/deep copying functionality via debug

    return n_previous_matches


def get_all_matches_of_round(comp_id, season, round_):
    global_instance = Global.get_instance()

    matches_of_round = []
    # Get match
    for match in global_instance.all_matches:
        if match.comp.id == comp_id and match.season == season and match.round == round_:
            matches_of_round.append(match)

    return matches_of_round


def get_all_regular_matches_in_season_up_to_round(comp_id, season, round_):
    global_instance = Global.get_instance()

    matches_up_to_round = []
    # Get match
    for match in global_instance.all_matches:
        if match.comp.id == comp_id and match.season == season and match.round.is_regular and \
                match.round.regular_rank_in_season <= round_.regular_rank_in_season:
            matches_up_to_round.append(match)

    return matches_up_to_round


def get_table_by_comp_season(comp_id, season):
    global_instance = Global.get_instance()

    for table in global_instance.all_tables:
        if table.comp_id == comp_id and table.season == season:
            return table

    return None
