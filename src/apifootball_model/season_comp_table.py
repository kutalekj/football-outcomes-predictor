"""
season_comp_table.py
"""

import http.client
import json
import settings
import utils as ut


# TODO: Make table available for each type of round with multiple rounds - e.g. DEN Superliga splits in two parts
# TODO: Maybe list of round types for which it is sensible to maintain table for each comp? - in settings
# TODO: Similar system for SUI Super League and even more crazy system holds for BEL Jupiler Pro League
# TODO: Handle also the transition from regular season part to the champions/rel. round - use previous tables and form
# "Regular Season - N", "Championship Round - N", "Relegation Round - N", "Relegation Round", "Championship Round"
# "Conference League Play-offs - Final" (or Semi/...), "Relegation Decider", "Relegation Play-offs - Final" (or ...)
# "Promotion Play-offs - Semi-finals" Pr ...), "Conference League Play-off Group - 3" (or ...)
class SeasonCompTable:
    def __init__(self, comp_id, comp_name, season):
        self.comp_id = comp_id
        self.comp_name = comp_name
        self.season = season
        self.teams = None
        self.team_stats = None

        self.conn = http.client.HTTPSConnection(settings.HOST)

    def init_teams_in_season_comp(self):
        request_string = "/teams?league=" + str(self.comp_id) + "&season=" + str(self.season)

        self.conn.request("GET", request_string, headers=settings.HEADERS)

        res = self.conn.getresponse()
        data = res.read()

        data_teams = json.loads(data)

        teams = []
        for team in data_teams['response']:
            teams.append({'id': int(team['team']['id']), 'name': team['team']['name']})

        self.teams = teams
        self.team_stats = {(team['id'], team['name']): {'points': 0, 'goals_for': 0, 'goals_against': 0} for team in
                           self.teams}

    def update_table(self, matches):
        for match in matches:
            home_team_id = match.home_team_id
            home_team_name = match.home_team_name
            away_team_id = match.away_team_id
            away_team_name = match.away_team_name
            home_goals = match.home_team_goals
            away_goals = match.away_team_goals

            self.team_stats[(home_team_id, home_team_name)]['goals_for'] += home_goals
            self.team_stats[(home_team_id, home_team_name)]['goals_against'] += away_goals
            self.team_stats[(away_team_id, away_team_name)]['goals_for'] += away_goals
            self.team_stats[(away_team_id, away_team_name)]['goals_against'] += home_goals

            if home_goals > away_goals:
                self.team_stats[(home_team_id, home_team_name)]['points'] += 3
            elif home_goals < away_goals:
                self.team_stats[(away_team_id, away_team_name)]['points'] += 3
            else:
                self.team_stats[(home_team_id, home_team_name)]['points'] += 1
                self.team_stats[(away_team_id, away_team_name)]['points'] += 1

    def calculate_and_get_teams_positions_in_season_at_round(self, round_):
        # Reset team stats
        self.team_stats = {(team['id'], team['name']): {'points': 0, 'goals_for': 0, 'goals_against': 0} for team in
                           self.teams}

        # Get all matches up to the wanted round
        regular_matches_up_to_round = ut.get_all_regular_matches_in_season_up_to_round(self.comp_id, self.season,
                                                                                       round_)

        # Update the table with matches up to the wanted round
        self.update_table(regular_matches_up_to_round)

        sorted_teams = sorted(self.teams, key=lambda team: (
            self.team_stats[(team['id'], team['name'])]['points'],
            self.team_stats[(team['id'], team['name'])]['goals_for'] - self.team_stats[(team['id'], team['name'])][
                'goals_against'],
            self.team_stats[(team['id'], team['name'])]['goals_for']), reverse=True)

        return sorted_teams

    def get_curr_team_position_in_season_at_round(self, team_id, round_):
        sorted_teams = self.calculate_and_get_teams_positions_in_season_at_round(round_)

        for position, team in enumerate(sorted_teams, start=1):
            if team['id'] == team_id:
                return position
        return None
