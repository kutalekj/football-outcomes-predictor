"""
season_comp_table.py
"""


import http.client
import json
import settings
import utils as ut


class SeasonCompTable:
    def __init__(self, comp_id, comp_name, season):
        self.comp_id = comp_id
        self.comp_name = comp_name
        self.season = season
        self.teams = None
        self.team_stats = None

        self.conn = http.client.HTTPSConnection(settings.HOST)

    def init_teams_in_season_comp(self):
        request_string = "/teams/league=" + str(self.comp_id) + "&season=" + str(self.season)

        self.conn.request("GET", request_string, headers=settings.HEADERS)

        res = self.conn.getresponse()
        data = res.read()

        data_teams = json.loads(data)

        teams = []
        for team in data_teams['response']:
            teams.append({'id': int(team['team']['id']), 'name': team['team']['name']})

        self.teams = teams
        self.team_stats = {team: {'points': 0, 'goals_for': 0, 'goals_against': 0} for team in self.teams}

    def update_table(self, matches):
        for match in matches:
            home_team = match.home_team_id
            away_team = match.away_team_id
            home_goals = match.home_team_goals
            away_goals = match.away_team_goals

            self.team_stats[home_team]['goals_for'] += home_goals
            self.team_stats[home_team]['goals_against'] += away_goals
            self.team_stats[away_team]['goals_for'] += away_goals
            self.team_stats[away_team]['goals_against'] += home_goals

            if home_goals > away_goals:
                self.team_stats[home_team]['points'] += 3
            elif home_goals < away_goals:
                self.team_stats[away_team]['points'] += 3
            else:
                self.team_stats[home_team]['points'] += 1
                self.team_stats[away_team]['points'] += 1

    def calculate_and_get_teams_positions_at_round(self, round_):
        # Reset team stats
        self.team_stats = {team: {'points': 0, 'goals_for': 0, 'goals_against': 0} for team in self.teams}

        # Get all matches up to the wanted round
        regular_matches_up_to_round = ut.get_all_regular_matches_up_to_round(self.comp_id, self.season, round_)

        # Update the table with matches up to the wanted round
        self.update_table(regular_matches_up_to_round)

        sorted_teams = sorted(self.teams, key=lambda team: (
            self.team_stats[team]['points'],
            self.team_stats[team]['goals_for'] - self.team_stats[team]['goals_against'],
            self.team_stats[team]['goals_for']), reverse=True)

        return sorted_teams

    def get_team_position_at_round(self, team_id, round_):
        sorted_teams = self.calculate_and_get_teams_positions_at_round(round_)

        for position, team in enumerate(sorted_teams, start=1):
            if team['id'] == team_id:
                return position
        return None
