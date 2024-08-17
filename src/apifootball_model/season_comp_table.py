"""
season_comp_table.py
"""

import http.client
import json
import settings
import utils as ut
from team import Team


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
            team_id = int(team['team']['id'])
            team_name = team['team']['name']

            new_team = ut.get_team_if_exists(team_id, team_name)

            # Team not found
            if new_team is None:
                raise Exception(f"Team {team_id}: {team_name} not found in existing ones. Should not happen here, "
                                f"since teams were already initialized during the Comp initialization.")

            teams.append(new_team)

        self.teams = teams
        self.team_stats = {(team.id, team.name): {'points': 0, 'games_played': 0, 'goals_for': 0, 'goals_against': 0,
                                                  'avg_points_per_game': 0} for team in self.teams}

    def update_table(self, matches):
        for match in matches:
            home_team_id = match.home_team_id
            home_team_name = match.home_team_name
            away_team_id = match.away_team_id
            away_team_name = match.away_team_name
            home_goals = match.home_team_goals
            away_goals = match.away_team_goals

            self.team_stats[(home_team_id, home_team_name)]['games_played'] += 1
            self.team_stats[(home_team_id, home_team_name)]['goals_for'] += home_goals
            self.team_stats[(home_team_id, home_team_name)]['goals_against'] += away_goals
            self.team_stats[(away_team_id, away_team_name)]['games_played'] += 1
            self.team_stats[(away_team_id, away_team_name)]['goals_for'] += away_goals
            self.team_stats[(away_team_id, away_team_name)]['goals_against'] += home_goals

            if home_goals > away_goals:
                self.team_stats[(home_team_id, home_team_name)]['points'] += 3.0
            elif home_goals < away_goals:
                self.team_stats[(away_team_id, away_team_name)]['points'] += 3.0
            else:
                self.team_stats[(home_team_id, home_team_name)]['points'] += 1.0
                self.team_stats[(away_team_id, away_team_name)]['points'] += 1.0

        # Normalize points (some teams might have more matches played up to a certain date)
        for team, team_stats in zip(self.teams, self.team_stats):
            team_stats[(team.id, team.name)]['avg_points_per_game'] = team_stats[(team.id, team.name)]['points'] / \
                                                                      team_stats[(team.id, team.name)]['games_played']

    # TODO: How to handle transitions between individual seasons? - in a new season there might be different teams
    def calculate_and_get_teams_positions_in_season_up_to_date(self, date):
        # Reset team stats
        self.team_stats = {(team['id'], team['name']): {'points': 0, 'goals_for': 0, 'goals_against': 0} for team in
                           self.teams}

        # Get all matches up to the wanted date
        regular_matches_up_to_date = ut.get_all_regular_matches_in_season_table_up_to_date(self, )

        # Update the table with matches up to the wanted date
        self.update_table(regular_matches_up_to_date)

        sorted_teams = sorted(self.teams, key=lambda team: (
            self.team_stats[(team['id'], team['name'])]['avg_points_per_game'],
            self.team_stats[(team['id'], team['name'])]['goals_for'] - self.team_stats[(team['id'], team['name'])][
                'goals_against'],
            self.team_stats[(team['id'], team['name'])]['goals_for']), reverse=True)

        return sorted_teams

    def get_curr_team_position_in_season_up_to_date(self, team_id, date):
        sorted_teams = self.calculate_and_get_teams_positions_in_season_up_to_date(date)

        for position, team in enumerate(sorted_teams, start=1):
            if team['id'] == team_id:
                return position
        return None
