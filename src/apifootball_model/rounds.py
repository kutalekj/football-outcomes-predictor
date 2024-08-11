"""
rounds.py

"Regular" round = a table is maintained for it, contributes to a team position in a table
(regular season, relegation or championship rounds or Conference league play-off group rounds)
"""


import settings
import http.client
import urllib.parse
import json


class Round:
    def __init__(self, comp_id, comp_name, season, name):
        self.comp_id = comp_id
        self.comp_name = comp_name
        self.season = season
        self.name = name

        self.teams_involved = []
        self.has_all_comp_season_teams = None

        self.is_regular = None

        self.regular_rank_in_season = None  # only for regular rounds
        self.total_rank_in_season = None

        self.regular_rank_all_time = None  # only for regular rounds
        self.total_rank_all_time = None

        self.conn = http.client.HTTPSConnection(settings.HOST)

    def is_round_regular(self, curr_comp):
        for keyword in curr_comp.regular_round_keywords:
            if keyword in self.name:
                return True

    def get_teams_involved(self, comp, season):
        # Get data from API
        round_name_encoded = urllib.parse.quote(self.name)  # Without this encoding, white spaces caused problems
        request_string = "/fixtures?league=" + str(comp.id) + "&season=" + str(season) + "&round=" + round_name_encoded
        self.conn.request("GET", request_string, headers=settings.HEADERS)

        res = self.conn.getresponse()
        data = res.read()

        fixtures_in_round = json.loads(data)

        # Get all the teams
        for fixture in fixtures_in_round['response']:
            home_team = {'id': int(fixture['teams']['home']['id']), 'name': fixture['teams']['home']['name']}
            away_team = {'id': int(fixture['teams']['away']['id']), 'name': fixture['teams']['away']['name']}

            self.teams_involved.append(home_team)
            self.teams_involved.append(away_team)

        # Remove duplicates (might not be needed, just assurance)
        self.teams_involved = [dict(t) for t in {tuple(sorted(team.items())) for team in self.teams_involved}]

    def has_all_comp_season_teams_involved(self, teams_in_comp, season):
        teams_in_curr_season_comp = []

        for teams_in_season_comp in teams_in_comp:
            if teams_in_season_comp['season'] == season:
                teams_in_curr_season_comp = teams_in_season_comp['teams']

        return all(team in self.teams_involved for team in teams_in_curr_season_comp)
