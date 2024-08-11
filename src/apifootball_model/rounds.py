"""
rounds.py

"Regular" round = a table is maintained for it, contributes to a team position in a table
(regular season, relegation or championship rounds or Conference league play-off group rounds)
"""

import settings
import http.client
import urllib.parse
import json
import utils as ut


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

    def get_teams_involved(self):
        all_matches_of_round = ut.get_all_matches_of_round(self.comp_id, self.season, self.total_rank_all_time)

        self.teams_involved = [{'id': match.home_team_id, 'name': match.home_team_name} for match in
                               all_matches_of_round]
        self.teams_involved += [{'id': match.away_team_id, 'name': match.away_team_name} for match in
                                all_matches_of_round]

    def has_all_comp_season_teams_involved(self, teams_in_comp, season):
        teams_in_curr_season_comp = []

        for teams_in_season_comp in teams_in_comp:
            if teams_in_season_comp['season'] == season:
                teams_in_curr_season_comp = teams_in_season_comp['teams']

        return all(team in self.teams_involved for team in teams_in_curr_season_comp)
