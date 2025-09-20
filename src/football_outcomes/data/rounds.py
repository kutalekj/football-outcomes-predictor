"""
rounds.py

"Regular" round = a table is maintained for it, contributes to a team position in a table
(regular season, relegation or championship rounds)
"""
from football_outcomes.config import settings
import http.client
import urllib.parse
import json
from football_outcomes.utils import common as ut


class Round:
    def __init__(self, comp_id, comp_name, season, name):
        self.comp_id = comp_id
        self.comp_name = comp_name
        self.season = season
        self.name = name

        self.is_regular = None

        self.conn = http.client.HTTPSConnection(settings.HOST)

    def is_round_regular(self, curr_comp):
        for keyword in curr_comp.regular_round_keywords:
            if keyword in self.name:
                return True
        return False

