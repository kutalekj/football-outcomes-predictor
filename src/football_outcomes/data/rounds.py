"""
rounds.py

"Regular" round = a table is maintained for it, contributes to a team position in a table
(regular season, relegation or championship rounds)
"""

import http.client

from football_outcomes.config import settings

# import json
# import urllib.parse


# from football_outcomes.utils import common as ut


class Round:
    def __init__(self, comp_id, comp_name, season, name):
        self.comp_id = comp_id
        self.comp_name = comp_name
        self.season = season
        self.name = name

        self.is_regular = None

        self.conn = http.client.HTTPSConnection(settings.HOST)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("conn", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.conn = http.client.HTTPSConnection(settings.HOST)

    def is_round_regular(self, curr_comp):
        for keyword in curr_comp.regular_round_keywords:
            if keyword in self.name:
                return True
        return False
