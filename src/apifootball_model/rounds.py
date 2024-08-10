"""
rounds.py
"""


class Round:
    def __init__(self, comp_id, comp_name, season, name):
        self.comp_id = comp_id
        self.comp_name = comp_name
        self.season = season
        self.name = name

        self.is_regular = None

        self.regular_rank_in_season = None  # only for regular rounds
        self.total_rank_in_season = None

        self.regular_rank_all_time = None  # only for regular rounds
        self.total_rank_all_time = None

    def is_round_regular(self, curr_comp):
        for keyword in curr_comp.regular_round_keywords:
            if keyword in self.name:
                return True
