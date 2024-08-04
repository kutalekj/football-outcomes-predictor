"""
rounds.py
"""


class Round:
    def __init__(self, comp_id, comp_name, season, name):
        self.comp_id = comp_id
        self.comp = comp_name
        self.season = season
        self.name = name

        self.is_regular = None

        self.regular_rank_in_season = None  # only for regular rounds
        self.total_rank_in_season = None

        self.regular_rank_all_time = None  # only for regular rounds
        self.total_rank_all_time = None

    # TODO: This is currently a dummy implementation!
    def is_round_regular(self):
        return True if 'Regular' in self.name else False
