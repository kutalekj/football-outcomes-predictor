"""
comp.py
"""

import http.client
import json
import settings
import rounds


class Comp:
    def __init__(self, id_, name):
        self.id = id_
        self.name = name
        self.rounds_per_season = []
        self.all_rounds_sorted = []

        self.conn = http.client.HTTPSConnection(settings.HOST)

    def get_round_by_comp_season_round_name(self, season, round_name):
        for season_rounds in self.rounds_per_season:
            if season_rounds['season'] == season:
                for round_ in season_rounds['rounds']:
                    if round_.name == round_name:
                        return round_

        return None

    def get_prev_round_by_total_rank(self, total_rank):
        return self.all_rounds_sorted[total_rank - 1] if total_rank > 1 else None

    def get_next_round_by_total_rank(self, total_rank):
        return self.all_rounds_sorted[total_rank + 1] if total_rank < len(self.all_rounds_sorted) else None

    def get_prev_round_in_season(self, season, rank_in_season):
        for season_rounds in self.rounds_per_season:
            if season_rounds['season'] == season:
                return season_rounds['rounds'][rank_in_season - 2] if rank_in_season > 1 else None

        raise ValueError(f"Season {str(season)} not found.")

    def get_next_round_in_season(self, season, rank_in_season):
        for season_rounds in self.rounds_per_season:
            if season_rounds['season'] == season:
                return season_rounds['rounds'][rank_in_season]\
                    if rank_in_season < len(season_rounds['rounds']) else None

        raise ValueError(f"Season {str(season)} not found.")

    def init_all_rounds(self):
        total_regular_rounds_counter = 0
        total_rounds_counter = 0

        for season in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1):
            regular_rounds_per_season_counter = 0
            rounds_per_season_counter = 0

            # Get data from API
            request_string = "/fixtures/rounds?league=" + str(self.id) + "&season=" + str(season)

            self.conn.request("GET", request_string, headers=settings.HEADERS)

            res = self.conn.getresponse()
            data = res.read()

            rounds_per_season = json.loads(data)

            # Create new Round instance
            season_rounds_list = []
            for round_name in rounds_per_season['response']:
                new_round = rounds.Round(self.id, self.name, season, round_name)

                # Regularity
                new_round.is_regular = new_round.is_round_regular()
                if new_round.is_regular:
                    regular_rounds_per_season_counter += 1
                    total_regular_rounds_counter += 1

                rounds_per_season_counter += 1
                total_rounds_counter += 1

                new_round.regular_rank_in_season = regular_rounds_per_season_counter
                new_round.total_rank_in_season = rounds_per_season_counter
                new_round.regular_rank_all_time = total_regular_rounds_counter
                new_round.total_rank_all_time = total_rounds_counter

                season_rounds_list.append(new_round)
                self.all_rounds_sorted.append(new_round)

            self.rounds_per_season.append({'season': season, 'rounds': season_rounds_list})
