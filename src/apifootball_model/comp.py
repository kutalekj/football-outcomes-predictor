"""
comp.py
"""

import http.client
import json
import settings
import rounds
from team import Team
from dateutil.parser import parse
from globals import Global
import utils as ut


class Comp:
    def __init__(self, id_, name, regular_keywords):
        self.id = id_
        self.name = name
        self.country = None

        self.rounds_per_season = []
        self.all_rounds_sorted = []  # Note the rounds are probably sorted in the order of when completely played...

        self.regular_round_keywords = regular_keywords

        self.teams_per_season = []
        self.start_end_dates_per_season = []

        self.conn = http.client.HTTPSConnection(settings.HOST)

    def get_round_by_comp_season_round_name(self, season, round_name):
        for season_rounds in self.rounds_per_season:
            if season_rounds['season'] == season:
                for round_ in season_rounds['rounds']:
                    if round_.name == round_name:
                        return round_

        return None

    def init_teams_in_comp(self):
        global_instance = Global.get_instance()

        # Init teams + get start/end date of each comp season
        for season in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1):

            request_string = "/leagues?id=" + str(self.id) + "&season=" + str(season)

            self.conn.request("GET", request_string, headers=settings.HEADERS)

            res = self.conn.getresponse()
            data = res.read()

            data_comp_season = json.loads(data)

            # Comp season might not have started yet
            if len(data_comp_season['response']) == 0:
                continue

            self.country = data_comp_season['response'][0]['country']['name']

            # Start/End date
            start_date_str = data_comp_season['response'][0]['seasons'][0]['start'] if \
                data_comp_season['response'][0]['seasons'][0]['year'] == season else None
            end_date_str = data_comp_season['response'][0]['seasons'][0]['end'] if \
                data_comp_season['response'][0]['seasons'][0]['year'] == season else None

            if start_date_str is None or end_date_str is None:
                raise ValueError(f"Unable to found corresponding season ([{season}] for comp {self.name})")

            self.start_end_dates_per_season.append({'season': season, 'start': parse(start_date_str),
                                                    'end': parse(end_date_str)})

            # Teams
            request_string = "/teams?league=" + str(self.id) + "&season=" + str(season)

            self.conn.request("GET", request_string, headers=settings.HEADERS)

            res = self.conn.getresponse()
            data = res.read()

            data_teams = json.loads(data)

            teams = []
            for team in data_teams['response']:
                team_id = int(team['team']['id'])
                team_name = team['team']['name']

                # Find team if exists
                new_team = ut.get_team_if_exists(team_id)

                # Team not existing yet
                if new_team is None:
                    new_team = Team(team_id, team_name)

                new_team.regularity_in_comp_season.append({'comp': self, 'season': season, 'is_regular': False})

                teams.append(new_team)  # Add team to teams list of a season of the current Comp

                global_instance.all_teams.append(new_team)  # Add team to the global teams list

            self.teams_per_season.append({'season': season, 'teams': teams})

        global_instance.all_teams = list(set(global_instance.all_teams))  # Remove duplicates

    def init_all_rounds(self):

        for season in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1):
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

                # Regularity (season comp table is only updated by regular round matches)
                new_round.is_regular = new_round.is_round_regular(self)

                season_rounds_list.append(new_round)
                self.all_rounds_sorted.append(new_round)  # TODO: Will these round listing variables be still needed?

            self.rounds_per_season.append({'season': season, 'rounds': season_rounds_list})

    def init_country_start_end_dates_in_seasons(self):
        global_instance = Global.get_instance()
        for season in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1):

            # For example, Coupe de France 2024 might not have started yet - unknown start/end dates
            if season not in [s['season'] for s in self.start_end_dates_per_season]:
                continue

            start_date = self.get_date_for_comp_season(season, "start")
            end_date = self.get_date_for_comp_season(season, "end")

            if self.country != "World":
                if start_date < global_instance.start_end_dates_per_country_season[self.country][season]['start']:
                    global_instance.start_end_dates_per_country_season[self.country][season]['start'] = start_date

                if end_date > global_instance.start_end_dates_per_country_season[self.country][season]['end']:
                    global_instance.start_end_dates_per_country_season[self.country][season]['end'] = end_date

            # "World" competitions (EU cups) are common for all the countries - affect their season start/end dates
            elif self.country == "World":
                for country in global_instance.start_end_dates_per_country_season.keys():
                    for seas in global_instance.start_end_dates_per_country_season[country].keys():
                        if start_date < global_instance.start_end_dates_per_country_season[country][seas]['start']\
                                and season == seas:
                            global_instance.start_end_dates_per_country_season[country][seas]['start'] = start_date

                        if end_date > global_instance.start_end_dates_per_country_season[country][seas]['end']\
                                and season == seas:
                            global_instance.start_end_dates_per_country_season[country][seas]['end'] = end_date

    def get_date_for_comp_season(self, season, date_type):
        # date_type should be either "start" or "end"
        for season_dates in self.start_end_dates_per_season:
            if season_dates['season'] == season:
                return season_dates[date_type]

        raise ValueError(f"Season {season} start/end date not found for competition {self.name}")
