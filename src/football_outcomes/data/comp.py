"""
comp.py
"""

import http.client
import json
import re
import time
from datetime import datetime, timezone

# import numpy as np
import requests
from dateutil.parser import parse

from football_outcomes.config import settings
from football_outcomes.config.globals import Global
from football_outcomes.data import rounds
from football_outcomes.data.team import Team
from football_outcomes.utils import common as ut


class Comp:
    def __init__(self, id_, name, regular_keywords):
        self.id = id_
        self.name = name
        self.country = None

        self.rounds_per_season = []
        self.all_rounds_sorted = []  # Note the rounds are probably sorted in the order of when completely played...

        self.regular_round_keywords = regular_keywords

        self.teams_per_season = []
        self.fs_teams_per_season = []
        self.start_end_dates_per_season = []

        self.conn = http.client.HTTPSConnection(settings.HOST)

    def get_round_by_comp_season_round_name(self, season, round_name):
        for season_rounds in self.rounds_per_season:
            if season_rounds["season"] == season:
                for round_ in season_rounds["rounds"]:
                    if round_.name == round_name:
                        return round_

        # Fixed for matching and returning e.g. "1st Round" from "1st Round - 1" (SUI and POR cup issues)
        round_regex = re.compile(r"^(?P<main>.+?\bRound)\s*[â€“â€”-]\s*(?P<num>\d+)$", re.IGNORECASE)
        m = round_regex.search(round_name)

        if m:
            round_name_trimmed = m.group("main")
            for season_rounds in self.rounds_per_season:  # search again on corrected round name
                if season_rounds["season"] == season:
                    for round_ in season_rounds["rounds"]:
                        if round_.name == round_name_trimmed:
                            print(
                                "\t\t\tRETURNED" + m.group("main") + f" FOR COMP [{self.name}], SEASON [{season}] "
                                f"AND ROUND NAME [{round_name}]!"
                            )
                            return round_

        return None

    def init_teams_in_comp(self):
        global_instance = Global.get_instance()
        print(f"[1]: Initializing comp [{self.name}].")

        for season in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1):

            # 1. Init AF season data
            request_string = "/leagues?id=" + str(self.id) + "&season=" + str(season)
            self.conn.request("GET", request_string, headers=settings.HEADERS)
            res = self.conn.getresponse()
            data = res.read()
            data_comp_season = json.loads(data)
            if len(data_comp_season["response"]) == 0:
                continue  # comp season might not have started yet

            self.country = data_comp_season["response"][0]["country"]["name"]

            start_date_str = (
                data_comp_season["response"][0]["seasons"][0]["start"]
                if data_comp_season["response"][0]["seasons"][0]["year"] == season
                else None
            )
            end_date_str = (
                data_comp_season["response"][0]["seasons"][0]["end"]
                if data_comp_season["response"][0]["seasons"][0]["year"] == season
                else None
            )

            if start_date_str is None or end_date_str is None:
                raise ValueError(f"Unable to found corresponding season ([{season}] for comp {self.name})")

            self.start_end_dates_per_season.append(
                {"season": season, "start": parse(start_date_str), "end": parse(end_date_str)}
            )

            # 2. Init FS season data (only regular comps)
            if len(self.regular_round_keywords) > 0:
                fs_season_id = self.get_fs_season_id(self.id, self.country, season)
                comp_season_teams_request_string_fs = (
                    settings.FS_HOST
                    + "/league-teams?key="
                    + settings.FS_KEY
                    + "&season_id="
                    + str(fs_season_id)
                    + "&include=stats"
                )
                res = requests.get(comp_season_teams_request_string_fs)
                data_comp_season_teams_fs = res.json()

                fs_teams_comp_season = [
                    {
                        "id": x["id"],
                        "name": x["name"],
                        "cleanName": x["cleanName"],
                        "english_name": x["english_name"],
                        "country": x["country"],
                        "season": x["season"],
                        "competition_id": x["competition_id"],
                        "full_name": x["full_name"],
                    }
                    for x in data_comp_season_teams_fs["data"]
                ]
                if len(fs_teams_comp_season) == 0:
                    raise ValueError(f"For an unknown reason no FS teams were found for comp {self.name} {str(season)}")

                # Init FS season teams (only regular teams)
                if len(self.regular_round_keywords) > 0:
                    self.fs_teams_per_season.append({"season": season, "fs_teams": fs_teams_comp_season})

            # 3. Init AF season teams
            request_string = "/teams?league=" + str(self.id) + "&season=" + str(season)
            self.conn.request("GET", request_string, headers=settings.HEADERS)
            res = self.conn.getresponse()
            data = res.read()
            data_teams = json.loads(data)

            teams = []
            for team in data_teams["response"]:
                team_id = int(team["team"]["id"])
                team_name = team["team"]["name"]

                # Find team if exists - create if not exists
                new_team = ut.get_team_if_exists(team_id)
                if new_team is None:
                    new_team = Team(team_id, team_name)

                new_team.regularity_in_comp_season.append({"comp": self, "season": season, "is_regular": False})

                teams.append(new_team)  # add team to teams list of a season of the current Comp
                global_instance.all_teams.append(new_team)  # add team to the global teams list

                time.sleep(0.05)

            self.teams_per_season.append({"season": season, "teams": teams})  # AF teams

            # 4. Get all FS league matches for each comp season
            fs_season_id = self.get_fs_season_id(self.id, self.country, season)
            comp_season_matches_request_string_fs = (
                settings.FS_HOST + "/league-matches?key=" + settings.FS_KEY + "&season_id=" + str(fs_season_id)
            )
            res = requests.get(comp_season_matches_request_string_fs)
            data_comp_season_matches_fs = res.json()

            fs_matches_comp_season = [x for x in data_comp_season_matches_fs["data"]]
            if len(fs_matches_comp_season) == 0:
                raise ValueError(f"For an unknown reason no FS matches were found for comp {self.name} {str(season)}")

            if (self.id, season) not in global_instance.fs_leagues_matches:
                global_instance.fs_leagues_matches[(self.id, season)] = []
            global_instance.fs_leagues_matches[(self.id, season)] = [
                {
                    "fs_match_id": int(x["id"]),
                    "fs_home_team_id": int(x["homeID"]),
                    "fs_away_team_id": int(x["awayID"]),
                    "season": (int(x["season"]) if "/" not in x["season"] else int(x["season"].split("/")[0])),
                    "datetime": datetime.fromtimestamp(x["date_unix"], tz=timezone.utc).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    ),
                }
                for x in fs_matches_comp_season
            ]

        global_instance.all_teams = list(set(global_instance.all_teams))  # Remove duplicates

    def init_all_rounds(self):

        for season in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1):

            # Get all round names in comp season
            request_string = "/fixtures/rounds?league=" + str(self.id) + "&season=" + str(season)
            self.conn.request("GET", request_string, headers=settings.HEADERS)
            res = self.conn.getresponse()
            data = res.read()
            rounds_per_season = json.loads(data)

            # Create new Round
            season_rounds_list = []
            for round_name in rounds_per_season["response"]:
                new_round = rounds.Round(self.id, self.name, season, round_name)

                new_round.is_regular = new_round.is_round_regular(self)  # season comp table only for regular matches

                season_rounds_list.append(new_round)
                self.all_rounds_sorted.append(new_round)  # TODO cleanup: remove this variable (from in_out_mega.py too)

            self.rounds_per_season.append({"season": season, "rounds": season_rounds_list})

            time.sleep(0.2)

    def init_country_start_end_dates_in_seasons(self):
        global_instance = Global.get_instance()
        for season in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1):

            # Comp season might not have started yet - unknown start/end dates
            if season not in [s["season"] for s in self.start_end_dates_per_season]:
                continue

            # TODO adj: for current season the final end dates are usually not available yet,
            #  resulting in "January" for instance - can copy end dates from prev season
            start_date = self.get_date_for_comp_season(season, "start")
            end_date = self.get_date_for_comp_season(season, "end")

            if self.country != "World":
                if start_date < global_instance.start_end_dates_per_country_season[self.country][season]["start"]:
                    global_instance.start_end_dates_per_country_season[self.country][season]["start"] = start_date

                if end_date > global_instance.start_end_dates_per_country_season[self.country][season]["end"]:
                    global_instance.start_end_dates_per_country_season[self.country][season]["end"] = end_date

            # "World" competitions (EU cups) are common for all the countries - affect their season start/end dates
            elif self.country == "World":
                for country in global_instance.start_end_dates_per_country_season.keys():
                    for seas in global_instance.start_end_dates_per_country_season[country].keys():
                        if (
                            start_date < global_instance.start_end_dates_per_country_season[country][seas]["start"]
                            and season == seas
                        ):
                            global_instance.start_end_dates_per_country_season[country][seas]["start"] = start_date

                        if (
                            end_date > global_instance.start_end_dates_per_country_season[country][seas]["end"]
                            and season == seas
                        ):
                            global_instance.start_end_dates_per_country_season[country][seas]["end"] = end_date

    def get_date_for_comp_season(self, season, date_type):
        # date_type should be either "start" or "end"
        for season_dates in self.start_end_dates_per_season:
            if season_dates["season"] == season:
                return season_dates[date_type]

        raise ValueError(f"Season {season} start/end date not found for competition {self.name}")

    @staticmethod
    def get_fs_leagues_list():
        global_instance = Global().get_instance()

        leagues_list_request_string = settings.FS_HOST + "/league-list?key=" + settings.FS_KEY
        res = requests.get(leagues_list_request_string)
        global_instance.fs_leagues_list = res.json()

    @staticmethod
    def get_fs_season_id(comp_id, comp_country, season):
        # Hotfix mismatching country names
        if comp_country == "Saudi-Arabia":
            comp_country = "Saudi Arabia"

        global_instance = Global().get_instance()
        league_list = global_instance.fs_leagues_list

        if league_list["pager"]["max_page"] != 1:
            raise ValueError("Multiple pages were obtained from FS leagues list request. Add handling in code...")

        comp_fs_alias = [x["fs_alias"] for x in settings.COMPS_v2 if x["id"] == comp_id]
        if len(comp_fs_alias) != 1:
            raise ValueError("Found none, or multiple FS aliases for a single competition")
        comp_fs_alias = comp_fs_alias[0]

        if comp_country == "World":
            comp_country = "Europe"  # UEFA competitions are under "Europe" in FS data, instead of "World" as in AF data
        wanted_comp = [
            x for x in league_list["data"] if x["country"] == comp_country and x["league_name"] == comp_fs_alias
        ]
        if len(wanted_comp) != 1:
            raise ValueError("Found none, or multiple FS competitions for a single competition")
        wanted_comp = wanted_comp[0]

        season_id = [
            x["id"]
            for x in wanted_comp["season"]
            if str(x["year"]) == (str(season) + str(season + 1)) or str(x["year"]) == (str(season))
        ]
        if len(season_id) == 0 and season == settings.LAST_SEASON:
            return None  # case for beginning of season (Aug/Sep/Oct), when start/end dates are not set properly yet
        elif len(season_id) != 1:
            raise ValueError("Found none, or multiple FS season IDs for a single competition")
        return season_id[0]
