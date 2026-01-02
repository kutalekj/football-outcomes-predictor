from __future__ import annotations

import csv
import http.client
import os
import pickle

# import random
# import statistics
import sys
import time
import zoneinfo
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# import numpy as np
import requests

from football_outcomes.config import settings
from football_outcomes.config.globals import Global
from football_outcomes.data import io as in_out

# from football_outcomes.data import io_mega as in_out_mega
from football_outcomes.data.comp import Comp
from football_outcomes.data.match import Match
from football_outcomes.data.season_comp_table import SeasonCompTable
from football_outcomes.data.state_io import export_summary_csvs, load_global_state, save_global_state
from football_outcomes.features.feature import MatchFeatures

# from football_outcomes.training.train_rnn import train
# from football_outcomes.training.train_ann import train
from football_outcomes.utils import common as utils

# from football_outcomes.training.train_compID_encoder import train
# from football_outcomes.training.train_teamID_encoder import train
# from football_outcomes.training.train_team_strength import train
ut = utils

global_instance = Global.get_instance()

# ----------------------------------- FS ONLY -----------------------------------


@dataclass
class FSDataBundle:
    comp_seasons: Dict[int, FSCompSeason] = field(default_factory=dict)
    teams: Dict[int, FSTeam] = field(default_factory=dict)
    players: Dict[int, FSPlayer] = field(default_factory=dict)
    matches: List[FSMatch] = field(default_factory=list)

    fs_leagues_list: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)  # e.g. snapshot version


class FSCompSeason:
    def __init__(self, id_, season_, country_, name_):
        self.id = id_
        self.season = season_
        self.country = country_
        self.name = name_
        self.format = None
        self.domestic_scale = None
        self.division = None
        self.total_game_week = None
        self.matches = []

        self.conn = http.client.HTTPSConnection(settings.HOST)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("conn", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate transient connection
        self.conn = http.client.HTTPSConnection(settings.HOST)


class FSTeam:
    def __init__(self, id_, name_, clean_n, english_n, full_n, shorthand_n, country_):
        self.id = id_
        self.name = name_
        self.comp_seasons = {}
        self.clean_name = clean_n
        self.english_name = english_n
        self.full_name = full_n
        self.shorthand_name = shorthand_n
        self.country = country_

    def __eq__(self, other):
        if isinstance(other, FSTeam):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)


class FSMatch:
    def __init__(self, id_):
        self.id = id_
        self.home_team = None
        self.away_team = None
        self.season = None
        self.round_id = None
        self.game_week = None
        self.datetime = None
        self.month = None
        self.hour_utc = None
        self.hour_local = None

        self.home_points = None
        self.away_points = None
        self.home_goals = None
        self.away_goals = None
        self.stats = {
            "home_corners": -1,
            "away_corners": -1,
            "home_offsides": -1,
            "away_offsides": -1,
            "home_red_cards": -1,
            "away_red_cards": -1,
            "home_yellow_cards": -1,
            "away_yellow_cards": -1,
            "home_shots_on_target": -1,
            "away_shots_on_target": -1,
            "home_shots_off_target": -1,
            "away_shots_off_target": -1,
            "home_total_shots": -1,
            "away_total_shots": -1,
            "home_fouls": -1,
            "away_fouls": -1,
            "home_possession": -1,
            "away_possession": -1,
            "home_attacks": -1,
            "away_attacks": -1,
            "home_dangerous_attacks": -1,
            "away_dangerous_attacks": -1,
            "home_xg": -1,
            "away_xg": -1,
            "home_prematch_xg": -1,
            "away_prematch_xg": -1,
        }

        self.referee_id = None
        self.home_coach_id = None
        self.away_coach_id = None

        self.odds = {
            "home_win": -1,
            "away_win": -1,
            "draw": -1,
            "over05": -1,
            "over15": -1,
            "over25": -1,
            "over35": -1,
            "over45": -1,
            "under05": -1,
            "under15": -1,
            "under25": -1,
            "under35": -1,
            "under45": -1,
            "btts_yes": -1,
            "btts_no": -1,
        }

        self.home_lineup = []
        self.away_lineup = []

    def __eq__(self, other):
        if isinstance(other, FSMatch):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)


class FSPlayer:
    def __init__(self, id_, name, first, last, short, known_as_):
        self.id = id_
        self.full_name = name
        self.first_name = first
        self.last_name = last
        self.shorthand = short
        self.known_as = known_as_
        self.position = None
        self.birthday = None
        self.nationality = None

    def __eq__(self, other):
        if isinstance(other, FSPlayer):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)


BASE_DIR = Path(__file__).resolve().parent
LOAD_SNAPSHOT_PATH = BASE_DIR / "cache" / "fs_full_25-01-01_v3.pkl"
SAVE_SNAPSHOT_PATH = BASE_DIR / "cache" / "fs_full_25-01-02.pkl"
SNAPSHOT_VERSION = 1  # bump if you make incompatible changes


def save_snapshot(bundle: FSDataBundle, path: Path = SAVE_SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle.meta["snapshot_version"] = SNAPSHOT_VERSION
    print(f"Saving snapshot to: {path.resolve()}")  # <--- add this

    with path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_snapshot(path: Path = LOAD_SNAPSHOT_PATH) -> FSDataBundle:
    with path.open("rb") as f:
        bundle: FSDataBundle = pickle.load(f)

    # simple version check (optional but recommended)
    version = bundle.meta.get("snapshot_version", 0)
    if version != SNAPSHOT_VERSION:
        raise RuntimeError(f"Incompatible snapshot version {version}; expected {SNAPSHOT_VERSION}.")

    return bundle


def try_load_snapshot(path: Path = LOAD_SNAPSHOT_PATH) -> Optional[FSDataBundle]:
    if not path.exists():
        return None
    try:
        return load_snapshot(path)
    except Exception as e:
        print(f"Warning: failed to load snapshot ({e}). Rebuilding from API…")
        return None


# 0. Try to load cache
cache = try_load_snapshot()
if cache is not None:
    all_comp_seasons = cache.comp_seasons
    all_teams = cache.teams
    all_players = cache.players
    all_matches = cache.matches

    # If you also want global_instance.fs_leagues_list back:
    global_instance.fs_leagues_list = cache.fs_leagues_list

    print("Loaded FS data from snapshot.")

    print(f"{str(len(all_comp_seasons))} comp seasons found.")
    print(f"{str(len(all_teams))} teams found.")
    print(f"{str(len(all_players))} players found.")
    print(f"{str(len(all_matches))} matches found.")

    id_to_match: dict[int, FSMatch] = {}
    for match in all_matches:
        id_to_match[match.id] = match
    all_matches = list(id_to_match.values())
    existing_match_ids: set[int] = set(id_to_match.keys())
    print(f"{str(len(all_matches))} matches found after dropping duplicates.")

    # --- constants for missing-value reporting ------------------------------------
    SIMPLE_ATTRS = [
        "round_id",
        "game_week",
        "datetime",
        "month",
        "hour_utc",
        "hour_local",
        "home_points",
        "away_points",
        "home_goals",
        "away_goals",
        "referee_id",
        "home_coach_id",
        "away_coach_id",
    ]

    STAT_KEYS = [
        "home_corners",
        "away_corners",
        "home_offsides",
        "away_offsides",
        "home_red_cards",
        "away_red_cards",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_shots_off_target",
        "away_shots_off_target",
        "home_total_shots",
        "away_total_shots",
        "home_fouls",
        "away_fouls",
        "home_possession",
        "away_possession",
        "home_attacks",
        "away_attacks",
        "home_dangerous_attacks",
        "away_dangerous_attacks",
        "home_xg",
        "away_xg",
        "home_prematch_xg",
        "away_prematch_xg",
    ]

    ODDS_KEYS = [
        "home_win",
        "away_win",
        "draw",
        "over05",
        "over15",
        "over25",
        "over35",
        "over45",
        "under05",
        "under15",
        "under25",
        "under35",
        "under45",
        "btts_yes",
        "btts_no",
    ]

    LEAGUE_COMPS = [
        "Belgium Pro League",
        "England Premier League",
        "England Championship",
        "England EFL League One",
        "England EFL League Two",
        "France Ligue 1",
        "France Ligue 2",
        "Netherlands Eredivisie",
        "Turkey Süper Lig",
        "Germany Bundesliga",
        "Germany 2. Bundesliga",
        "Saudi Arabia Professional League",
        "India Indian Super League",
        "Australia A-League",
        "Austria Bundesliga",
        "Spain La Liga",
        "Spain Segunda División",
        "Italy Serie A",
        "Italy Serie B",
        "Scotland Premiership",
        "Poland Ekstraklasa",
        "Denmark Superliga",
        "Portugal Liga NOS",
        "Switzerland Super League",
    ]

    def _is_missing(val) -> bool:
        """Helper: treat None or negative numeric values as 'missing'."""
        if val is None:
            return True
        if isinstance(val, (int, float)) and val < 0:
            return True
        return False

    sum_matches = 0
    for key in sorted(all_comp_seasons, key=lambda id_: all_comp_seasons[id_].name):
        # matches_sorted = sorted(all_comp_seasons[key].matches, key=lambda x: x.datetime)
        comp = all_comp_seasons[key]  # only include league competitions
        if comp.name not in LEAGUE_COMPS:
            continue
        matches_sorted = sorted(comp.matches, key=lambda x: x.datetime)

        if not matches_sorted or len(matches_sorted) == 0:
            print(f"\tNO MATCHES in {all_comp_seasons[key].name} " f"(" f"{str(all_comp_seasons[key].season)})...")
        sum_matches += len(matches_sorted)

        """
        print(f"{all_comp_seasons[key].name} ({str(all_comp_seasons[key].season)}):"
              f"\t\t{str(len(matches_sorted))} matches "
              f"(from {matches_sorted[0].datetime} to {matches_sorted[-1].datetime})")
        """

        # initialise counters for every attribute we care about
        missing_counts: dict[str, int] = {}

        for attr in SIMPLE_ATTRS:
            missing_counts[attr] = 0
        for attr in STAT_KEYS:
            missing_counts[attr] = 0
        for attr in ODDS_KEYS:
            missing_counts[attr] = 0

        home_lineup_short = 0  # matches where len(home_lineup) < 11
        away_lineup_short = 0  # matches where len(away_lineup) < 11

        for m in matches_sorted:
            # 1) simple FSMatch attributes
            for attr in SIMPLE_ATTRS:
                val = getattr(m, attr, None)
                if _is_missing(val):
                    missing_counts[attr] += 1

            # 2) stats dict (sentinel -1 = missing)
            for attr in STAT_KEYS:
                val = m.stats.get(attr, None)
                if _is_missing(val):
                    missing_counts[attr] += 1

            # 3) odds dict (sentinel -1 = missing)
            for attr in ODDS_KEYS:
                val = m.odds.get(attr, None)
                if _is_missing(val):
                    missing_counts[attr] += 1

            # 4) lineups – only care about “incomplete XI”
            if len(m.home_lineup) < 11:
                home_lineup_short += 1
            if len(m.away_lineup) < 11:
                away_lineup_short += 1

        # Build a summary: report EVERY attribute, even if 0% missing
        pct_parts: list[str] = []

        for attr in SIMPLE_ATTRS:
            pct = 100.0 * missing_counts[attr] / len(matches_sorted)
            pct_parts.append(f"{attr}: {pct:.2f}%")

        for attr in STAT_KEYS:
            pct = 100.0 * missing_counts[attr] / len(matches_sorted)
            pct_parts.append(f"{attr}: {pct:.2f}%")

        for attr in ODDS_KEYS:
            pct = 100.0 * missing_counts[attr] / len(matches_sorted)
            pct_parts.append(f"{attr}: {pct:.2f}%")

        # lineups (always report as “len<11” percentages)
        pct_parts.append(f"home_lineup_len<11: " f"{100.0 * home_lineup_short / len(matches_sorted):.2f}%")
        pct_parts.append(f"away_lineup_len<11: " f"{100.0 * away_lineup_short / len(matches_sorted):.2f}%")

        prefix = f"{all_comp_seasons[key].name} ({all_comp_seasons[key].season}): "
        print(prefix + ", ".join(pct_parts))

    print(f"{str(sum_matches)} matches found.")

    pass

    # else:
    #     print("No valid snapshot. Fetching from API…")

    # 1. League List (comps)
    #     Comp.get_fs_leagues_list()

    """
    belgian_comp_seasons = [x for x in global_instance.fs_leagues_list['data'] if
                            x['country'] == 'Belgium']  # TODO: List of country names
    belgian_league = [x for x in belgian_comp_seasons if
                      x['name'] == 'Belgium Pro League'][0]  # TODO: List of comp names
    belgian_league_seasons = [x for x in belgian_league['season'] if
                              x['year'] >= 20212022]  # e.g. {'id': 6079, 'year': 20212022}

    'Belgium', 'England', 'France', 'Netherlands', 'Turkey', 'Germany', 'Saudi Arabia', 'India',
    'Australia', 'Austria'
    'Spain', 'Italy', 'Scotland', 'Poland', 'Denmark', 'Portugal', 'Switzerland', 'Europe'
    'Belgium Pro League', 'England Premier League', 'England Championship',
    'England EFL League One', 'England EFL League Two',
    'France Ligue 1', 'France Ligue 2', 'Netherlands Eredivisie', 'Turkey Süper Lig',
    'Germany Bundesliga'
    'Germany 2. Bundesliga', 'Saudi Arabia Professional League', 'India Indian Super League',
    'Australia A-League'
    'Austria Bundesliga', 'Spain La Liga', 'Spain Segunda División', 'Italy Serie A',
    'Italy Serie B',
    'Scotland Premiership', 'Poland Ekstraklasa', 'Denmark Superliga', 'Portugal Liga NOS',
    'Switzerland Super League',
    'Spain Copa del Rey', 'Scotland Scottish League Cup', 'Scotland Scottish Cup',
    'Poland Polish Cup',
    'Turkey Turkish Cup', 'Switzerland Swiss Cup', 'Saudi Arabia Kings Cup',
    'Portugal Taça de Portugal',
    'Portugal Portuguese League Cup', 'Netherlands KNVB Cup', 'Austria Austrian Cup',
    'Europe UEFA Champions League',
    'Europe UEFA Europa League', 'Europe UEFA Europa Conference League', 'Germany DFB Pokal',
     'Italy Coppa Italia',
    'France Coupe de France', 'England FA Cup', 'England EFL Trophy', 'Denmark Danish Cup',
    'Belgium Belgian Cup', 'Australia FFA Cup'
    """

    comp_seasons = [
        x
        for x in global_instance.fs_leagues_list["data"]
        if x["country"]
        in [
            "Belgium",
            "England",
            "France",
            "Netherlands",
            "Turkey",
            "Germany",
            "Saudi Arabia",
            "India",
            "Australia",
            "Austria",
            "Spain",
            "Italy",
            "Scotland",
            "Poland",
            "Denmark",
            "Portugal",
            "Switzerland",
            "Europe",
        ]
    ]
    leagues = [
        x
        for x in comp_seasons
        if x["name"]
        in [
            "Belgium Pro League",
            "England Premier League",
            "England Championship",
            "England EFL League One",
            "England EFL League Two",
            "France Ligue 1",
            "France Ligue 2",
            "Netherlands Eredivisie",
            "Turkey Süper Lig",
            "Germany Bundesliga",
            "Germany 2. Bundesliga",
            "Saudi Arabia Professional League",
            "India Indian Super League",
            "Australia A-League",
            "Austria Bundesliga",
            "Spain La Liga",
            "Spain Segunda División",
            "Italy Serie A",
            "Italy Serie B",
            "Scotland Premiership",
            "Poland Ekstraklasa",
            "Denmark Superliga",
            "Portugal Liga NOS",
            "Switzerland Super League",
            "Spain Copa del Rey",
            "Scotland Scottish League Cup",
            "Scotland Scottish Cup",
            "Poland Polish Cup",
            "Turkey Turkish Cup",
            "Switzerland Swiss Cup",
            "Saudi Arabia Kings Cup",
            "Portugal Taça de Portugal",
            "Portugal Portuguese League Cup",
            "Netherlands KNVB Cup",
            "Austria Austrian Cup",
            "Europe UEFA Champions League",
            "Europe UEFA Europa League",
            "Europe UEFA Europa Conference League",
            "Germany DFB Pokal",
            "Italy Coppa Italia",
            "France Coupe de France",
            "England FA Cup",
            "England EFL Trophy",
            "Denmark Danish Cup",
            "Belgium Belgian Cup",
            "Australia FFA Cup",
        ]
    ]
    league_seasons = []
    for league in leagues:
        league_seasons += [x for x in league["season"] if x["year"] >= 20212022]

    # TODO: Create global variables
    # all_comp_seasons: dict[int, FSCompSeason] = {}
    # all_teams: dict[int, FSTeam] = {}
    # all_players: dict[int, FSPlayer] = {}
    # all_matches: list[FSMatch] = []
    for comp_season in league_seasons:
        league = next(x for x in leagues if comp_season in x["season"])
        # league_name = [x['name'] for x in leagues if comp_season in x['season']][0]
        league_name = league["name"]
        country_name = league["country"]

        if comp_season["id"] in all_comp_seasons:
            new_comp_season = all_comp_seasons[comp_season["id"]]  # reuse existing season
        else:
            new_comp_season = FSCompSeason(
                comp_season["id"], int(str(comp_season["year"])[:4]), country_name, league_name
            )
            # new_comp_season = FSCompSeason(comp_season['id'], int(str(comp_season['year'])[:4]),
            # leagues[0]['country'], league_name)
            all_comp_seasons[new_comp_season.id] = new_comp_season

        # 2. League Stats
        request_string = (
            settings.FS_HOST
            + "/league-season?key="
            + settings.FS_KEY
            + "&season_id="
            + str(new_comp_season.id)
            + "&include=stats"
        )
        res = requests.get(request_string)
        res_data = res.json()["data"]
        new_comp_season.format = res_data["format"]  # e.g. "Domestic League"
        new_comp_season.domestic_scale = res_data["domestic_scale"]
        new_comp_season.division = res_data["division"]
        new_comp_season.total_game_week = res_data["total_game_week"]

        # 3. League Teams
        request_string = (
            settings.FS_HOST
            + "/league-teams?key="
            + settings.FS_KEY
            + "&season_id="
            + str(new_comp_season.id)
            + "&include=stats"
        )
        res = requests.get(request_string)
        res_data = res.json()

        for t in res_data["data"]:
            team_id = t["id"]
            if team_id not in all_teams:
                new_team = FSTeam(
                    team_id, t["name"], t["cleanName"], t["english_name"], t["full_name"], t["shortHand"], t["country"]
                )
                new_team.comp_seasons[t["competition_id"]] = []  # list of players in roster

                all_teams[team_id] = new_team
            else:
                team = all_teams[team_id]
                team.comp_seasons[t["competition_id"]] = []  # list of players in roster

        # League Tables
        """
        request_string = (settings.FS_HOST + "/league-tables?key=" + settings.FS_KEY + "&season_id=" +
                          str(new_comp_season.id) + "&include=stats")
        res = requests.get(request_string)
        res_data = res.json()['data']
        """

        # 4. League Players
        request_string = (
            settings.FS_HOST + "/league-players?key=" + settings.FS_KEY + "&season_id=" + str(new_comp_season.id)
        )
        res = requests.get(request_string)
        res_data = res.json()

        all_rows = []
        all_rows.extend(res_data["data"])
        max_page = res_data["pager"]["max_page"]

        for page in range(2, max_page + 1):
            request_string = (
                settings.FS_HOST
                + "/league-players?key="
                + settings.FS_KEY
                + "&season_id="
                + str(new_comp_season.id)
                + "&page="
                + str(page)
            )
            res = requests.get(request_string)
            page_data = res.json()
            all_rows.extend(page_data["data"])

        for player in all_rows:
            if player["id"] not in all_players:
                new_player = FSPlayer(
                    player["id"],
                    player["full_name"],
                    player["first_name"],
                    player["last_name"],
                    player["shorthand"],
                    player["known_as"],
                )
                new_player.position = player["position"]

                ts = player.get("birthday")
                birthday_dt = None
                if isinstance(ts, int) and ts > 0:
                    try:
                        birthday_dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )
                    except (OSError, OverflowError, ValueError):  # still something wrong -> None
                        print(
                            f"Invalid birthday timestamp {ts} for player {player['id']}." f" Setting birthday to None."
                        )
                else:  # None, 0, -1, string, etc.
                    print(f"Missing or invalid birthday {ts} for player {player['id']}. " f"Setting birthday to None.")
                if birthday_dt is None:
                    print(f"Skipping player {player['id']} due to invalid birthday.")
                    continue
                new_player.birthday = birthday_dt

                new_player.nationality = player["nationality"]

                try:
                    all_teams[player["club_team_id"]].comp_seasons[player["competition_id"]].append(new_player)
                except Exception:
                    print(
                        f"Found player [{new_player.full_name}, {new_comp_season.name} "
                        f"{str(new_comp_season.season)}] playing for unknown team "
                        f"[{str(player['club_team_id'])}]. Skipping..."
                    )
                    continue
                if player["club_team_2_id"] != -1:
                    try:
                        all_teams[player["club_team_2_id"]].comp_seasons[player["competition_id"]].append(new_player)
                    except Exception:
                        print(
                            f"Found player [{new_player.full_name}, {new_comp_season.name} "
                            f"{str(new_comp_season.season)}] playing for unknown second team "
                            f"[{str(player['club_team_2_id'])}]. Skipping..."
                        )
                        continue
                all_players[player["id"]] = new_player
        print(
            f"All {str(len(all_rows))} players in [{new_comp_season.name}, "
            f"{str(new_comp_season.season)}] were successfully retrieved."
        )

        # 5. League Matches
        request_string = (
            settings.FS_HOST + "/league-matches?key=" + settings.FS_KEY + "&season_id=" + str(new_comp_season.id)
        )
        res = requests.get(request_string)
        res_data = res.json()

        matches_data: list[dict] = []
        matches_data.extend(res_data["data"])
        pager = res_data.get("pager", {})
        max_page = pager.get("max_page", 1)
        for page in range(2, max_page + 1):
            request_string = (
                settings.FS_HOST
                + "/league-matches?key="
                + settings.FS_KEY
                + "&season_id="
                + str(new_comp_season.id)
                + "&page="
                + str(page)
            )
            res = requests.get(request_string)
            page_data = res.json()
            matches_data.extend(page_data["data"])
        if not matches_data:
            raise ValueError(
                f"For an unknown reason no matches were found "
                f"for FSCompSeason {new_comp_season.name} {str(new_comp_season.season)}"
            )

        for m in matches_data:
            if m["competition_id"] != new_comp_season.id:
                raise ValueError(
                    f"FSCompSeason ID of match ([{str(m['competition_id'])}]) should"
                    f"correspond to the FSCompSeason ID ([{str(new_comp_season.id)}])."
                )
            if m["status"] != "complete":
                print(
                    f"Found [{m['status']}] match in FSCompSeason [{str(new_comp_season.id)}, "
                    f"{new_comp_season.name}] {m['home_name']} vs. {m['away_name']}. Skipping..."
                )
                continue

            if m["id"] in existing_match_ids:
                existing_match = id_to_match[m["id"]]  # reuse already existing match and its links
                if existing_match not in new_comp_season.matches:
                    new_comp_season.matches.append(existing_match)
                continue

            new_match = FSMatch(m["id"])
            try:
                new_match.home_team = all_teams[m["homeID"]]
            except Exception:
                print(f"Found non-existing home team ID {str(m['homeID'])}. Skipping match...")
                continue
            try:
                new_match.away_team = all_teams[m["awayID"]]
            except Exception:
                print(f"Found non-existing away team ID {str(m['awayID'])}. Skipping match...")
                continue
            new_match.season = int(m["season"].split("/")[0])
            if new_match.season != new_comp_season.season:
                raise ValueError(
                    f"Seasons of the FSCompSeason ({str(new_comp_season.season)}) and"
                    f"the match ({str(new_match.season)}) do not match."
                )
            new_match.round_id = m["roundID"]
            new_match.game_week = m["game_week"]

            tz_local = zoneinfo.ZoneInfo("Europe/Brussels")  # TODO: Correct local timezones
            dt_utc = datetime.fromtimestamp(m["date_unix"], tz=timezone.utc)
            new_match.month = dt_utc.month
            new_match.hour_utc = dt_utc.hour
            dt_local = dt_utc.astimezone(tz_local)
            new_match.hour_local = dt_local.hour
            new_match.datetime = dt_utc.replace(hour=0, minute=0, second=0, microsecond=0)

            new_match.home_goals = m["homeGoalCount"]
            new_match.away_goals = m["awayGoalCount"]
            if new_match.home_goals == -1 or new_match.away_goals == -1:
                raise ValueError(
                    f"Information about home goals ({str(new_match.home_goals)}) or "
                    f"away goals ({str(new_match.away_goals)}) is missing."
                )
            if new_match.home_goals > new_match.away_goals:
                new_match.home_points = 3
                new_match.away_points = 0
            elif new_match.away_goals > new_match.home_goals:
                new_match.home_points = 0
                new_match.away_points = 3
            else:
                new_match.home_points = 1
                new_match.away_points = 1

            new_match.referee_id = m["refereeID"]
            new_match.home_coach_id = m["coach_a_ID"]
            new_match.away_coach_id = m["coach_b_ID"]
            new_match.stats["home_corners"] = m["team_a_corners"]
            new_match.stats["away_corners"] = m["team_b_corners"]
            new_match.stats["home_offsides"] = m["team_a_offsides"]
            new_match.stats["away_offsides"] = m["team_b_offsides"]
            new_match.stats["home_red_cards"] = m["team_a_red_cards"]
            new_match.stats["away_red_cards"] = m["team_b_red_cards"]
            new_match.stats["home_yellow_cards"] = m["team_a_yellow_cards"]
            new_match.stats["away_yellow_cards"] = m["team_b_yellow_cards"]
            new_match.stats["home_shots_on_target"] = m["team_a_shotsOnTarget"]
            new_match.stats["away_shots_on_target"] = m["team_b_shotsOnTarget"]
            new_match.stats["home_shots_off_target"] = m["team_a_shotsOffTarget"]
            new_match.stats["away_shots_off_target"] = m["team_b_shotsOffTarget"]
            new_match.stats["home_total_shots"] = m["team_a_shots"]
            new_match.stats["away_total_shots"] = m["team_b_shots"]
            new_match.stats["home_fouls"] = m["team_a_fouls"]
            new_match.stats["away_fouls"] = m["team_b_fouls"]
            new_match.stats["home_possession"] = m["team_a_possession"]
            new_match.stats["away_possession"] = m["team_b_possession"]
            new_match.stats["home_attacks"] = m["team_a_attacks"]
            new_match.stats["away_attacks"] = m["team_b_attacks"]
            new_match.stats["home_dangerous_attacks"] = m["team_a_dangerous_attacks"]
            new_match.stats["away_dangerous_attacks"] = m["team_b_dangerous_attacks"]
            new_match.stats["home_xg"] = m["team_a_xg"]
            new_match.stats["away_xg"] = m["team_b_xg"]
            new_match.stats["home_prematch_xg"] = m["team_a_xg_prematch"]
            new_match.stats["away_prematch_xg"] = m["team_b_xg_prematch"]

            new_match.odds["home_win"] = m["odds_ft_1"]
            new_match.odds["draw"] = m["odds_ft_x"]
            new_match.odds["away_win"] = m["odds_ft_2"]
            new_match.odds["over05"] = m["odds_ft_over05"]
            new_match.odds["over15"] = m["odds_ft_over15"]
            new_match.odds["over25"] = m["odds_ft_over25"]
            new_match.odds["over35"] = m["odds_ft_over35"]
            new_match.odds["over45"] = m["odds_ft_over45"]
            new_match.odds["under05"] = m["odds_ft_under05"]
            new_match.odds["under15"] = m["odds_ft_under15"]
            new_match.odds["under25"] = m["odds_ft_under25"]
            new_match.odds["under35"] = m["odds_ft_under35"]
            new_match.odds["under45"] = m["odds_ft_under45"]
            new_match.odds["btts_yes"] = m["odds_btts_yes"]
            new_match.odds["btts_no"] = m["odds_btts_no"]

            request_string = settings.FS_HOST + "/match?key=" + settings.FS_KEY + "&match_id=" + str(new_match.id)
            res = requests.get(request_string)
            res_data = res.json()["data"]

            if isinstance(res_data, list):  # normalize res_data so that it's dict with 'lineups'
                if not res_data:
                    print(f"No match data for match {str(new_match.id)}. Skipping...")
                    continue
                candidate = next(
                    (item for item in res_data if isinstance(item, dict) and "lineups" in item), None
                )  # find dict actually containing lineups
                if candidate is None:
                    print(f"No lineups entry in match data for match {str(new_match.id)}. " f"Skipping...")
                    continue
                res_data = candidate
            elif not isinstance(res_data, dict):
                print(f"Unexpected match data type ({type(res_data)}) for match {str(new_match.id)}" f". Skipping...")
                continue

            lineups = res_data.get("lineups", {})
            team_a_lineup = lineups.get("team_a", [])
            team_b_lineup = lineups.get("team_b", [])
            if not isinstance(team_a_lineup, list) or not isinstance(team_b_lineup, list):
                print(f"Unexpected lineup structure [{new_comp_season.name}, {new_match.season}]. Skipping match...")
                continue
            if not all(isinstance(p.get("player_id"), int) for p in (team_a_lineup + team_b_lineup)):
                print(f"Found a non-integer player ID [{new_comp_season.name}, {new_match.season}]. Skipping match...")
                continue
            home_lineup_player_ids = [x["player_id"] for x in team_a_lineup]
            for p_id in home_lineup_player_ids:
                try:
                    player = all_players[p_id]
                except Exception:
                    print(
                        f"Found unknown player [{p_id}] [{new_comp_season.name}, "
                        f"{str(new_match.season)}]. Skipping..."
                    )
                    continue
                new_match.home_lineup.append(player)
            away_lineup_player_ids = [x["player_id"] for x in team_b_lineup]
            for p_id in away_lineup_player_ids:
                try:
                    player = all_players[p_id]
                except Exception:
                    print(
                        f"Found unknown player [{p_id}] [{new_comp_season.name}, "
                        f"{str(new_match.season)}]. Skipping..."
                    )
                    continue
                new_match.away_lineup.append(player)
            if len(new_match.home_lineup) != 11 or len(new_match.away_lineup) != 11:
                print(
                    f"\tUnexpected lineup length found (home team: "
                    f"{str(len(new_match.home_lineup))}, away team: "
                    f"{str(len(new_match.away_lineup))})"
                )

            new_comp_season.matches.append(new_match)
            all_matches.append(new_match)
            existing_match_ids.add(new_match.id)
            id_to_match[m["id"]] = new_match
            time.sleep(2.5)
        print(
            f"All {str(len(matches_data))} matches [{new_comp_season.name}, "
            f"{str(new_comp_season.season)}] were successfully retrieved."
        )
        time.sleep(30.0)

    bundle = FSDataBundle(
        comp_seasons=all_comp_seasons,
        teams=all_teams,
        players=all_players,
        matches=all_matches,
        fs_leagues_list=global_instance.fs_leagues_list,
    )
    save_snapshot(bundle)
    print("Snapshot saved.")

sys.exit(0)

# ----------------------------------- FS ONLY -----------------------------------

if not settings.ALL_LOAD:

    # 0. Load average skills and team strengths (SF)
    in_out.load_sf_avg_team_strength()

    # 1. Init comps (seasons, teams, AF rounds, FS matches)
    Comp.get_fs_leagues_list()

    for comp in settings.COMPS_v2:
        new_comp = Comp(comp["id"], comp["name"], comp["regular_round_keywords"])
        new_comp.init_teams_in_comp()
        new_comp.init_all_rounds()

        global_instance.all_comps.append(new_comp)
    global_instance.all_teams = sorted(global_instance.all_teams, key=lambda team_: team_.id)

    # 2. Init country start/end dates, comp season tables and default GK skills
    for comp in global_instance.all_comps:
        for season in [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]:

            # Country start/end dates
            if comp.country not in global_instance.start_end_dates_per_country_season:
                global_instance.start_end_dates_per_country_season[comp.country] = {}

            if season not in global_instance.start_end_dates_per_country_season[comp.country]:
                global_instance.start_end_dates_per_country_season[comp.country][season] = {
                    "start": datetime.datetime.max,
                    "end": datetime.datetime.min,
                }

            if len(comp.regular_round_keywords) == 0:
                continue  # omit the cups - do not create tables for them

            new_table = SeasonCompTable(comp.id, comp.name, season)
            new_table.init_teams_in_season_comp()

            global_instance.all_tables.append(new_table)

    for comp in global_instance.all_comps:
        comp.init_country_start_end_dates_in_seasons()

    # 3a. Load match data from local
    if settings.MATCH_DATA_LOAD:
        in_out.load_matches(settings.M_LOAD_CSV)
    all_loaded_comp_seasons = list(set([(x.comp.id, x.season) for x in global_instance.all_matches]))

    # 3b. Get new match data from API
    Match.get_new_matches_data_using_api(existing=all_loaded_comp_seasons)

    # 4. Correct (set) team regularity and match AF teams with FS teams
    for team in global_instance.all_teams:
        team.matches = sorted(team.matches, key=lambda match_: match_.datetime)  # sort team matches by datetime (asc.)
        team.correct_team_regularity_and_match_af_fs_teams()  # assume each team plays exactly in one reg. comp season!

    # 5a. Exclude irregular teams from table calculations
    SeasonCompTable.exclude_irregular_teams_from_table_calc()

    # 5b. Get FS players from all teams for each comp season (represented by table)
    SeasonCompTable.get_fs_player_rosters_per_regular_comp_season_team()

    # 6. Load individual player stats from sofifa CSV files
    in_out.load_player_stats()

    # 7. Calculate features for each match (must be done chronologically asc.!)
    global_instance.all_matches = sorted(global_instance.all_matches, key=lambda match_: match_.datetime)
    for match in global_instance.all_matches:

        # DEBUG
        if match.home_team.name == "Genk" or match.away_team.name == "Genk":
            stop_here = True

        # Match AF/FS match lineups
        ut.get_fs_match_lineups(match)  # match players in AF match lineup with those in teams' FS comp season roster

        # Get xG match stats (FS)
        if (
            match.round.is_regular
            and match.datetime > settings.GET_XG_IF_MATCH_DATE_NEWER_THAN.replace(tzinfo=match.datetime.tzinfo)
            and (match.total_xg == -1 and match.total_pre_match_xg == -1)
        ):
            ut.get_fs_match_xg(match)

        # Feature calculation
        match.features_before_match_played = match.calculate_match_features()
        match.feature_vector_before_match_played = MatchFeatures.match_features_to_vector(
            match.features_before_match_played
        )
else:
    # in_out_mega.load_all_matches_data()

    load_path = settings.PROCESSED_DIR / "test1.fop"
    load_global_state(load_path)

# 8a. Store matches to local
if settings.MATCH_DATA_STORE:
    in_out.store_matches(settings.M_STORE_CSV)

# 8b. Store all data to local
if settings.ALL_STORE:
    # in_out_mega.store_all_matches_data()

    save_path = settings.PROCESSED_DIR / "test1_full.fop"
    snapshot_path = save_global_state(save_path)
    print(f"Saved snapshot to: {snapshot_path}")

    export_summary_csvs()

# MISSING PLAYERS CHECKING
output_dir = r"C:\Users\kutalekj\Downloads"
os.makedirs(output_dir, exist_ok=True)

print(id(global_instance))

# --- Dump first list (tuples of 2) - unchanged ---
mp2_path = os.path.join(output_dir, "mp2_AF_FS.csv")
with open(mp2_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(getattr(global_instance, "mp2_AF_FS_players_matching_potential_misses_couples", []))

# --- Dump second (dict[comp_id][season] -> list of tuples) ---
mp5_path = os.path.join(output_dir, "mp5_DOB.csv")
with open(mp5_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    mp5_dict = getattr(global_instance, "mp5_DOB_misses_couples", {})
    writer.writerow(
        [
            "comp_id",
            "season",
            "match_datetime",
            "home_team",
            "away_team",
            "fs_known_as",
            "fs_birthday",
        ]
    )
    for comp_id in sorted(mp5_dict.keys()):
        seasons_map = mp5_dict[comp_id]
        for season in sorted(seasons_map.keys()):
            for tup in seasons_map[season]:
                writer.writerow([comp_id, season, *tup])

# --- Dump third (dict[comp_id][season] -> list of tuples) ---
mp6_path = os.path.join(output_dir, "mp6_FS_SF.csv")
with open(mp6_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    mp6_dict = getattr(global_instance, "mp6_FS_SF_players_matching_potential_misses_couples", {})
    # CSV header is optional; if you want one, uncomment:
    # writer.writerow(["comp_id", "season", "fs_name", "sf_name", "score"])
    for comp_id in sorted(mp6_dict.keys()):
        seasons_map = mp6_dict[comp_id]
        for season in sorted(seasons_map.keys()):
            for tup in seasons_map[season]:
                writer.writerow([comp_id, season, *tup])

print(f"CSV files saved to:\n{mp2_path}\n{mp6_path}\n")

# --- Write numerical variables (dict[comp_id][season] -> int) to a text file ---
mp_out_path = os.path.join(output_dir, "mp_out.txt")

numeric_vars = [
    "mp0_all_players_involved_in_AF_FS_checking",
    "mpX_OK_players_AF_FS_matching",
    "mp1a_AF_lineups_missing",
    "mp1b_FS_lineups_missing",
    "mp2_AF_FS_players_matching_potential_misses",  # still a scalar
    "mp3_all_players_involved_in_team_strength_calculation",
    "mp4_team_strength_complete_lineup_imitation",
    "mp5_team_strength_DOB_missing",
    "mp6_team_strength_FS_SF_matching",
    "mp7_team_strength_imitated_skills_as_no_CSV_data",
    "mp7_SKILLS_team_strength_imitated_skills_as_no_data",
    "mp8a_team_strength_imitated_players_as_no_CSV_data",
    "mp8b_team_strength_imitated_players_as_no_CSV_data",
    "mp9_team_strength_balancing_field_to_gk",
    "mp9_team_strength_balancing_gk_to_def",
    "mp9_team_strength_balancing_gk_to_mid",
    "mp9_team_strength_balancing_gk_to_att",
]

with open(mp_out_path, mode="w", encoding="utf-8") as f:
    f.write(f"CSV files saved to:\n{mp2_path}\n{mp6_path}\n\n")
    for var_name in numeric_vars:
        value = getattr(global_instance, var_name, None)
        if isinstance(value, dict):
            f.write(f"{var_name}:\n")
            for cid in sorted(value.keys()):
                inner = value[cid]
                if isinstance(inner, dict):  # per-season map
                    for season in sorted(inner.keys()):
                        f.write(f"  {cid} / {season}: {inner[season]}\n")
                else:
                    # fallback if some older var remains {cid: int}
                    f.write(f"  {cid}: {inner}\n")
        else:
            f.write(f"{var_name}: {value}\n")

print(f"Output written to:\n{mp_out_path}")

"""
# 9a. Distribute regular matches into rounds for training
regular_matches = [x for x in global_instance.all_matches if x.round.is_regular]
regular_matches = sorted(regular_matches, key=lambda match_: match_.datetime)

# 9b. Create mapping of categorical feature values to indices
team_id_map, comp_id_map = ut.get_categorical_features_maps(regular_matches)

# 10. Train
regular_matches_in_rounds = ut.distribute_matches_into_rounds(regular_matches)
train(regular_matches_in_rounds, team_id_map, comp_id_map)
"""

"""
regular_matches_in_rounds = ut.distribute_matches_into_rounds(regular_matches)
# regular_matches_in_rounds = ut.distribute_matches_into_rounds_uniformly(regular_matches)
for i, r in enumerate(regular_matches_in_rounds):
    print(f"{str(len(r))} matches found in round {str(i)}")
# TODO: Maybe ensure that there are at least N matches in each round?

# TODO: Run again without uniformly distributed rounds and check if the total number of matches per comp are as expected
# TODO: ...because now there are always 1250 matches in training data, which is less then without uniform, right?
# TODO: ...so since there are less training data, there are more validation data - more total number of val matches?

# 10. Train
train(regular_matches_in_rounds)

"""
print("breakpoint")
