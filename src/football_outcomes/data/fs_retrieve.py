import time
import zoneinfo
from datetime import datetime, timezone

import requests

import football_outcomes.config.fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_models import (
    FSCompSeason,
    FSDataBundle,
    FSMatch,
    FSPlayer,
    FSTeam,
)


def fill_globals_with_cache(cache: FSDataBundle, update_leagues_list: bool = False) -> None:
    global_instance = Global.get_instance()
    global_instance.all_comp_seasons = cache.comp_seasons
    global_instance.all_teams = cache.teams
    global_instance.all_players = cache.players
    global_instance.all_matches = cache.matches

    print(f"{str(len(global_instance.all_comp_seasons))} comp seasons loaded from snapshot.")
    print(f"{str(len(global_instance.all_teams))} teams loaded from snapshot.")
    print(f"{str(len(global_instance.all_players))} players loaded from snapshot.")
    print(f"{str(len(global_instance.all_matches))} matches loaded from snapshot.")

    if update_leagues_list:
        request_string = sett.FS_HOST + "/league-list?key=" + sett.FS_KEY
        res = requests.get(request_string)
        res_data = res.json()
        global_instance.leagues_list = res_data["data"]
    else:
        global_instance.leagues_list = cache.leagues_list

    global_instance.sofifa_snapshots = getattr(cache, "sofifa_snapshots", None) or []
    global_instance.sofifa_player_occurrences = getattr(cache, "sofifa_player_occurrences", None) or {}
    global_instance.sofifa_players_by_dob = getattr(cache, "sofifa_players_by_dob", None) or {}
    global_instance.fs_to_sofifa_cache = getattr(cache, "fs_to_sofifa_cache", None) or {}
    global_instance.sofifa_teams_by_league = getattr(cache, "sofifa_teams_by_league", None) or {}
    global_instance.sofifa_team_meta = getattr(cache, "sofifa_team_meta", None) or {}
    global_instance.sofifa_players_by_team = getattr(cache, "sofifa_players_by_team", None) or {}
    global_instance.fs_team_to_sofifa_team = getattr(cache, "fs_team_to_sofifa_team", None) or {}

    print(f"{len(global_instance.sofifa_snapshots)} sofifa snapshots loaded from snapshot.")
    print(f"{len(global_instance.fs_to_sofifa_cache)} fs->sofifa cached matches loaded from snapshot.")


def retrieve_new_data() -> FSDataBundle:
    global_instance = Global.get_instance()

    id_to_match: dict[int, FSMatch] = {}
    for match in global_instance.all_matches:
        id_to_match[match.id] = match
    global_instance.all_matches = list(id_to_match.values())
    existing_match_ids: set[int] = set(id_to_match.keys())
    print(f"{str(len(global_instance.all_matches))} matches found after dropping duplicates.")

    comp_seasons = [x for x in global_instance.leagues_list if x["country"] in sett.COUNTRIES]
    leagues = [x for x in comp_seasons if x["name"] in sett.COMPS_LEAGUE + sett.COMPS_CUP + sett.COMPS_EUROPE]
    league_seasons = []
    for league in leagues:
        league_seasons += [x for x in league["season"] if x["year"] >= 20212022]

    for comp_season in league_seasons:
        league = next(x for x in leagues if comp_season in x["season"])
        league_name = league["name"]
        country_name = league["country"]

        if comp_season["id"] in global_instance.all_comp_seasons:
            new_comp_season = global_instance.all_comp_seasons[comp_season["id"]]  # reuse existing season
        else:
            new_comp_season = FSCompSeason(
                comp_season["id"], int(str(comp_season["year"])[:4]), country_name, league_name
            )
            global_instance.all_comp_seasons[new_comp_season.id] = new_comp_season

        # 1. League Stats
        request_string = (
            sett.FS_HOST
            + "/league-season?key="
            + sett.FS_KEY
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

        # 2. League Teams
        request_string = (
            sett.FS_HOST
            + "/league-teams?key="
            + sett.FS_KEY
            + "&season_id="
            + str(new_comp_season.id)
            + "&include=stats"
        )
        res = requests.get(request_string)
        res_data = res.json()

        for t in res_data["data"]:
            team_id = t["id"]
            if team_id not in global_instance.all_teams:
                new_team = FSTeam(
                    team_id, t["name"], t["cleanName"], t["english_name"], t["full_name"], t["shortHand"], t["country"]
                )
                new_team.comp_seasons[t["competition_id"]] = []  # list of players in roster

                global_instance.all_teams[team_id] = new_team
            else:
                team = global_instance.all_teams[team_id]
                team.comp_seasons[t["competition_id"]] = []  # list of players in roster

        # 3. League Players
        request_string = sett.FS_HOST + "/league-players?key=" + sett.FS_KEY + "&season_id=" + str(new_comp_season.id)
        res = requests.get(request_string)
        res_data = res.json()

        all_rows = []
        all_rows.extend(res_data["data"])
        max_page = res_data["pager"]["max_page"]

        for page in range(2, max_page + 1):
            request_string = (
                sett.FS_HOST
                + "/league-players?key="
                + sett.FS_KEY
                + "&season_id="
                + str(new_comp_season.id)
                + "&page="
                + str(page)
            )
            res = requests.get(request_string)
            page_data = res.json()
            all_rows.extend(page_data["data"])

        for player in all_rows:
            if player["id"] not in global_instance.all_players:
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
                    global_instance.all_teams[player["club_team_id"]].comp_seasons[player["competition_id"]].append(
                        new_player
                    )
                except Exception:
                    print(
                        f"Found player [{new_player.full_name}, {new_comp_season.name} "
                        f"{str(new_comp_season.season)}] playing for unknown team "
                        f"[{str(player['club_team_id'])}]. Skipping..."
                    )
                    continue
                if player["club_team_2_id"] != -1:
                    try:
                        global_instance.all_teams[player["club_team_2_id"]].comp_seasons[
                            player["competition_id"]
                        ].append(new_player)
                    except Exception:
                        print(
                            f"Found player [{new_player.full_name}, {new_comp_season.name} "
                            f"{str(new_comp_season.season)}] playing for unknown second team "
                            f"[{str(player['club_team_2_id'])}]. Skipping..."
                        )
                        continue
                global_instance.all_players[player["id"]] = new_player
        print(
            f"All {str(len(all_rows))} players in [{new_comp_season.name}, "
            f"{str(new_comp_season.season)}] were successfully retrieved."
        )

        # 4. League Matches
        request_string = sett.FS_HOST + "/league-matches?key=" + sett.FS_KEY + "&season_id=" + str(new_comp_season.id)
        res = requests.get(request_string)
        res_data = res.json()

        matches_data: list[dict] = []
        matches_data.extend(res_data["data"])
        pager = res_data.get("pager", {})
        max_page = pager.get("max_page", 1)
        for page in range(2, max_page + 1):
            request_string = (
                sett.FS_HOST
                + "/league-matches?key="
                + sett.FS_KEY
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
                new_match.home_team = global_instance.all_teams[m["homeID"]]
            except Exception:
                print(f"Found non-existing home team ID {str(m['homeID'])}. Skipping match...")
                continue
            try:
                new_match.away_team = global_instance.all_teams[m["awayID"]]
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

            request_string = sett.FS_HOST + "/match?key=" + sett.FS_KEY + "&match_id=" + str(new_match.id)
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
                print(
                    f"Unexpected lineup structure [{new_comp_season.name}, {new_match.season}]. " f"Skipping match..."
                )
                continue
            if not all(isinstance(p.get("player_id"), int) for p in (team_a_lineup + team_b_lineup)):
                print(
                    f"Found a non-integer player ID [{new_comp_season.name}, {new_match.season}]. " f"Skipping match..."
                )
                continue
            home_lineup_player_ids = [x["player_id"] for x in team_a_lineup]
            for p_id in home_lineup_player_ids:
                try:
                    player = global_instance.all_players[p_id]
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
                    player = global_instance.all_players[p_id]
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
            global_instance.all_matches.append(new_match)
            existing_match_ids.add(new_match.id)
            id_to_match[m["id"]] = new_match
            time.sleep(2.5)
        print(
            f"All {str(len(matches_data))} matches [{new_comp_season.name}, "
            f"{str(new_comp_season.season)}] were successfully retrieved."
        )
        time.sleep(30.0)

    return FSDataBundle(
        comp_seasons=global_instance.all_comp_seasons,
        teams=global_instance.all_teams,
        players=global_instance.all_players,
        matches=global_instance.all_matches,
        leagues_list=global_instance.leagues_list,
    )
