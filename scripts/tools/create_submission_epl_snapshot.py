from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_io import load_snapshot, save_snapshot
from football_outcomes.data.fs_models import FSDataBundle

TARGET_COMPETITION = "England Premier League"
TARGET_SOFIFA_LEAGUE_ID = 13
TARGET_SOFIFA_LEAGUE_NAME = "Premier League"


def _match_in_epl(match: Any) -> bool:
    return getattr(match, "comp_name", None) == TARGET_COMPETITION and getattr(match, "season", None) in {
        2021,
        2022,
        2023,
        2024,
    }


def _collect_team_ids(matches: list[Any]) -> set[int]:
    out: set[int] = set()
    for m in matches:
        if getattr(m, "home_team", None) is not None:
            out.add(int(m.home_team.id))
        if getattr(m, "away_team", None) is not None:
            out.add(int(m.away_team.id))
    return out


def _collect_player_ids(matches: list[Any]) -> set[int]:
    out: set[int] = set()
    for m in matches:
        for attr in ("home_lineup", "away_lineup", "home_players", "away_players"):
            players = getattr(m, attr, None)
            if not players:
                continue
            for p in players:
                pid = getattr(p, "id", None)
                if pid is not None:
                    out.add(int(pid))
    return out


def _filter_sofifa_snapshots(snapshots):
    """
    Keep only SOFIFA records from the English Premier League (EPL).
    Uses the SOFIFA fields club_league_id=13 and/or club_league_name='Premier League'.
    """
    filtered = []

    for snap_date, players in snapshots:
        kept = {}
        for sofifa_id, rec in players.items():
            if not isinstance(rec, dict):
                continue

            league_id = rec.get("club_league_id")
            league_name = str(rec.get("club_league_name") or "")

            try:
                league_id_int = int(league_id) if league_id is not None else None
            except Exception:
                league_id_int = None

            if league_id_int == TARGET_SOFIFA_LEAGUE_ID or league_name == TARGET_SOFIFA_LEAGUE_NAME:
                kept[int(sofifa_id)] = rec

        if kept:
            filtered.append((snap_date, kept))

    return filtered


def _filter_sofifa_player_occurrences(occurrences, kept_sofifa_ids: set[int]):
    return {int(pid): occ for pid, occ in occurrences.items() if int(pid) in kept_sofifa_ids}


def _filter_sofifa_players_by_dob(players_by_dob, kept_sofifa_ids: set[int]):
    out = {}

    for dob, triples in players_by_dob.items():
        kept = []
        for triple in triples:
            if not triple:
                continue
            sofifa_id = int(triple[0])
            if sofifa_id in kept_sofifa_ids:
                kept.append(triple)

        if kept:
            out[dob] = kept

    return out


def _filter_sofifa_team_meta(team_meta):
    out = {}

    for club_id, meta in team_meta.items():
        league_id = meta.get("league_id")
        league_ids = meta.get("league_ids", set())
        league_name = str(meta.get("league") or "")

        try:
            league_id_int = int(league_id) if league_id is not None else None
        except Exception:
            league_id_int = None

        league_ids_int = set()
        for x in league_ids or []:
            try:
                league_ids_int.add(int(x))
            except Exception:
                pass

        if (
            league_id_int == TARGET_SOFIFA_LEAGUE_ID
            or TARGET_SOFIFA_LEAGUE_ID in league_ids_int
            or league_name == TARGET_SOFIFA_LEAGUE_NAME
        ):
            out[int(club_id)] = meta

    return out


def _filter_sofifa_players_by_team(players_by_team, kept_team_ids: set[int], kept_sofifa_ids: set[int]):
    out = {}

    for club_id, players in players_by_team.items():
        cid = int(club_id)
        if cid not in kept_team_ids:
            continue

        kept = []
        for row in players:
            if row and int(row[0]) in kept_sofifa_ids:
                kept.append(row)

        if kept:
            out[cid] = kept

    return out


def create_epl_sample_snapshot(input_path: Path, output_path: Path) -> None:
    print(f"[load] {input_path}")
    bundle = load_snapshot(input_path)

    matches = [m for m in bundle.matches if _match_in_epl(m)]
    matches.sort(key=lambda m: (m.datetime, m.hour_utc if m.hour_utc is not None else -1, m.id))

    if not matches:
        raise RuntimeError("No English Premier League matches found in the source snapshot.")

    fs_team_ids = _collect_team_ids(matches)
    fs_player_ids = _collect_player_ids(matches)

    comp_season_ids = {
        int(getattr(m, "comp_season_id")) for m in matches if getattr(m, "comp_season_id", None) is not None
    }

    comp_seasons = {int(cid): cs for cid, cs in bundle.comp_seasons.items() if int(cid) in comp_season_ids}

    # EPL teams
    teams = {int(tid): t for tid, t in bundle.teams.items() if int(tid) in fs_team_ids}

    # FS players appearing in matches from the bundle
    players = {int(pid): p for pid, p in bundle.players.items() if int(pid) in fs_player_ids}

    sofifa_snapshots = _filter_sofifa_snapshots(getattr(bundle, "sofifa_snapshots", []))
    kept_sofifa_ids = {int(pid) for _, snap_players in sofifa_snapshots for pid in snap_players.keys()}

    sofifa_player_occurrences = _filter_sofifa_player_occurrences(
        getattr(bundle, "sofifa_player_occurrences", {}),
        kept_sofifa_ids,
    )
    sofifa_players_by_dob = _filter_sofifa_players_by_dob(
        getattr(bundle, "sofifa_players_by_dob", {}),
        kept_sofifa_ids,
    )

    sofifa_team_meta = _filter_sofifa_team_meta(getattr(bundle, "sofifa_team_meta", {}))
    kept_sofifa_team_ids = set(sofifa_team_meta.keys())

    sofifa_players_by_team = _filter_sofifa_players_by_team(
        getattr(bundle, "sofifa_players_by_team", {}),
        kept_sofifa_team_ids,
        kept_sofifa_ids,
    )

    fs_team_to_sofifa_team = {
        int(fs_id): int(sf_id)
        for fs_id, sf_id in getattr(bundle, "fs_team_to_sofifa_team", {}).items()
        if int(fs_id) in fs_team_ids and int(sf_id) in kept_sofifa_team_ids
    }

    fs_to_sofifa_cache = {
        int(fs_id): value
        for fs_id, value in getattr(bundle, "fs_to_sofifa_cache", {}).items()
        if int(fs_id) in fs_player_ids
    }

    sofifa_teams_by_league = {
        TARGET_SOFIFA_LEAGUE_ID: [
            row
            for row in getattr(bundle, "sofifa_teams_by_league", {}).get(TARGET_SOFIFA_LEAGUE_ID, [])
            if int(row[0]) in kept_sofifa_team_ids
        ]
    }

    out = FSDataBundle(
        comp_seasons=comp_seasons,
        teams=teams,
        players=players,
        matches=matches,
        leagues_list=getattr(bundle, "leagues_list", None),
        sofifa_snapshots=sofifa_snapshots,
        sofifa_player_occurrences=sofifa_player_occurrences,
        sofifa_players_by_dob=sofifa_players_by_dob,
        fs_to_sofifa_cache=fs_to_sofifa_cache,
        sofifa_team_meta=sofifa_team_meta,
        sofifa_players_by_team=sofifa_players_by_team,
        sofifa_teams_by_league=sofifa_teams_by_league,
        fs_team_to_sofifa_team=fs_team_to_sofifa_team,
    )

    out.meta.update(getattr(bundle, "meta", {}) or {})
    out.meta["description"] = "Submission sample snapshot: English Premier League, seasons 2021-2024."
    out.meta["source_snapshot"] = str(input_path)
    out.meta["competition"] = TARGET_COMPETITION
    out.meta["seasons"] = [2021, 2022, 2023, 2024]
    out.meta["num_matches"] = len(matches)
    out.meta["num_teams"] = len(teams)
    out.meta["num_fs_players"] = len(players)
    out.meta["num_sofifa_snapshots"] = len(sofifa_snapshots)

    print("[sample]")
    print(f"  matches: {len(matches)}")
    print(f"  comp seasons: {len(comp_seasons)}")
    print(f"  teams: {len(teams)}")
    print(f"  FS players: {len(players)}")
    print(f"  SOFIFA snapshots: {len(sofifa_snapshots)}")
    print(f"  SOFIFA players kept: {len(kept_sofifa_ids)}")
    print(f"  FS->SOFIFA team mappings: {len(fs_team_to_sofifa_team)}")
    print(f"[save] {output_path}")

    save_snapshot(out, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reduced EPL sample snapshot for thesis submission.")
    parser.add_argument(
        "--input",
        type=Path,
        default=sett.LOAD_SNAPSHOT_PATH,
        help="Path to the full source snapshot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(sett.PROJECT_ROOT) / "data" / "submission" / "epl_sample_snapshot.pkl",
        help="Output path for the reduced submission snapshot.",
    )
    args = parser.parse_args()

    create_epl_sample_snapshot(args.input, args.output)


if __name__ == "__main__":
    main()
