from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from rapidfuzz import fuzz

from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import (
    Global,
)
from football_outcomes.data import sofifa_player_matching as _player_matching

_norm_name = _player_matching.normalize_name
_dbg = _player_matching.debug_log


def _norm_country(s: Optional[str]) -> str:
    return _norm_name(s)


def _norm_league(s: Optional[str]) -> str:
    return _norm_name(s)


def _norm_team(s: Optional[str]) -> str:
    return _norm_name(s)


def _try_get_first(rec: dict, keys: list[str]) -> Optional[Any]:
    for k in keys:
        if k in rec and rec[k] not in (None, "", -1):
            return rec[k]
    return None


def _extract_sofifa_team_info(rec: Dict[str, Any]) -> tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
    """
    Returns: (club_id, club_name, league_id, league_name)

    Uses your actual SOFIFA snapshot schema:
      - club_id, club_name
      - club_league_id, club_league_name

    Country is intentionally not used because it's too sparse in your data.
    """
    club_id = rec.get("club_id", None)
    try:
        club_id = int(club_id) if club_id is not None else None
    except Exception:
        club_id = None

    club_name = rec.get("club_name", None)
    if isinstance(club_name, dict):
        club_name = club_name.get("name")

    league_id = rec.get("club_league_id", None)
    try:
        league_id = int(league_id) if league_id is not None else None
    except Exception:
        league_id = None

    league_name = rec.get("club_league_name", None)
    if isinstance(league_name, dict):
        league_name = league_name.get("name")

    return (
        club_id,
        str(club_name) if club_name not in (None, "") else None,
        league_id,
        str(league_name) if league_name not in (None, "") else None,
    )


def build_sofifa_team_indexes(force: bool = False) -> None:
    """
    Builds:
      - g.sofifa_players_by_team: club_id -> list[(sofifa_id, name, full_name, dob_date)]
      - g.sofifa_team_meta: club_id -> {"name":..., "league_id":..., "league":...}
      - g.sofifa_teams_by_league: league_id -> list[(club_id, club_name)]

    IMPORTANT:
    Requires rec to contain:
      club_id, club_name, club_league_id, club_league_name
    """
    g = Global.get_instance()

    if (not force) and getattr(g, "sofifa_players_by_team", None):
        if len(g.sofifa_players_by_team) > 0:
            return

    g.sofifa_players_by_team = {}
    g.sofifa_team_meta = {}
    g.sofifa_teams_by_league = {}

    # sanity: if snapshots don't contain club fields, do nothing but warn
    snaps = getattr(g, "sofifa_snapshots", [])
    if not snaps:
        return

    # quick schema probe
    probe_rec = None
    for _, snap_players in snaps:
        if snap_players:
            probe_rec = next(iter(snap_players.values()))
            break

    if not isinstance(probe_rec, dict) or "club_id" not in probe_rec:
        # You loaded SOFIFA snapshots in old minimal schema (no club info).
        # Team mapping cannot be built from it.
        if getattr(sett, "DEBUG_TEAM_STRENGTH", False):
            _dbg(
                "[sofifa_team_indexes] missing club_id/club_name in sofifa_snapshots records; "
                "rebuild SOFIFA snapshots with club fields."
            )
        return

    seen: set[tuple[int, int]] = set()  # (sofifa_player_id, club_id)

    for snap_date, snap_players in snaps:
        if not snap_players:
            continue

        for sofifa_id, rec in snap_players.items():
            if not isinstance(rec, dict):
                continue

            club_id, club_name, league_id, league_name = _extract_sofifa_team_info(rec)
            if club_id is None or club_name is None:
                continue

            key = (int(sofifa_id), int(club_id))
            if key in seen:
                continue
            seen.add(key)

            # team meta
            meta = g.sofifa_team_meta.get(club_id, {})
            meta.setdefault("name", club_name)

            if league_id is not None:
                s = meta.setdefault("league_ids", set())
                s.add(int(league_id))

                # optional: keep one representative league_id for display/debug
                meta.setdefault("league_id", int(league_id))

            if league_name is not None:
                # optional: keep last seen name, or a set too
                meta.setdefault("league", league_name)

            g.sofifa_team_meta[club_id] = meta

            # player tuple
            sf_name = str(rec.get("name") or "")
            sf_full = str(rec.get("full_name") or sf_name)
            dob = rec.get("dob")
            dob_date = dob if isinstance(dob, date) else None

            g.sofifa_players_by_team.setdefault(club_id, []).append((int(sofifa_id), sf_name, sf_full, dob_date))

    # league -> teams list
    for club_id, meta in g.sofifa_team_meta.items():
        lids = meta.get("league_ids") or set()
        for lid in lids:
            g.sofifa_teams_by_league.setdefault(int(lid), []).append((int(club_id), str(meta.get("name", ""))))

    # sort deterministic
    for lid in list(g.sofifa_teams_by_league.keys()):
        g.sofifa_teams_by_league[lid] = sorted(g.sofifa_teams_by_league[lid], key=lambda x: x[0])


def match_fs_teams_to_sofifa_teams(force: bool = False) -> None:
    """
    Build g.fs_team_to_sofifa_team for teams participating in sett.COMPS_LEAGUE.
    Uses (country, league) filter to avoid e.g. Austria vs Germany Bundesliga.
    """
    g = Global.get_instance()
    if (not force) and getattr(g, "fs_team_to_sofifa_team", None):
        # if already built and non-empty, don't redo
        if len(g.fs_team_to_sofifa_team) > 0:
            return

    build_sofifa_team_indexes(force=False)  # build SOFIFA team/player indexes

    g.fs_team_to_sofifa_team = {}

    # Collect FS teams by (country, comp_name) from league matches
    league_matches = [m for m in getattr(g, "all_matches", []) if getattr(m, "comp_name", None) in sett.COMPS_LEAGUE]
    fs_team_entries: dict[int, tuple[str, str, str]] = {}  # fs_team_id -> (fs_team_name, country, comp_name)

    for m in league_matches:
        if m.home_team is not None:
            fs_team_entries[m.home_team.id] = (m.home_team.name, m.country or "", m.comp_name or "")
        if m.away_team is not None:
            fs_team_entries[m.away_team.id] = (m.away_team.name, m.country or "", m.comp_name or "")

    # Match each FS team to best SOFIFA team within same (country, league)
    for fs_tid, (fs_name, fs_country, fs_league) in fs_team_entries.items():

        # 0) Manual override (strongest rule)
        manual_map = getattr(sett, "FS_TEAM_ID_TO_SOFIFA_TEAM_ID", {})
        if fs_tid in manual_map:
            forced = manual_map[fs_tid]
            if forced in (None, -1):
                g.fs_team_to_sofifa_team[fs_tid] = -1  # sentinel for "no mapping"
                _dbg(f"[team_map][MANUAL] FS '{fs_name}' ({fs_country}/{fs_league}) -> -1 (no sofifa team)")
            else:
                g.fs_team_to_sofifa_team[fs_tid] = int(forced)
                _dbg(f"[team_map][MANUAL] FS '{fs_name}' ({fs_country}/{fs_league}) -> {int(forced)}")
            continue

        # 1) FS league name -> SOFIFA league_id
        sofifa_league_id = sett.FS_LEAGUE_TO_SOFIFA_LEAGUE_ID.get(fs_league)
        if sofifa_league_id is None:
            _dbg(f"[fs->sofifa team] missing league mapping for FS league: {fs_league!r}")
            continue

        candidates = g.sofifa_teams_by_league.get(int(sofifa_league_id), [])
        if not candidates:
            _dbg(f"[fs->sofifa team] no sofifa candidates for league_id={sofifa_league_id} (FS league={fs_league!r})")
            continue

        fs_name_n = _norm_team(fs_name)

        best_team_id = None
        best_score = -1.0
        top_k = []

        for sf_team_id, sf_team_name in candidates:
            s = float(fuzz.ratio(fs_name_n, _norm_team(sf_team_name)))
            top_k.append((s, sf_team_id, sf_team_name))
            if s > best_score:
                best_score = s
                best_team_id = sf_team_id

        top_k.sort(reverse=True, key=lambda x: x[0])

        if getattr(sett, "DEBUG_TEAM_STRENGTH", False):
            _dbg(
                f"[team_map][AUTO] FS '{fs_name}' ({fs_country}/{fs_league}) best="
                f"{best_score:.1f} -> {best_team_id} "
                f"top={top_k[: getattr(sett, 'SF_TEAM_MATCH_MAX_CANDIDATES', 3)]}"
            )

        if best_team_id is not None:
            g.fs_team_to_sofifa_team[fs_tid] = int(best_team_id)
        else:
            g.fs_team_to_sofifa_team[fs_tid] = -1  # leave unmapped (+could add min threshold here)
            if getattr(sett, "DEBUG_TEAM_STRENGTH", False):
                _dbg(f"[team_map][AUTO] FS '{fs_name}' -> -1")


normalize_country = _norm_country
normalize_league = _norm_league
normalize_team = _norm_team
try_get_first = _try_get_first
extract_sofifa_team_info = _extract_sofifa_team_info
