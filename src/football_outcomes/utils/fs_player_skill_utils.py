from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from football_outcomes.data.fs_models import FSMatch

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_models import FSPlayer
from football_outcomes.utils.fs_common import normalize_fs_player_position

_debug_log_path: Optional[str] = None


def _get_team_strength_log_path() -> str:
    """
    Lazily create a log path (once per run).
    Writes to: <project_root>/logs/team_strength_debug_YYYYMMDD_HHMMSS.log
    """
    global _debug_log_path
    if _debug_log_path is not None:
        return _debug_log_path

    # Try to create a local logs/ directory relative to current working dir
    # logs_dir = os.path.join(os.getcwd(), "logs")
    logs_dir = sett.LOG_DIR
    os.makedirs(logs_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _debug_log_path = os.path.join(logs_dir, f"team_strength_debug_{ts}.log")
    return _debug_log_path


def _dbg(msg: str) -> None:
    """Write debug line to file if enabled; never prints to console."""
    if not getattr(sett, "DEBUG_TEAM_STRENGTH", False):
        return
    path = _get_team_strength_log_path()
    # Use append with explicit UTF-8; line-buffering behavior is fine for a debug log
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip("\n") + "\n")


# ---------- helpers: name normalization ETC. ----------


def _norm_name(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    # minimal normalization; you can extend (remove accents, punctuation) if needed
    for ch in [".", ",", "'", '"', "(", ")", "-", "_"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def _player_display_name(p: FSPlayer) -> str:
    # FSPlayer has full_name + known_as; keep it deterministic
    if p.known_as and len(p.known_as.strip()) > 0:
        return p.known_as
    return p.full_name


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


# ---------- helpers: matching ----------


@dataclass
class MatchCandidate:
    sofifa_id: int
    name: str
    full_name: str
    dob: date
    score: float  # [0..100]


@dataclass
class MatchResult:
    sofifa_id: Optional[int]
    score_best: float
    score_second: float
    used_dob_gate: bool
    reason: str  # for debugging
    sofifa_best_name: Optional[str] = None  # default keeps backward compatibility


def _similarity(fs_name: str, fs_full: str, sf_name: str, sf_full: str) -> float:
    # Use max of known_as vs name/full_name and full_name vs name/full_name
    a1 = fuzz.ratio(fs_name, sf_name)
    a2 = fuzz.ratio(fs_name, sf_full)
    b1 = fuzz.ratio(fs_full, sf_name)
    b2 = fuzz.ratio(fs_full, sf_full)
    return float(max(a1, a2, b1, b2))


def _name_key_last_firstinit(full_name_norm: str) -> Optional[str]:
    """
    Key = "<last>|<first_initial>"
    Example: "kevin de bruyne" -> "bruyne|k"
    """
    if not full_name_norm:
        return None
    tokens = [t for t in full_name_norm.split() if len(t) >= 1]
    if len(tokens) == 0:
        return None
    last = tokens[-1]
    first = tokens[0]
    if len(last) < 2 or len(first) < 1:
        return None
    return f"{last}|{first[0]}"


def _ensure_sofifa_namekey_index() -> None:
    """
    Build g.sofifa_players_by_namekey lazily from g.sofifa_players_by_dob.
    Index maps "last|firstInitial" -> list of (sofifa_id, name, full_name).
    """
    g = Global.get_instance()
    if hasattr(g, "sofifa_players_by_namekey") and g.sofifa_players_by_namekey is not None:
        return

    idx: Dict[str, List[Tuple[int, str, str]]] = {}
    seen: Set[int] = set()

    for _, players in g.sofifa_players_by_dob.items():
        for sofifa_id, sf_name, sf_full in players:
            if sofifa_id in seen:
                continue
            seen.add(sofifa_id)
            k = _name_key_last_firstinit(_norm_name(sf_full))
            if k is None:
                continue
            idx.setdefault(k, []).append((sofifa_id, sf_name, sf_full))

    g.sofifa_players_by_namekey = idx


def _build_name_bucket(fs_player: FSPlayer, max_candidates: int = 200) -> List[Tuple[int, str, str]]:
    """
    Name-only shortlist: same (last name + first initial).
    Much narrower than "same last name" and usually more useful.
    """
    g = Global.get_instance()
    _ensure_sofifa_namekey_index()

    fs_full = _norm_name(fs_player.full_name or "")
    fs_disp = _norm_name(_player_display_name(fs_player) or "")

    k = _name_key_last_firstinit(fs_full) or _name_key_last_firstinit(fs_disp)
    if k is None:
        return []

    candidates = g.sofifa_players_by_namekey.get(k, [])
    if not candidates:
        return []

    return candidates[:max_candidates]


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


def _match_fs_to_sofifa(
    fs_player: FSPlayer,
    *,
    fs_team_id: Optional[int] = None,
    dob_bucket: Optional[List[Tuple[int, str, str]]] = None,
) -> MatchResult:
    """
    Implements your desired rule:

    if dob matches and similarity > LOWER: OK
    elif similarity > HIGHER: OK
    else: NOT_OK

    Additionally, returns 2nd best similarity to detect ambiguity.
    """
    g = Global.get_instance()

    cached = g.fs_to_sofifa_cache.get(fs_player.id)

    use_cache = getattr(sett, "USE_FS_TO_SOFIFA_CACHE", True)
    if use_cache and cached is not None:
        if len(cached) == 5:
            sofifa_id, best, second, used_dob_gate, reason = cached
            sofifa_best_name = None
        elif len(cached) == 6:
            sofifa_id, best, second, used_dob_gate, reason, sofifa_best_name = cached
        else:
            cached = None  # Bad/corrupt cache entry -> ignore and recompute

        # Decide whether to trust cache or retry matching
        retry_failed = getattr(sett, "FS_TO_SOFIFA_CACHE_RETRY_FAILED", True)
        retry_ambig = getattr(sett, "FS_TO_SOFIFA_CACHE_RETRY_AMBIGUOUS", True)
        min_margin = float(getattr(sett, "FS_TO_SOFIFA_CACHE_MIN_MARGIN", 0.0))
        trust_reasons = set(
            getattr(sett, "FS_TO_SOFIFA_CACHE_ONLY_TRUST_REASONS", {"dob_gate_pass", "high_threshold_pass"})
        )

        margin = float(best) - float(second) if (best is not None and second is not None) else 999.0
        is_success = sofifa_id is not None and reason in trust_reasons
        is_failed = sofifa_id is None
        is_ambiguous = (sofifa_id is not None) and (margin < min_margin)

        # 1) successful + not ambiguous => trust cache
        if is_success and not (retry_ambig and is_ambiguous):
            return MatchResult(
                sofifa_id,
                float(best),
                float(second),
                bool(used_dob_gate),
                f"cache:{reason}",
                sofifa_best_name=sofifa_best_name,
            )

        # 2) failed => retry if enabled
        if is_failed and not retry_failed:
            return MatchResult(
                None,
                float(best),
                float(second),
                bool(used_dob_gate),
                f"cache:{reason}",
                sofifa_best_name=sofifa_best_name,
            )

        # Otherwise: fall through and recompute, and overwrite cache below.

    fs_name = _norm_name(_player_display_name(fs_player))
    fs_full = _norm_name(fs_player.full_name)

    fs_dob = fs_player.birthday  # TODO: Check this
    if fs_dob is not None:
        # normalize to date
        if isinstance(fs_dob, datetime):
            fs_dob_date = fs_dob.date()
        else:
            fs_dob_date = fs_dob
    else:
        fs_dob_date = None

    best: Optional[MatchCandidate] = None
    second_score = -1.0

    # -------- Candidate sources + hierarchy --------
    # Step 1: (Team + DOB) -> easiest threshold
    # Step 2: (Team only)  -> medium threshold
    # Step 3: (DOB only)   -> your current DOB gate
    # Step 4: (Name-only bucket last|firstInitial) -> high threshold

    candidates: List[Tuple[int, str, str, Optional[date]]] = []
    stage = "none"
    used_dob_gate = False

    sf_team_id = None
    if fs_team_id is not None:
        sf_team_id = g.fs_team_to_sofifa_team.get(int(fs_team_id))

    team_players = None
    if sf_team_id not in (None, -1):
        team_players = g.sofifa_players_by_team.get(int(sf_team_id), None)

    # Helper to add candidates from a list[(id,name,full,dob?)] shape
    def _add_from_list(lst, dob_val: Optional[date]) -> None:
        for item in lst:
            if len(item) == 4:
                sofifa_id, sf_name, sf_full, _dob = item
            else:
                sofifa_id, sf_name, sf_full = item
            candidates.append((int(sofifa_id), str(sf_name), str(sf_full), dob_val))

    # STEP 1: team + DOB intersection
    if fs_dob_date is not None and team_players:
        same_dob_team = [(sid, n, f, d) for (sid, n, f, d) in team_players if d == fs_dob_date]
        if same_dob_team:
            stage = "team+dob"
            used_dob_gate = True
            _add_from_list(same_dob_team, fs_dob_date)

    # STEP 2: team only
    if not candidates and team_players:
        stage = "team_only"
        used_dob_gate = False
        _add_from_list(team_players, None)

    # STEP 3: DOB only (your current behavior)
    if not candidates and fs_dob_date is not None:
        dob_list = g.sofifa_players_by_dob.get(fs_dob_date, [])
        if dob_list:
            stage = "dob_only"
            used_dob_gate = True
            _add_from_list(dob_list, fs_dob_date)

    # STEP 4: name-only bucket last|firstInitial
    if not candidates:
        nb = _build_name_bucket(fs_player, max_candidates=getattr(sett, "SF_NAME_BUCKET_MAX", 200))
        if nb:
            stage = "name_bucket"
            used_dob_gate = False
            _add_from_list(nb, None)

    # Evaluate
    for sofifa_id, sf_name, sf_full, _ in candidates:
        score = _similarity(fs_name, fs_full, _norm_name(sf_name), _norm_name(sf_full))
        if best is None or score > best.score:
            if best is not None:
                second_score = max(second_score, best.score)
            best = MatchCandidate(sofifa_id, sf_name, sf_full, fs_dob_date or date(1900, 1, 1), score)
        else:
            second_score = max(second_score, score)

    best_name: Optional[str] = None
    if best is not None:
        best_name = best.name

    if not candidates:
        res = MatchResult(None, 0.0, 0.0, used_dob_gate, "no_candidates", sofifa_best_name=None)
    elif best is None:
        res = MatchResult(None, 0.0, 0.0, used_dob_gate, "no_best", sofifa_best_name=None)
    else:
        if stage == "team+dob":
            if best.score >= sett.SF_MATCH_LOW_THRESHOLD:
                res = MatchResult(
                    best.sofifa_id, best.score, second_score, True, "team_dob_pass", sofifa_best_name=best_name
                )
            else:
                res = MatchResult(None, best.score, second_score, True, "team_dob_fail", sofifa_best_name=best_name)

        elif stage == "team_only":
            if best.score >= sett.SF_MATCH_HIGH_THRESHOLD:
                res = MatchResult(
                    best.sofifa_id, best.score, second_score, False, "team_only_pass", sofifa_best_name=best_name
                )
            else:
                res = MatchResult(None, best.score, second_score, False, "team_only_fail", sofifa_best_name=best_name)

        elif stage == "dob_only":
            # your original DOB gate rule
            if best.score >= sett.SF_MATCH_MODERATE_THRESHOLD:
                res = MatchResult(
                    best.sofifa_id, best.score, second_score, True, "dob_gate_pass", sofifa_best_name=best_name
                )
            else:
                res = MatchResult(None, best.score, second_score, True, "dob_gate_fail", sofifa_best_name=best_name)

        elif stage == "name_bucket":
            # name-only requires high threshold
            if best.score >= sett.SF_MATCH_HIGH_THRESHOLD:
                res = MatchResult(
                    best.sofifa_id, best.score, second_score, False, "name_bucket_pass", sofifa_best_name=best_name
                )
            else:
                res = MatchResult(None, best.score, second_score, False, "name_bucket_fail", sofifa_best_name=best_name)

        else:
            # should not happen, but keep safe
            res = MatchResult(
                None, best.score, second_score, used_dob_gate, "unknown_stage", sofifa_best_name=best_name
            )

    # Cache write
    g.fs_to_sofifa_cache[fs_player.id] = (
        res.sofifa_id,
        res.score_best,
        res.score_second,
        res.used_dob_gate,
        res.reason,
        res.sofifa_best_name,
    )
    return res


# ---------- helpers: skill retrieval across snapshots ----------


def _ordered_snapshot_candidates(occurrences: List[Tuple[int, date]], match_date: date) -> List[Tuple[int, date]]:
    """
    Order snapshots as:
      1) past snapshots closest first (match_date - snap_date >= 0), by abs diff ascending
      2) future snapshots closest first, by abs diff ascending
    Keep only within +/- SF_MAX_TIMEDELTA_DAYS.
    Limit to SF_MAX_SNAPSHOTS_TO_SCAN.
    """
    max_days = sett.SF_MAX_TIMEDELTA_DAYS

    past = []
    future = []
    for idx, d in occurrences:
        dd = (match_date - d).days
        if abs(dd) > max_days:
            continue
        if dd >= 0:
            past.append((abs(dd), idx, d))
        else:
            future.append((abs(dd), idx, d))

    past.sort(key=lambda x: x[0])
    future.sort(key=lambda x: x[0])

    ordered = [(idx, d) for _, idx, d in past] + [(idx, d) for _, idx, d in future]
    return ordered[: sett.SF_MAX_SNAPSHOTS_TO_SCAN]


def _merge_skills_from_snapshots(
    sofifa_id: int,
    match_dt: datetime,
) -> Tuple[List[float], int, int]:
    """
    Returns:
      - skills: 34-length vector
      - snapshots_used: number of snapshots that contributed ≥1 value
      - closest_delta_days: (match_date - snapshot_date) of first contributing snapshot
    """
    g = Global.get_instance()

    match_date = match_dt.date()
    occ = g.sofifa_player_occurrences.get(sofifa_id, [])
    if not occ:
        return [-1.0] * len(sett.PLAYER_SKILLS), 0, 0

    candidates = _ordered_snapshot_candidates(occ, match_date)
    if not candidates:
        return [-1.0] * len(sett.PLAYER_SKILLS), 0, 0

    out = [-1.0] * len(sett.PLAYER_SKILLS)

    snapshots_used = 0
    closest_delta_days = None

    for snap_idx, snap_date in candidates:
        snap_players = g.sofifa_snapshots[snap_idx][1]  # (date, dict)
        rec = snap_players.get(sofifa_id)
        if rec is None:
            continue

        skills = rec.get("skills")
        if not skills or len(skills) != len(sett.PLAYER_SKILLS):
            continue

        contributed = False
        for i, v in enumerate(skills):
            if out[i] == -1.0 and v is not None:
                out[i] = float(v)
                contributed = True

        if contributed:
            snapshots_used += 1
            if closest_delta_days is None:
                closest_delta_days = (match_date - snap_date).days

        if -1.0 not in out:
            break

    if closest_delta_days is None:
        closest_delta_days = 0

    return out, snapshots_used, closest_delta_days


# ---------- helpers: lineup handling ----------

_FS_POS_ORDER = {
    "Goalkeeper": 0,
    "Defender": 1,
    "Midfielder": 2,
    "Forward": 3,
}


def _pos_rank(p: FSPlayer) -> int:
    pos = normalize_fs_player_position(p.position or "", p.known_as)
    return _FS_POS_ORDER.get(pos, 99)


def _gk_role_score(skills: List[float]) -> float:
    """Positive => looks like GK, negative => looks like outfield."""
    if not skills or len(skills) < sett.GK_SKILL_END_INDEX:
        return 0.0
    gk = skills[sett.GK_SKILL_START_INDEX : sett.GK_SKILL_END_INDEX]
    out = skills[: sett.GK_SKILL_START_INDEX]
    # ignore -1 values
    gk_vals = [x for x in gk if x != -1.0]
    out_vals = [x for x in out if x != -1.0]
    if not gk_vals or not out_vals:
        return 0.0
    return (sum(gk_vals) / len(gk_vals)) - (sum(out_vals) / len(out_vals))


def _ensure_one_goalkeeper_row(rows: List[Tuple[FSPlayer, List[float]]]) -> List[Tuple[FSPlayer, List[float]]]:
    """
    Enforce exactly one GK row if sett.FORCE_EXACTLY_ONE_GK_ROW.

    Strategy:
      - Prefer FSPosition == Goalkeeper for GK slot.
      - If none, choose the row with the highest GK role score.
      - If still ambiguous or all missing, insert a missing GK row at front.
      - Ensure other rows remain in stable order.
    """
    if not getattr(sett, "FORCE_EXACTLY_ONE_GK_ROW", True):
        return rows

    # Find candidate GK rows
    gk_rows = [(i, p, s) for i, (p, s) in enumerate(rows) if (p.position == "Goalkeeper")]
    if gk_rows:
        # pick the first GK in order, others are treated as normal outfield rows (still kept)
        best_i, _, _ = gk_rows[0]
        # move best GK to front
        chosen = rows.pop(best_i)
        rows.insert(0, chosen)
        return rows

    # No explicit GK in FS lineup, choose by skill signature
    scored = []
    for i, (p, s) in enumerate(rows):
        scored.append((i, _gk_role_score(s)))
    scored.sort(key=lambda x: x[1], reverse=True)
    if scored and scored[0][1] >= getattr(sett, "GK_ROLE_SCORE_MIN_GAP", 0.5):
        best_i = scored[0][0]
        chosen = rows.pop(best_i)
        rows.insert(0, chosen)
        return rows

    # Cannot find a GK-like player: insert a missing GK row
    missing_player = FSPlayer(-1, "MISSING_GK", "", "", "", "MISSING_GK")
    missing_player.position = "Goalkeeper"
    rows.insert(0, (missing_player, [-1.0] * len(sett.PLAYER_SKILLS)))
    return rows


def _select_and_sort_lineup(curr_match: "FSMatch", team_id: int) -> tuple[list[FSPlayer], str]:
    """
    Return (sorted_lineup, side) for the requested team in the match.

    The ordering mirrors calculate_team_strength():
      - lineup chosen by side
      - sorted by coarse FS position
      - padded/truncated to exactly TEAM_STRENGTH_NUM_PLAYERS
    """
    if curr_match.home_team is not None and curr_match.home_team.id == team_id:
        lineup = getattr(curr_match, "home_lineup", None)
        side = "home"
    elif curr_match.away_team is not None and curr_match.away_team.id == team_id:
        lineup = getattr(curr_match, "away_lineup", None)
        side = "away"
    else:
        raise ValueError(f"Team {team_id} not in match {curr_match.id}")

    if lineup is None:
        lineup = []
    elif not isinstance(lineup, list):
        raise TypeError(f"Lineup must be list[FSPlayer], got {type(lineup)}")

    if len(lineup) > sett.TEAM_STRENGTH_NUM_PLAYERS:
        raise ValueError(f"Lineup has >{sett.TEAM_STRENGTH_NUM_PLAYERS} players: {len(lineup)}")

    lineup_sorted = sorted(lineup, key=_pos_rank)

    # If no explicit goalkeeper is present, insert one to keep row structure aligned.
    has_gk = any(normalize_fs_player_position(p.position or "", p.known_as) == "Goalkeeper" for p in lineup_sorted)
    if not has_gk:
        missing_gk = FSPlayer(-1, "MISSING_GK", "", "", "", "MISSING_GK")
        missing_gk.position = "Goalkeeper"
        lineup_sorted.insert(0, missing_gk)

    while len(lineup_sorted) < sett.TEAM_STRENGTH_NUM_PLAYERS:
        mp = FSPlayer(-1, "MISSING", "", "", "", "MISSING")
        mp.position = "Unknown"
        lineup_sorted.append(mp)

    lineup_sorted = lineup_sorted[: sett.TEAM_STRENGTH_NUM_PLAYERS]
    return lineup_sorted, side


def calculate_team_position_indices(curr_match: "FSMatch", team_id: int) -> List[int]:
    """
    Build a TEAM_STRENGTH_NUM_PLAYERS-length vector of coarse FS player-position indices.

    The returned order is aligned with the team-strength rows:
      - lineup selected by side
      - sorted by coarse position (GK/DEF/MID/FWD)
      - missing rows padded with "Unknown"
    """
    lineup_sorted, _ = _select_and_sort_lineup(curr_match, team_id)

    pos_idxs: List[int] = []
    for p in lineup_sorted:
        pos = normalize_fs_player_position(getattr(p, "position", "") or "", p.known_as)
        pos_idxs.append(int(sett.FS_PLAYER_POSITION_TO_IDX[pos]))

    return pos_idxs


# ---------- main: calculate_team_strength ----------


def calculate_team_strength(curr_match: "FSMatch", team_id: int) -> list[list[float]]:
    """
    Returns 11x34 matrix of player skills for team_id in curr_match.

    - Sort lineup by FS position group (GK/DEF/MID/FWD).
    - Match FSPlayer to Sofifa:
        * If DOB matches: accept if similarity >= LOWER threshold
        * Else accept only if similarity >= HIGHER threshold
    - Build skills vector:
        * Start from the closest past snapshot, then fill missing from other snapshots
        * Allow looking into near future snapshots (past-first), within +/- window
    - Missing player or missing skills remain -1.
    - Pad/truncate to exactly 11 players.
    - Optionally enforce exactly one GK row (no imputation; missing GK becomes [-1]*34 GK row).
    """
    # --- pick lineup
    # Adjust these attribute names to your FSMatch:
    # I assume you now store actual FSPlayer objects lists.
    if curr_match.home_team is not None and curr_match.home_team.id == team_id:
        lineup = getattr(curr_match, "home_lineup", None)
        side = "home"
    elif curr_match.away_team is not None and curr_match.away_team.id == team_id:
        lineup = getattr(curr_match, "away_lineup", None)
        side = "away"
    else:
        raise ValueError(f"Team {team_id} not in match {curr_match.id}")

    # Log None vs [] difference as you requested
    if lineup is None:
        if getattr(sett, "DEBUG_TEAM_STRENGTH", False):
            _dbg(f"[team_strength] lineup=None for team_id={team_id} match={curr_match.id} ({side})")

        lineup = []
    elif isinstance(lineup, list) and len(lineup) == 0:
        if getattr(sett, "DEBUG_TEAM_STRENGTH", False):
            _dbg(f"[team_strength] lineup=[] for team_id={team_id} match={curr_match.id} ({side})")

    if not isinstance(lineup, list):
        raise TypeError(f"Lineup must be list[FSPlayer], got {type(lineup)}")

    if len(lineup) > sett.TEAM_STRENGTH_NUM_PLAYERS:
        # You can decide to truncate; but I’d rather fail fast because it indicates bad upstream data
        raise ValueError(f"Lineup has >{sett.TEAM_STRENGTH_NUM_PLAYERS} players: {len(lineup)}")

    # --- stable ordering
    lineup_sorted = sorted(lineup, key=_pos_rank)

    # --- match and pull skills
    rows: List[Tuple[FSPlayer, List[float]]] = []

    for p in lineup_sorted:
        mr = _match_fs_to_sofifa(p, fs_team_id=team_id)

        if mr.sofifa_id is None:
            skills = [-1.0] * len(sett.PLAYER_SKILLS)
            if getattr(sett, "DEBUG_TEAM_STRENGTH", False):
                _dbg(
                    f"[team_strength] UNMATCHED fs='{_player_display_name(p)}' "
                    f"dob={getattr(p, 'birthday', None)} sf_name={mr.sofifa_best_name} score={mr.score_best:.1f} "
                    f"reason={mr.reason}"
                )
        else:
            skills, snapshots_used, delta_days = _merge_skills_from_snapshots(mr.sofifa_id, curr_match.datetime)
            if getattr(sett, "DEBUG_TEAM_STRENGTH", False):
                missing_cells = sum(1 for x in skills if x == -1.0)
                _dbg(
                    f"[team_strength] MATCH fs='{_player_display_name(p)}' -> sf_id={mr.sofifa_id} "
                    f"score={mr.score_best:.1f} (2nd={mr.score_second:.1f}) "
                    f"(sf_name={mr.sofifa_best_name}) "
                    f"league={curr_match.comp_name.replace(' ', '_')} "
                    f"match_dt={curr_match.datetime.isoformat()} "
                    f"missing={missing_cells}/{len(skills)} "
                    f"snapshots_used={snapshots_used} "
                    f"delta_days={delta_days} "
                    f"reason={mr.reason}"
                )

        rows.append((p, skills))

    # --- enforce goalkeeper presence/slot (no imputation, only reordering / missing row)
    rows = _ensure_one_goalkeeper_row(rows)

    # --- pad/truncate to 11 rows (pad with missing players)
    while len(rows) < sett.TEAM_STRENGTH_NUM_PLAYERS:
        mp = FSPlayer(-1, "MISSING", "", "", "", "MISSING")
        mp.position = "Unknown"
        rows.append((mp, [-1.0] * len(sett.PLAYER_SKILLS)))

    rows = rows[: sett.TEAM_STRENGTH_NUM_PLAYERS]

    # return matrix only
    return [skills for _, skills in rows]
