from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from football_outcomes.data.fs_models import FSMatch

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Tuple

from rapidfuzz import fuzz

from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_models import FSPlayer

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


# ---------- helpers: name normalization ----------


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


def _similarity(fs_name: str, fs_full: str, sf_name: str, sf_full: str) -> float:
    # Use max of known_as vs name/full_name and full_name vs name/full_name
    a1 = fuzz.ratio(fs_name, sf_name)
    a2 = fuzz.ratio(fs_name, sf_full)
    b1 = fuzz.ratio(fs_full, sf_name)
    b2 = fuzz.ratio(fs_full, sf_full)
    return float(max(a1, a2, b1, b2))


def _match_fs_to_sofifa(
    fs_player: FSPlayer,
    *,
    dob_bucket: Optional[List[Tuple[int, str, str]]] = None,
) -> MatchResult:
    """
    Implements your desired rule:

    if dob matches and similarity > LOWER: OK
    elif similarity > HIGHER: OK
    else: NOT_OK

    Additionally returns 2nd best similarity to detect ambiguity.
    """
    g = Global.get_instance()

    cached = g.fs_to_sofifa_cache.get(fs_player.id)
    if cached is not None:
        sofifa_id, best, second, used_dob_gate, reason = cached
        return MatchResult(sofifa_id, best, second, used_dob_gate, f"cache:{reason}")

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

    # Candidate sources:
    # 1) DOB bucket if available (fast, usually 10-20 candidates)
    # 2) Otherwise: global scan is too expensive, so we *only* allow no-DOB path
    #    when we can still evaluate some candidates; here we rely on dob_bucket if passed
    used_dob_gate = False

    candidates: List[Tuple[int, str, str, Optional[date]]] = []

    if fs_dob_date is not None:
        dob_list = g.sofifa_players_by_dob.get(fs_dob_date, [])
        if dob_list:
            used_dob_gate = True
            for sofifa_id, sf_name, sf_full in dob_list:
                candidates.append((sofifa_id, sf_name, sf_full, fs_dob_date))

    # If no DOB candidates, we cannot scan entire Sofifa; so we only proceed with
    # whatever dob_bucket passed (optional future extension).
    # For now: no DOB => only accept if dob_bucket provided.
    if not candidates and dob_bucket:
        for sofifa_id, sf_name, sf_full in dob_bucket:
            candidates.append((sofifa_id, sf_name, sf_full, None))

    # Evaluate
    for sofifa_id, sf_name, sf_full, dob in candidates:
        score = _similarity(fs_name, fs_full, _norm_name(sf_name), _norm_name(sf_full))
        if best is None or score > best.score:
            if best is not None:
                second_score = max(second_score, best.score)
            best = MatchCandidate(sofifa_id, sf_name, sf_full, fs_dob_date or date(1900, 1, 1), score)
        else:
            second_score = max(second_score, score)

    if not candidates:
        res = MatchResult(None, 0.0, 0.0, used_dob_gate, "no_candidates")
    elif best is None:
        res = MatchResult(None, 0.0, 0.0, used_dob_gate, "no_best")
    else:
        # Apply your rule
        if used_dob_gate:
            if best.score >= sett.SF_MATCH_LOWER_THRESHOLD:
                res = MatchResult(best.sofifa_id, best.score, second_score, True, "dob_gate_pass")
            else:
                res = MatchResult(None, best.score, second_score, True, "dob_gate_fail")
        else:
            # No DOB gate: require higher threshold
            if best.score >= sett.SF_MATCH_HIGHER_THRESHOLD:
                res = MatchResult(best.sofifa_id, best.score, second_score, False, "high_threshold_pass")
            else:
                res = MatchResult(None, best.score, second_score, False, "high_threshold_fail")

    # Cache write
    g.fs_to_sofifa_cache[fs_player.id] = (
        res.sofifa_id,
        res.score_best,
        res.score_second,
        res.used_dob_gate,
        res.reason,
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
) -> List[float]:
    """
    Get a 34-length skills vector for sofifa_id by:
      - selecting snapshot candidates around match date (past-first, then future)
      - starting with closest snapshot’s skills
      - filling missing skill cells (-1) by scanning next snapshots in order
    No imputation beyond using another snapshot; if still missing => -1 remains.
    """
    g = Global.get_instance()

    match_date = match_dt.date()
    occ = g.sofifa_player_occurrences.get(sofifa_id, [])
    if not occ:
        return [-1.0] * len(sett.PLAYER_SKILLS)

    candidates = _ordered_snapshot_candidates(occ, match_date)
    if not candidates:
        return [-1.0] * len(sett.PLAYER_SKILLS)

    out = [-1.0] * len(sett.PLAYER_SKILLS)

    for snap_idx, snap_date in candidates:
        snap_players = g.sofifa_snapshots[snap_idx][1]  # (date, dict)
        rec = snap_players.get(sofifa_id)
        if rec is None:
            continue
        skills = rec.get("skills")
        if not skills or len(skills) != len(sett.PLAYER_SKILLS):
            continue

        # Fill missing cells only
        for i, v in enumerate(skills):
            if out[i] == -1.0 and v is not None:
                out[i] = float(v)

        if -1.0 not in out:
            break

    return out


# ---------- helpers: lineup handling ----------

_FS_POS_ORDER = {
    "Goalkeeper": 0,
    "Defender": 1,
    "Midfielder": 2,
    "Forward": 3,
}


def _pos_rank(p: FSPlayer) -> int:
    return _FS_POS_ORDER.get(p.position or "", 99)


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
        # Defensive: if FSPlayer has no birthday, we still allow HIGH threshold path only if
        # we have candidates (we don't scan globally). For now, that means it will likely fail.
        mr = _match_fs_to_sofifa(p)

        if mr.sofifa_id is None:
            skills = [-1.0] * len(sett.PLAYER_SKILLS)
            if getattr(sett, "DEBUG_TEAM_STRENGTH", False):
                _dbg(
                    f"[team_strength] UNMATCHED fs='{_player_display_name(p)}' "
                    f"dob={getattr(p, 'birthday', None)} score={mr.score_best:.1f} "
                    f"reason={mr.reason}"
                )
        else:
            skills = _merge_skills_from_snapshots(mr.sofifa_id, curr_match.datetime)
            if getattr(sett, "DEBUG_TEAM_STRENGTH", False):
                missing_cells = sum(1 for x in skills if x == -1.0)
                _dbg(
                    f"[team_strength] MATCH fs='{_player_display_name(p)}' -> sf_id={mr.sofifa_id} "
                    f"score={mr.score_best:.1f} (2nd={mr.score_second:.1f}) "
                    f"missing={missing_cells}/{len(skills)} gate={'dob' if mr.used_dob_gate else 'high'}"
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
