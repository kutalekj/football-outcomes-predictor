from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from rapidfuzz import fuzz

from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import (
    Global,
)
from football_outcomes.data.fs_models import (
    FSPlayer,
)

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


get_team_strength_log_path = _get_team_strength_log_path
debug_log = _dbg
normalize_name = _norm_name
player_display_name = _player_display_name
similarity = _similarity
name_key_last_first_initial = _name_key_last_firstinit
ensure_sofifa_namekey_index = _ensure_sofifa_namekey_index
build_name_bucket = _build_name_bucket
match_fs_to_sofifa = _match_fs_to_sofifa
