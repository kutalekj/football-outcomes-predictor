from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import FSMatch


def match_sort_key(m: FSMatch) -> Tuple:
    # datetime is at UTC 00:00, so hour_utc is needed for intraday ordering
    h = m.hour_utc if m.hour_utc is not None else -1
    return m.datetime, h, m.id


def clip01(x: float) -> float:
    if x <= 0.0:
        return sett.ALMOST_ZERO
    if x >= 1.0:
        return sett.ALMOST_ONE
    return x


def min_max_scaling_with_clipping(value: float, max_value: float) -> float:
    if max_value <= 0:
        return sett.ALMOST_ZERO
    return clip01(float(value) / float(max_value))


def normalize_points(points_per_game: float) -> float:
    # points_per_game is in [0, 3]
    return clip01(points_per_game / 3.0)


def normalize_goals(goals_per_game: float) -> float:
    return min_max_scaling_with_clipping(goals_per_game, sett.GOALS_NORM_COEFFICIENT)


def normalize_sog(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.SOG_NORM_COEFFICIENT)


def normalize_total_shots(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.TOTAL_SHOTS_NORM_COEFFICIENT)


def normalize_shots_in_box(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.SHOTS_IN_BOX_NORM_COEFFICIENT)


def normalize_corners(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.CORNER_KICKS_NORM_COEFFICIENT)


def normalize_match_load(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.MATCH_LOAD_NORM_COEFFICIENT)


def normalize_team_xg(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.TEAM_XG_NORM_COEFFICIENT)


def normalize_total_xg(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.TOTAL_XG_NORM_COEFFICIENT)


def normalize_team_pre_match_xg(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.TEAM_PRE_MATCH_XG_NORM_COEFFICIENT)


def normalize_total_pre_match_xg(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.TOTAL_PRE_MATCH_XG_NORM_COEFFICIENT)


def normalize_elo(raw_elo: float, min_elo: float = 1000.0, max_elo: float = 2000.0) -> float:
    return clip01((raw_elo - min_elo) / (max_elo - min_elo))


def hour_month_cyclic(hour_utc: int, month: int) -> Tuple[float, float, float, float]:
    # hour in [0..23], month in [1..12]
    hour_angle = 2.0 * math.pi * (float(hour_utc) / 24.0)
    month_angle = 2.0 * math.pi * ((float(month) - 1.0) / 12.0)
    return (
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(month_angle),
        math.cos(month_angle),
    )


@dataclass
class TeamMatchIndex:
    matches_by_team: Dict[int, List[FSMatch]]
    idx_by_match_id_for_team: Dict[Tuple[int, int], int]  # (team_id, match_id) -> index in matches_by_team[team_id]


def build_team_match_index(all_matches: List[FSMatch]) -> TeamMatchIndex:
    matches_by_team: Dict[int, List[FSMatch]] = defaultdict(list)
    for m in all_matches:
        if m.home_team is not None:
            matches_by_team[m.home_team.id].append(m)
        if m.away_team is not None:
            matches_by_team[m.away_team.id].append(m)

    for team_id in list(matches_by_team.keys()):
        matches_by_team[team_id].sort(key=match_sort_key)

    idx_map: Dict[Tuple[int, int], int] = {}
    for team_id, ms in matches_by_team.items():
        for i, m in enumerate(ms):
            idx_map[(team_id, m.id)] = i

    return TeamMatchIndex(matches_by_team=dict(matches_by_team), idx_by_match_id_for_team=idx_map)


def get_previous_match(index: TeamMatchIndex, team_id: int, curr_match: FSMatch) -> Optional[FSMatch]:
    i = index.idx_by_match_id_for_team.get((team_id, curr_match.id))
    if i is None or i <= 0:
        return None
    return index.matches_by_team[team_id][i - 1]


def get_n_previous_matches(index: TeamMatchIndex, team_id: int, curr_match: FSMatch, n: int) -> List[FSMatch]:
    i = index.idx_by_match_id_for_team.get((team_id, curr_match.id))
    if i is None:
        return []
    start = max(0, i - n)
    return index.matches_by_team[team_id][start:i]


def _is_team_home(m: FSMatch, team_id: int) -> bool:
    return (m.home_team is not None) and (m.home_team.id == team_id)


def _is_team_away(m: FSMatch, team_id: int) -> bool:
    return (m.away_team is not None) and (m.away_team.id == team_id)


def get_n_previous_role_matches(
    index: TeamMatchIndex, team_id: int, curr_match: FSMatch, n: int, role: str
) -> List[FSMatch]:
    """Get previous matches where the team played in the given role ('home' or 'away')."""
    previous_matches = get_n_previous_matches(index, team_id, curr_match, n=500)  # get more then filter
    if role == "home":
        role_prevs = [m for m in previous_matches if _is_team_home(m, team_id)]
    elif role == "away":
        role_prevs = [m for m in previous_matches if _is_team_away(m, team_id)]
    else:
        raise ValueError("role must be 'home' or 'away'")
    return role_prevs[-n:]  # last n among role-matches


def avg_goals_scored_conceded_role_last_n(
    index: TeamMatchIndex,
    team_id: int,
    curr_match: FSMatch,
    n: int,
    role: str,
) -> Tuple[float, float]:
    ms = get_n_previous_role_matches(index, team_id, curr_match, n, role)
    scored = []
    conceded = []
    for m in ms:
        s, c = _team_goals_scored_conceded(m, team_id)
        if s is None or c is None:
            continue
        scored.append(s)
        conceded.append(c)
    if not scored:
        return sett.ALMOST_ZERO, sett.ALMOST_ZERO
    return normalize_goals(sum(scored) / len(scored)), normalize_goals(sum(conceded) / len(conceded))


def is_within_days(curr_match: FSMatch, prev_match: FSMatch, days: int) -> bool:
    if curr_match.datetime is None or prev_match.datetime is None:
        return False
    return (curr_match.datetime - prev_match.datetime).days <= days


def match_load_per_day_last_n_days(index: TeamMatchIndex, team_id: int, curr_match: FSMatch, days: int) -> float:
    # Count matches strictly before curr_match within the last N days
    prev_matches = get_n_previous_matches(index, team_id, curr_match, n=200)  # cap; enough for any days window
    cnt = 0
    for m in reversed(prev_matches):
        if is_within_days(curr_match, m, days):
            cnt += 1
        else:
            break
    return normalize_match_load(float(cnt) / float(days))


def _team_points_in_match(m: FSMatch, team_id: int) -> Optional[float]:
    if m.home_team and m.home_team.id == team_id:
        return float(m.home_points) if m.home_points is not None else None
    if m.away_team and m.away_team.id == team_id:
        return float(m.away_points) if m.away_points is not None else None
    return None


def _team_goals_scored_conceded(m: FSMatch, team_id: int) -> Tuple[Optional[float], Optional[float]]:
    if m.home_team and m.home_team.id == team_id:
        return float(m.home_goals), float(m.away_goals)
    if m.away_team and m.away_team.id == team_id:
        return float(m.away_goals), float(m.home_goals)
    return None, None


def _team_stat_value(m: FSMatch, team_id: int, home_key: str, away_key: str) -> Optional[float]:
    if m.home_team and m.home_team.id == team_id:
        v = m.stats.get(home_key, -1)
    elif m.away_team and m.away_team.id == team_id:
        v = m.stats.get(away_key, -1)
    else:
        return None
    if v is None or v == -1:
        return None
    return float(v)


def avg_points_last_n(index: TeamMatchIndex, team_id: int, curr_match: FSMatch, n: int) -> float:
    prev_matches = get_n_previous_matches(index, team_id, curr_match, n)
    vals = [v for v in (_team_points_in_match(m, team_id) for m in prev_matches) if v is not None]
    if not vals:
        return sett.ALMOST_ZERO
    return normalize_points(sum(vals) / float(len(vals)))


def avg_goals_last_n(index: TeamMatchIndex, team_id: int, curr_match: FSMatch, n: int) -> float:
    prev_matches = get_n_previous_matches(index, team_id, curr_match, n)
    scored = []
    for m in prev_matches:
        s, _ = _team_goals_scored_conceded(m, team_id)
        if s is not None:
            scored.append(s)
    if not scored:
        return sett.ALMOST_ZERO
    return normalize_goals(sum(scored) / float(len(scored)))


def avg_stat_last_n(
    index: TeamMatchIndex, team_id: int, curr_match: FSMatch, n: int, home_key: str, away_key: str, norm_fn
) -> float:
    prev_matches = get_n_previous_matches(index, team_id, curr_match, n)
    vals = []
    for m in prev_matches:
        v = _team_stat_value(m, team_id, home_key, away_key)
        if v is not None:
            vals.append(v)
    if not vals:
        return sett.ALMOST_ZERO
    return norm_fn(sum(vals) / float(len(vals)))


def avg_total_stat_last_n(index: TeamMatchIndex, team_id: int, curr_match: FSMatch, n: int, total_fn, norm_fn) -> float:
    # "total" means home+away stat for the match (e.g., total xG)
    prev_matches = get_n_previous_matches(index, team_id, curr_match, n)
    vals = []
    for m in prev_matches:
        v = total_fn(m)
        if v is not None:
            vals.append(v)
    if not vals:
        return sett.ALMOST_ZERO
    return norm_fn(sum(vals) / float(len(vals)))


def total_xg(m: FSMatch) -> Optional[float]:
    hxg = m.stats.get("home_xg", -1)
    axg = m.stats.get("away_xg", -1)
    if hxg in (-1, None) or axg in (-1, None):
        return None
    return float(hxg) + float(axg)


def total_pre_match_xg(m: FSMatch) -> Optional[float]:
    hxg = m.stats.get("home_prematch_xg", -1)
    axg = m.stats.get("away_prematch_xg", -1)
    if hxg in (-1, None) or axg in (-1, None):
        return None
    return float(hxg) + float(axg)


def calculate_elo_for_match(
    team_index_league: TeamMatchIndex, team_index_all: TeamMatchIndex, curr_match: FSMatch
) -> Tuple[float, float]:
    """
    Weighted ELO update.

    Returns:
        (home_elo_pre_norm, away_elo_pre_norm)  -- ELO values that must be used as *pre-match* features.

    Side effects:
        Stores post-match raw ELO on curr_match for propagation:
            curr_match.home_elo_after_match_raw
            curr_match.away_elo_after_match_raw

    Weighting:
      - League matches (curr_match.comp_name in sett.COMPS_LEAGUE): weight = 1.0
      - Non-league matches:
            if BOTH teams are "reliable" (enough league history): weight = sett.ELO_NON_LEAGUE_WEIGHT
            else: weight = 0.0  (ignore match for ELO updates)

    Reliability:
      Uses total league match count available for the team in the dataset:
          len(team_index_league.matches_by_team.get(team_id, []))
      (This is intentionally simple and stable. If later you want "reliable up to date", we can refine.)

    Season transition:
      If previous match season != current match season:
          elo := INIT_ELO + alpha * (elo - INIT_ELO)
      where alpha = sett.ELO_SEASON_REGRESSION (default 1.0 if missing).
    """

    if curr_match.home_team is None or curr_match.away_team is None:
        return sett.ALMOST_ZERO, sett.ALMOST_ZERO

    home_id = curr_match.home_team.id
    away_id = curr_match.away_team.id

    def _read_team_elo_after_match_raw(m: FSMatch, team_id: int) -> float:
        """
        Read the stored raw ELO for 'team_id' from match m.
        Supports both new names (*_elo_after_match_raw) and your older field name (*_elo_before_match_raw).
        """
        # Identify whether this team was home or away in match m
        is_home = m.home_team is not None and m.home_team.id == team_id

        # Prefer new field names
        if is_home and hasattr(m, "home_elo_after_match_raw"):
            v = getattr(m, "home_elo_after_match_raw")
            if v is not None:
                return float(v)
        if (not is_home) and hasattr(m, "away_elo_after_match_raw"):
            v = getattr(m, "away_elo_after_match_raw")
            if v is not None:
                return float(v)

        # Fallback to old field names (your previous code stored post-match ELO in *_before_match_raw)
        if is_home and hasattr(m, "home_elo_before_match_raw"):
            v = getattr(m, "home_elo_before_match_raw")
            if v is not None:
                return float(v)
        if (not is_home) and hasattr(m, "away_elo_before_match_raw"):
            v = getattr(m, "away_elo_before_match_raw")
            if v is not None:
                return float(v)

        return sett.INIT_ELO

    def _season_regress(prev_match: Optional[FSMatch], curr: FSMatch, elo: float) -> float:
        if prev_match is None:
            return elo
        prev_season = getattr(prev_match, "season", None)
        curr_season = getattr(curr, "season", None)
        if prev_season is None or curr_season is None:
            return elo
        if prev_season == curr_season:
            return elo

        mu = sett.INIT_ELO
        alpha = getattr(sett, "ELO_SEASON_REGRESSION", 1.0)  # 1.0 => no regression by default
        return float(mu + alpha * (elo - mu))

    def _prev_elo_pre_match(team_id: int) -> float:
        pm = get_previous_match(team_index_all, team_id, curr_match)
        if pm is None:
            return sett.INIT_ELO
        elo_after_prev = _read_team_elo_after_match_raw(pm, team_id)
        return _season_regress(pm, curr_match, elo_after_prev)

    rh = _prev_elo_pre_match(home_id)
    ra = _prev_elo_pre_match(away_id)

    # These are the PRE-MATCH feature values:
    home_pre_norm = normalize_elo(rh)
    away_pre_norm = normalize_elo(ra)

    # --- Decide update weight
    is_league = getattr(curr_match, "comp_name", None) in sett.COMPS_LEAGUE

    min_hist = getattr(sett, "MIN_ELO_MATCHES", 0)
    home_hist = len(team_index_league.matches_by_team.get(home_id, []))
    away_hist = len(team_index_league.matches_by_team.get(away_id, []))
    home_reliable = home_hist >= min_hist
    away_reliable = away_hist >= min_hist

    if is_league:
        update_weight = 1.0
    else:
        # Only use non-league matches if both teams are sufficiently "known"
        if home_reliable and away_reliable:
            update_weight = float(getattr(sett, "ELO_NON_LEAGUE_WEIGHT", 0.0))
        else:
            update_weight = 0.0

    # If we don't have a result, we cannot update.
    if curr_match.home_goals is None or curr_match.away_goals is None:
        update_weight = 0.0

    # --- Expected score
    c = 10.0
    d = 400.0
    exp_home = 1.0 / (1.0 + (c ** ((ra - rh) / d)))
    exp_away = 1.0 - exp_home

    # --- Actual outcome
    if update_weight > 0.0:
        hg = float(curr_match.home_goals)
        ag = float(curr_match.away_goals)
        if hg > ag:
            ah, aa = 1.0, 0.0
        elif hg < ag:
            ah, aa = 0.0, 1.0
        else:
            ah, aa = 0.5, 0.5

        k = float(sett.ELO_K) * update_weight
        rh_new = rh + k * (ah - exp_home)
        ra_new = ra + k * (aa - exp_away)
    else:
        rh_new, ra_new = rh, ra

    # --- Store post-match ELO for propagation
    # (Even if you haven't declared these attributes yet, Python will set them.)
    curr_match.home_elo_after_match_raw = rh_new
    curr_match.away_elo_after_match_raw = ra_new

    return home_pre_norm, away_pre_norm


def debug_print_match_and_features(match):
    f = match.features_before_match
    if f is None:
        print("No features computed.")
        return

    # --- MATCH STATISTICS (what you used to print)
    print("\n\tMATCH_STATISTICS:")
    print(f"{match.datetime} h={match.hour_utc}: {match.comp_name}, {match.season}, round_id={match.round_id}")

    hsot = match.stats.get("home_shots_on_target", -1)
    asot = match.stats.get("away_shots_on_target", -1)
    hts = match.stats.get("home_total_shots", -1)
    ats = match.stats.get("away_total_shots", -1)
    hc = match.stats.get("home_corners", -1)
    ac = match.stats.get("away_corners", -1)
    hp = match.stats.get("home_possession", -1)
    ap = match.stats.get("away_possession", -1)
    hxg = match.stats.get("home_xg", -1)
    axg = match.stats.get("away_xg", -1)
    hpxg = match.stats.get("home_prematch_xg", -1)
    apxg = match.stats.get("away_prematch_xg", -1)

    print(
        f"{match.home_team.name} {match.home_goals} "
        f"(sot={hsot}, shots={hts}, corners={hc}, poss={hp}, xg={hxg}, prem_xg={hpxg})  -  "
        f"{match.away_team.name} {match.away_goals} "
        f"(sot={asot}, shots={ats}, corners={ac}, poss={ap}, xg={axg}, prem_xg={apxg})"
    )

    # --- FEATURES BEFORE MATCH (with denorm like you used to)
    print("\n\tFEATURES_BEFORE_MATCH:")

    print(
        f"ELO home/away={f.home_elo:.3f}/{f.away_elo:.3f} "
        f"(raw stored on match={match.home_elo_after_match_raw:.1f}/{match.away_elo_after_match_raw:.1f})"
    )

    def denorm(x, coeff):  # assumes your normalize is x = raw/coeff
        return x * coeff

    print(
        f"Avg xG last5/20 home={f.home_avg_xg_last_5:.3f}/{f.home_avg_xg_last_20:.3f} "
        f"(denorm={denorm(f.home_avg_xg_last_5, sett.TEAM_XG_NORM_COEFFICIENT):.3f}/"
        f"{denorm(f.home_avg_xg_last_20, sett.TEAM_XG_NORM_COEFFICIENT):.3f})"
    )
    print(
        f"Avg xG last5/20 away={f.away_avg_xg_last_5:.3f}/{f.away_avg_xg_last_20:.3f} "
        f"(denorm={denorm(f.away_avg_xg_last_5, sett.TEAM_XG_NORM_COEFFICIENT):.3f}/"
        f"{denorm(f.away_avg_xg_last_20, sett.TEAM_XG_NORM_COEFFICIENT):.3f})"
    )

    print(
        f"Match load last10/25 days home={f.home_match_load_per_day_last_10_days:.3f}/"
        f"{f.home_match_load_per_day_last_25_days:.3f} "
        f"(denorm={(f.home_match_load_per_day_last_10_days * sett.MATCH_LOAD_NORM_COEFFICIENT):.3f}/"
        f"{(f.home_match_load_per_day_last_25_days * sett.MATCH_LOAD_NORM_COEFFICIENT):.3f})"
    )
    print(
        f"Match load last10/25 days away={f.away_match_load_per_day_last_10_days:.3f}/"
        f"{f.away_match_load_per_day_last_25_days:.3f} "
        f"(denorm={(f.away_match_load_per_day_last_10_days * sett.MATCH_LOAD_NORM_COEFFICIENT):.3f}/"
        f"{(f.away_match_load_per_day_last_25_days * sett.MATCH_LOAD_NORM_COEFFICIENT):.3f})"
    )

    print(f"Table position home/away={f.home_curr_position:.3f}/{f.away_curr_position:.3f}")
