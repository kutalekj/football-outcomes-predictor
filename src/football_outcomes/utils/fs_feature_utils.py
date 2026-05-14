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


def normalize_season(season: int) -> float:
    if season is None:
        return 0.0
    return clip01((float(season) - float(sett.FIRST_SEASON)) / (2024.0 - float(sett.FIRST_SEASON)))


def normalize_points(points_per_game: float) -> float:
    # points_per_game is in [0, 3]
    return clip01(points_per_game / 3.0)


def normalize_goals(goals_per_game: float) -> float:
    return min_max_scaling_with_clipping(goals_per_game, sett.GOALS_NORM_COEFFICIENT)


def normalize_shots_on_g(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.SHOTS_ON_G_NORM_COEFFICIENT)


def normalize_shots_off_g(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.SHOTS_OFF_G_NORM_COEFFICIENT)


def normalize_total_shots(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.TOTAL_SHOTS_NORM_COEFFICIENT)


def normalize_corners(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.CORNER_KICKS_NORM_COEFFICIENT)


def normalize_fouls(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.FOULS_NORM_COEFFICIENT)


def normalize_attacks(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.ATTACKS_NORM_COEFFICIENT)


def normalize_dang_attacks(v: float) -> float:
    return min_max_scaling_with_clipping(v, sett.DANG_ATTACKS_NORM_COEFFICIENT)


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


def normalize_elo(
    raw_elo: float, min_elo: float = sett.ELO_MIN_NORM_COEFFICIENT, max_elo: float = sett.ELO_MAX_NORM_COEFFICIENT
) -> float:
    return clip01((raw_elo - min_elo) / (max_elo - min_elo))


def _to01_from_pm1(x: float) -> float:
    return clip01((x + 1.0) / 2.0)


def hour_month_cyclic(hour_utc: int, month: int) -> Tuple[float, float, float, float]:
    # hour in [0..23], month in [1..12]
    hour_angle = 2.0 * math.pi * (float(hour_utc) / 24.0)
    month_angle = 2.0 * math.pi * ((float(month) - 1.0) / 12.0)

    return (
        _to01_from_pm1(math.sin(hour_angle)),
        _to01_from_pm1(math.cos(hour_angle)),
        _to01_from_pm1(math.sin(month_angle)),
        _to01_from_pm1(math.cos(month_angle)),
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


def calculate_elo_for_match(team_index_league: TeamMatchIndex, curr_match: FSMatch) -> Tuple[float, float]:
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

    # TODO: Remove this debug print
    if curr_match.home_team.name == "KRC Genk" or curr_match.away_team.name == "KRC Genk":
        pass

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
        pm = get_previous_match(team_index_league, team_id, curr_match)
        if pm is None:
            return sett.INIT_ELO
        elo_after_prev = _read_team_elo_after_match_raw(pm, team_id)
        return _season_regress(pm, curr_match, elo_after_prev)

    rh = _prev_elo_pre_match(home_id)
    ra = _prev_elo_pre_match(away_id)

    # These are the PRE-MATCH feature values:
    home_pre_norm = normalize_elo(rh)
    away_pre_norm = normalize_elo(ra)

    # --- Expected score
    c = 10.0
    d = 400.0
    exp_home = 1.0 / (1.0 + (c ** ((ra - rh) / d)))
    exp_away = 1.0 - exp_home

    # --- Actual outcome
    hg = float(curr_match.home_goals)
    ag = float(curr_match.away_goals)
    if hg > ag:
        ah, aa = 1.0, 0.0
    elif hg < ag:
        ah, aa = 0.0, 1.0
    else:
        ah, aa = 0.5, 0.5

    k = float(sett.ELO_K)
    rh_new = rh + k * (ah - exp_home)
    ra_new = ra + k * (aa - exp_away)

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

    def denorm(x, coeff):
        # assumes normalize: x_norm = raw / coeff
        if x is None:
            return None
        return x * coeff

    def fmt(x, digits=3):
        if x is None:
            return "None"
        try:
            return f"{x:.{digits}f}"
        except Exception:
            return str(x)

    def fmt_denorm(x, coeff, digits=3):
        d = denorm(x, coeff)
        if d is None:
            return "None"
        return f"{d:.{digits}f}"

    # FEATURES BEFORE MATCH (Genk-side only)
    print(f"\n\tFEATURES_BEFORE_MATCH: (pos. in season: {f.match_position_in_season:.2f})")
    TARGET_TEAM = "KRC Genk"

    home_name = getattr(match.home_team, "name", "") or ""
    away_name = getattr(match.away_team, "name", "") or ""

    # simple, robust match (substring, case-insensitive)
    genk_is_home = TARGET_TEAM.lower() in home_name.lower()
    genk_is_away = TARGET_TEAM.lower() in away_name.lower()

    if not (genk_is_home or genk_is_away):
        print(
            f"\t[WARN] Target team '{TARGET_TEAM}' not found in match: home='{home_name}', away='{away_name}'. "
            f"Defaulting to home-side prints."
        )
        genk_is_home = True

    side = "home" if genk_is_home else "away"
    side_team_name = home_name if genk_is_home else away_name

    def pick(home_val, away_val):
        return home_val if genk_is_home else away_val

    def pick_attr(home_attr: str, away_attr: str):
        return getattr(f, home_attr) if genk_is_home else getattr(f, away_attr)

    def print_pair(
        label: str,
        home_attr_5: str,
        home_attr_20: str,
        away_attr_5: str,
        away_attr_20: str,
        coeff=None,
        indent_tabs: int = 0,
    ):
        v5 = pick_attr(home_attr_5, away_attr_5)
        v20 = pick_attr(home_attr_20, away_attr_20)
        prefix = "\t" * indent_tabs
        if coeff is None:
            print(f"{label}  {prefix}{side}({side_team_name})={fmt(v5)}/{fmt(v20)}")
        else:
            print(
                f"{label}  {prefix}{side}({side_team_name})={fmt(v5)}/{fmt(v20)} "
                f"(denorm={fmt_denorm(v5, coeff)}/{fmt_denorm(v20, coeff)})"
            )

    # ELO (single values)
    elo_val = pick(f.home_elo, f.away_elo)
    elo_raw = pick(match.home_elo_after_match_raw, match.away_elo_after_match_raw)
    print(f"ELO {side}({side_team_name})={fmt(elo_val)} " f"(raw after match={elo_raw:.1f})")

    # --- xG features (TEAM & TOTAL, prematch & in-match)
    print_pair(
        "Avg xG (TEAM) last 5/20",
        "home_avg_xg_last_5",
        "home_avg_xg_last_20",
        "away_avg_xg_last_5",
        "away_avg_xg_last_20",
        coeff=sett.TEAM_XG_NORM_COEFFICIENT,
        indent_tabs=9,
    )

    print_pair(
        "Avg xG (TOTAL) last 5/20",
        "home_avg_xg_total_last_5",
        "home_avg_xg_total_last_20",
        "away_avg_xg_total_last_5",
        "away_avg_xg_total_last_20",
        coeff=sett.TOTAL_XG_NORM_COEFFICIENT,
        indent_tabs=9,
    )

    print_pair(
        "Avg pre-match xG (TEAM) last 5/20",
        "home_avg_pre_match_xg_last_5",
        "home_avg_pre_match_xg_last_20",
        "away_avg_pre_match_xg_last_5",
        "away_avg_pre_match_xg_last_20",
        coeff=sett.TEAM_PRE_MATCH_XG_NORM_COEFFICIENT,
        indent_tabs=5,
    )

    print_pair(
        "Avg pre-match xG (TOTAL) last 5/20",
        "home_avg_pre_match_xg_total_last_5",
        "home_avg_pre_match_xg_total_last_20",
        "away_avg_pre_match_xg_total_last_5",
        "away_avg_pre_match_xg_total_last_20",
        coeff=sett.TOTAL_PRE_MATCH_XG_NORM_COEFFICIENT,
        indent_tabs=5,
    )

    # --- match load (10/25 days) (single line, two values)
    ml10 = pick(f.home_match_load_per_day_last_10_days, f.away_match_load_per_day_last_10_days)
    ml25 = pick(f.home_match_load_per_day_last_25_days, f.away_match_load_per_day_last_25_days)
    print(
        f"Avg match load per day last 10/25 days  "
        f"{side}({side_team_name})={fmt(ml10)}/{fmt(ml25)} "
        f"(denorm={fmt_denorm(ml10, sett.MATCH_LOAD_NORM_COEFFICIENT)}/"
        f"{fmt_denorm(ml25, sett.MATCH_LOAD_NORM_COEFFICIENT)})"
    )

    # --- points
    print_pair(
        "Avg points last 5/20",
        "home_avg_points_last_5",
        "home_avg_points_last_20",
        "away_avg_points_last_5",
        "away_avg_points_last_20",
        coeff=None,
        indent_tabs=10,
    )

    # --- goals
    print_pair(
        "Avg goals last 5/20",
        "home_avg_goals_last_5",
        "home_avg_goals_last_20",
        "away_avg_goals_last_5",
        "away_avg_goals_last_20",
        coeff=sett.GOALS_NORM_COEFFICIENT,
        indent_tabs=9,
    )

    # --- shots on target
    print_pair(
        "Avg SOT last 5/20",
        "home_avg_shots_on_target_last_5",
        "home_avg_shots_on_target_last_20",
        "away_avg_shots_on_target_last_5",
        "away_avg_shots_on_target_last_20",
        coeff=sett.SHOTS_ON_G_NORM_COEFFICIENT,
        indent_tabs=12,
    )

    # --- shots off target
    print_pair(
        "Avg shots OFF target last 5/20",
        "home_avg_shots_off_target_last_5",
        "home_avg_shots_off_target_last_20",
        "away_avg_shots_off_target_last_5",
        "away_avg_shots_off_target_last_20",
        coeff=sett.SHOTS_OFF_G_NORM_COEFFICIENT,
        indent_tabs=4,
    )

    # --- total shots
    print_pair(
        "Avg total shots last 5/20",
        "home_avg_total_shots_last_5",
        "home_avg_total_shots_last_20",
        "away_avg_total_shots_last_5",
        "away_avg_total_shots_last_20",
        coeff=sett.TOTAL_SHOTS_NORM_COEFFICIENT,
        indent_tabs=8,
    )

    # --- corners
    print_pair(
        "Avg corner kicks last 5/20",
        "home_avg_corner_kicks_last_5",
        "home_avg_corner_kicks_last_20",
        "away_avg_corner_kicks_last_5",
        "away_avg_corner_kicks_last_20",
        coeff=sett.CORNER_KICKS_NORM_COEFFICIENT,
        indent_tabs=6,
    )

    # --- possession (no coeff provided)
    print_pair(
        "Avg possession last 5/20",
        "home_avg_ball_possession_last_5",
        "home_avg_ball_possession_last_20",
        "away_avg_ball_possession_last_5",
        "away_avg_ball_possession_last_20",
        coeff=None,
        indent_tabs=8,
    )

    # --- fouls
    print_pair(
        "Avg fouls last 5/20",
        "home_avg_fouls_last_5",
        "home_avg_fouls_last_20",
        "away_avg_fouls_last_5",
        "away_avg_fouls_last_20",
        coeff=sett.FOULS_NORM_COEFFICIENT,
        indent_tabs=9,
    )

    # --- attacks
    print_pair(
        "Avg attacks last 5/20",
        "home_avg_attacks_last_5",
        "home_avg_attacks_last_20",
        "away_avg_attacks_last_5",
        "away_avg_attacks_last_20",
        coeff=sett.ATTACKS_NORM_COEFFICIENT,
        indent_tabs=10,
    )

    # --- dangerous attacks
    print_pair(
        "Avg dangerous attacks last 5/20",
        "home_avg_dang_attacks_last_5",
        "home_avg_dang_attacks_last_20",
        "away_avg_dang_attacks_last_5",
        "away_avg_dang_attacks_last_20",
        coeff=sett.DANG_ATTACKS_NORM_COEFFICIENT,
        indent_tabs=4,
    )

    # --- home/away split goals scored/conceded (still depends on home/away context!)
    # If Genk is home: use home_scored_home / home_conceded_home
    # If Genk is away: use away_scored_away / away_conceded_away
    scored5 = pick(f.home_avg_goals_scored_home_last_5, f.away_avg_goals_scored_away_last_5)
    scored20 = pick(f.home_avg_goals_scored_home_last_20, f.away_avg_goals_scored_away_last_20)
    conc5 = pick(f.home_avg_goals_conceded_home_last_5, f.away_avg_goals_conceded_away_last_5)
    conc20 = pick(f.home_avg_goals_conceded_home_last_20, f.away_avg_goals_conceded_away_last_20)

    print(
        f"Avg goals scored ({side} split) last 5/20  "
        f"{side}({side_team_name})={fmt(scored5)}/{fmt(scored20)} "
        f"(denorm={fmt_denorm(scored5, sett.GOALS_NORM_COEFFICIENT)}/"
        f"{fmt_denorm(scored20, sett.GOALS_NORM_COEFFICIENT)})"
    )
    print(
        f"Avg goals conceded ({side} split) last 5/20  "
        f"{side}({side_team_name})={fmt(conc5)}/{fmt(conc20)} "
        f"(denorm={fmt_denorm(conc5, sett.GOALS_NORM_COEFFICIENT)}/"
        f"{fmt_denorm(conc20, sett.GOALS_NORM_COEFFICIENT)})"
    )

    # --- table position
    pos = pick(f.home_curr_position, f.away_curr_position)
    print(f"Table position {side}({side_team_name})={fmt(pos)}")

    # MATCH STATISTICS
    print("\n\tMATCH_STATISTICS:")
    print(f"{match.datetime} h={match.hour_utc}: {match.comp_name}, {match.season}, round_id={match.round_id}")

    s = match.stats or {}
    hsot = s.get("home_shots_on_target", -1)
    asot = s.get("away_shots_on_target", -1)
    hsoff = s.get("home_shots_off_target", -1)
    asoff = s.get("away_shots_off_target", -1)
    hts = s.get("home_total_shots", -1)
    ats = s.get("away_total_shots", -1)
    hc = s.get("home_corners", -1)
    ac = s.get("away_corners", -1)
    hp = s.get("home_possession", -1)
    ap = s.get("away_possession", -1)
    hf = s.get("home_fouls", -1)
    af = s.get("away_fouls", -1)
    hat = s.get("home_attacks", -1)
    aat = s.get("away_attacks", -1)
    hdat = s.get("home_dangerous_attacks", -1)
    adat = s.get("away_dangerous_attacks", -1)
    hxg = s.get("home_xg", -1)
    axg = s.get("away_xg", -1)
    hpxg = s.get("home_prematch_xg", -1)
    apxg = s.get("away_prematch_xg", -1)

    print(
        f"{match.home_team.name} {match.home_goals} "
        f"(sot={hsot}, soff={hsoff}, shots={hts}, corners={hc}, poss={hp}, fouls={hf}, att={hat}, "
        f"dangatt={hdat}, xg={hxg}, prem_xg={hpxg})  -  "
        f"{match.away_team.name} {match.away_goals} "
        f"(sot={asot}, soff={asoff}, shots={ats}, corners={ac}, poss={ap}, fouls={af}, att={aat}, "
        f"dangatt={adat}, xg={axg}, prem_xg={apxg})"
    )
