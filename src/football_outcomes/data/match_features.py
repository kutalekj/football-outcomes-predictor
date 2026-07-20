from __future__ import annotations

from typing import TYPE_CHECKING

from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import (
    Global,
)
from football_outcomes.data.team_strength_matrix import (
    calculate_team_strength,
)
from football_outcomes.utils import fs_feature_utils as fu

if TYPE_CHECKING:
    from football_outcomes.data.fs_models import (
        FSMatch,
        FSMatchFeatures,
    )
    from football_outcomes.utils.fs_feature_utils import (
        TeamMatchIndex,
    )


def calculate_match_features(
    match: FSMatch,
    team_index_league: TeamMatchIndex,
    team_index_all: TeamMatchIndex,
) -> FSMatchFeatures:
    """Construct all pre-match features for one match."""

    from football_outcomes.data.fs_models import (
        FSMatchFeatures,
    )

    # Paste the existing implementation here.
    if match.home_team is None or match.away_team is None:
        raise ValueError("Match missing teams.")

    comp_season_id = match.comp_season_id
    if comp_season_id is None:
        raise ValueError(f"Match {match.id} missing comp_season_id.")

    # comp_id must be integer category index (stable)
    comp_name = match.comp_name or ""
    try:
        comp_id = sett.COMPS_LEAGUE.index(comp_name)
    except ValueError:
        # non-league comps can still exist in globals; we just keep -1
        comp_id = -1

    hour = int(match.hour_utc or 0)
    month = int(match.month or 1)
    hs, hc, ms, mc = fu.hour_month_cyclic(hour, month)

    mf = FSMatchFeatures(
        comp_id=comp_id,
        season=fu.normalize_season(match.season),
        home_team_id=match.home_team.id,
        away_team_id=match.away_team.id,
        hours_sin=hs,
        hours_cos=hc,
        month_sin=ms,
        month_cos=mc,
    )

    # ---- ELO (pre-match, computed from previous matches only, then stored on the match for next matches)
    mf.home_elo, mf.away_elo = fu.calculate_elo_for_match(
        team_index_league=team_index_league,
        curr_match=match,
    )

    # ---- Match position in season (requires populated first/last dates)
    g = Global.get_instance()
    cs = g.all_comp_seasons.get(comp_season_id)
    if cs is not None and cs.first_match_date is not None and cs.last_match_date is not None:
        total_seconds = (cs.last_match_date - cs.first_match_date).total_seconds()
        curr_seconds = (match.datetime - cs.first_match_date).total_seconds()
        # hour-level tie-break
        curr_seconds += float(hour) * 3600.0
        mf.match_position_in_season = fu.clip01(curr_seconds / total_seconds) if total_seconds > 0 else sett.ALMOST_ZERO
    else:
        mf.match_position_in_season = sett.ALMOST_ZERO

    # ---- xG averages
    mf.home_avg_xg_last_5 = fu.avg_stat_last_n(
        team_index_league, match.home_team.id, match, 5, "home_xg", "away_xg", fu.normalize_team_xg
    )
    mf.home_avg_xg_last_20 = fu.avg_stat_last_n(
        team_index_league, match.home_team.id, match, 20, "home_xg", "away_xg", fu.normalize_team_xg
    )
    mf.away_avg_xg_last_5 = fu.avg_stat_last_n(
        team_index_league, match.away_team.id, match, 5, "home_xg", "away_xg", fu.normalize_team_xg
    )
    mf.away_avg_xg_last_20 = fu.avg_stat_last_n(
        team_index_league, match.away_team.id, match, 20, "home_xg", "away_xg", fu.normalize_team_xg
    )

    mf.home_avg_xg_total_last_5 = fu.avg_total_stat_last_n(
        team_index_league, match.home_team.id, match, 5, fu.total_xg, fu.normalize_total_xg
    )
    mf.home_avg_xg_total_last_20 = fu.avg_total_stat_last_n(
        team_index_league, match.home_team.id, match, 20, fu.total_xg, fu.normalize_total_xg
    )
    mf.away_avg_xg_total_last_5 = fu.avg_total_stat_last_n(
        team_index_league, match.away_team.id, match, 5, fu.total_xg, fu.normalize_total_xg
    )
    mf.away_avg_xg_total_last_20 = fu.avg_total_stat_last_n(
        team_index_league, match.away_team.id, match, 20, fu.total_xg, fu.normalize_total_xg
    )

    # ---- pre-match xG averages
    mf.home_avg_pre_match_xg_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        5,
        "home_prematch_xg",
        "away_prematch_xg",
        fu.normalize_team_pre_match_xg,
    )
    mf.home_avg_pre_match_xg_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        20,
        "home_prematch_xg",
        "away_prematch_xg",
        fu.normalize_team_pre_match_xg,
    )
    mf.away_avg_pre_match_xg_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        5,
        "home_prematch_xg",
        "away_prematch_xg",
        fu.normalize_team_pre_match_xg,
    )
    mf.away_avg_pre_match_xg_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        20,
        "home_prematch_xg",
        "away_prematch_xg",
        fu.normalize_team_pre_match_xg,
    )

    mf.home_avg_pre_match_xg_total_last_5 = fu.avg_total_stat_last_n(
        team_index_league, match.home_team.id, match, 5, fu.total_pre_match_xg, fu.normalize_total_pre_match_xg
    )
    mf.home_avg_pre_match_xg_total_last_20 = fu.avg_total_stat_last_n(
        team_index_league, match.home_team.id, match, 20, fu.total_pre_match_xg, fu.normalize_total_pre_match_xg
    )
    mf.away_avg_pre_match_xg_total_last_5 = fu.avg_total_stat_last_n(
        team_index_league, match.away_team.id, match, 5, fu.total_pre_match_xg, fu.normalize_total_pre_match_xg
    )
    mf.away_avg_pre_match_xg_total_last_20 = fu.avg_total_stat_last_n(
        team_index_league, match.away_team.id, match, 20, fu.total_pre_match_xg, fu.normalize_total_pre_match_xg
    )

    # ---- match load
    mf.home_match_load_per_day_last_10_days = fu.match_load_per_day_last_n_days(
        team_index_all, match.home_team.id, match, 10
    )
    mf.home_match_load_per_day_last_25_days = fu.match_load_per_day_last_n_days(
        team_index_all, match.home_team.id, match, 25
    )
    mf.away_match_load_per_day_last_10_days = fu.match_load_per_day_last_n_days(
        team_index_all, match.away_team.id, match, 10
    )
    mf.away_match_load_per_day_last_25_days = fu.match_load_per_day_last_n_days(
        team_index_all, match.away_team.id, match, 25
    )

    # ---- points/goals
    mf.home_avg_points_last_5 = fu.avg_points_last_n(team_index_league, match.home_team.id, match, 5)
    mf.home_avg_points_last_20 = fu.avg_points_last_n(team_index_league, match.home_team.id, match, 20)
    mf.away_avg_points_last_5 = fu.avg_points_last_n(team_index_league, match.away_team.id, match, 5)
    mf.away_avg_points_last_20 = fu.avg_points_last_n(team_index_league, match.away_team.id, match, 20)

    mf.home_avg_goals_last_5 = fu.avg_goals_last_n(team_index_league, match.home_team.id, match, 5)
    mf.home_avg_goals_last_20 = fu.avg_goals_last_n(team_index_league, match.home_team.id, match, 20)
    mf.away_avg_goals_last_5 = fu.avg_goals_last_n(team_index_league, match.away_team.id, match, 5)
    mf.away_avg_goals_last_20 = fu.avg_goals_last_n(team_index_league, match.away_team.id, match, 20)

    # ---- shots/corners/etc.
    mf.home_avg_shots_on_target_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        5,
        "home_shots_on_target",
        "away_shots_on_target",
        fu.normalize_shots_on_g,
    )
    mf.home_avg_shots_on_target_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        20,
        "home_shots_on_target",
        "away_shots_on_target",
        fu.normalize_shots_on_g,
    )
    mf.away_avg_shots_on_target_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        5,
        "home_shots_on_target",
        "away_shots_on_target",
        fu.normalize_shots_on_g,
    )
    mf.away_avg_shots_on_target_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        20,
        "home_shots_on_target",
        "away_shots_on_target",
        fu.normalize_shots_on_g,
    )

    mf.home_avg_shots_off_target_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        5,
        "home_shots_off_target",
        "away_shots_off_target",
        fu.normalize_shots_off_g,
    )
    mf.home_avg_shots_off_target_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        20,
        "home_shots_off_target",
        "away_shots_off_target",
        fu.normalize_shots_off_g,
    )
    mf.away_avg_shots_off_target_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        5,
        "home_shots_off_target",
        "away_shots_off_target",
        fu.normalize_shots_off_g,
    )
    mf.away_avg_shots_off_target_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        20,
        "home_shots_off_target",
        "away_shots_off_target",
        fu.normalize_shots_off_g,
    )

    mf.home_avg_total_shots_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        5,
        "home_total_shots",
        "away_total_shots",
        fu.normalize_total_shots,
    )
    mf.home_avg_total_shots_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        20,
        "home_total_shots",
        "away_total_shots",
        fu.normalize_total_shots,
    )
    mf.away_avg_total_shots_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        5,
        "home_total_shots",
        "away_total_shots",
        fu.normalize_total_shots,
    )
    mf.away_avg_total_shots_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        20,
        "home_total_shots",
        "away_total_shots",
        fu.normalize_total_shots,
    )

    mf.home_avg_corner_kicks_last_5 = fu.avg_stat_last_n(
        team_index_league, match.home_team.id, match, 5, "home_corners", "away_corners", fu.normalize_corners
    )
    mf.home_avg_corner_kicks_last_20 = fu.avg_stat_last_n(
        team_index_league, match.home_team.id, match, 20, "home_corners", "away_corners", fu.normalize_corners
    )
    mf.away_avg_corner_kicks_last_5 = fu.avg_stat_last_n(
        team_index_league, match.away_team.id, match, 5, "home_corners", "away_corners", fu.normalize_corners
    )
    mf.away_avg_corner_kicks_last_20 = fu.avg_stat_last_n(
        team_index_league, match.away_team.id, match, 20, "home_corners", "away_corners", fu.normalize_corners
    )

    # possession/fouls/attacks/dang attacks are already in [0..100] or similar.
    # For now, we normalize by simple /100 for possession and /50 for fouls/attacks-ish later if needed.
    # To avoid inventing wrong caps, we keep them in [0..1] by clipping after /100 or /200 etc would be risky.
    # So: scale possession by /100; keep the rest min-max with conservative caps later (you can tune).
    def poss_norm(x: float) -> float:
        return fu.clip01(x / 100.0)

    mf.home_avg_ball_possession_last_5 = fu.avg_stat_last_n(
        team_index_league, match.home_team.id, match, 5, "home_possession", "away_possession", poss_norm
    )
    mf.home_avg_ball_possession_last_20 = fu.avg_stat_last_n(
        team_index_league, match.home_team.id, match, 20, "home_possession", "away_possession", poss_norm
    )
    mf.away_avg_ball_possession_last_5 = fu.avg_stat_last_n(
        team_index_league, match.away_team.id, match, 5, "home_possession", "away_possession", poss_norm
    )
    mf.away_avg_ball_possession_last_20 = fu.avg_stat_last_n(
        team_index_league, match.away_team.id, match, 20, "home_possession", "away_possession", poss_norm
    )

    # fouls/attacks/dangerous attacks left as scaled by conservative caps (TODO: tune later)
    mf.home_avg_fouls_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        5,
        "home_fouls",
        "away_fouls",
        fu.normalize_fouls,
    )
    mf.home_avg_fouls_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        20,
        "home_fouls",
        "away_fouls",
        fu.normalize_fouls,
    )
    mf.away_avg_fouls_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        5,
        "home_fouls",
        "away_fouls",
        fu.normalize_fouls,
    )
    mf.away_avg_fouls_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        20,
        "home_fouls",
        "away_fouls",
        fu.normalize_fouls,
    )

    mf.home_avg_attacks_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        5,
        "home_attacks",
        "away_attacks",
        fu.normalize_attacks,
    )
    mf.home_avg_attacks_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        20,
        "home_attacks",
        "away_attacks",
        fu.normalize_attacks,
    )
    mf.away_avg_attacks_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        5,
        "home_attacks",
        "away_attacks",
        fu.normalize_attacks,
    )
    mf.away_avg_attacks_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        20,
        "home_attacks",
        "away_attacks",
        fu.normalize_attacks,
    )

    mf.home_avg_dang_attacks_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        5,
        "home_dangerous_attacks",
        "away_dangerous_attacks",
        fu.normalize_dang_attacks,
    )
    mf.home_avg_dang_attacks_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.home_team.id,
        match,
        20,
        "home_dangerous_attacks",
        "away_dangerous_attacks",
        fu.normalize_dang_attacks,
    )
    mf.away_avg_dang_attacks_last_5 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        5,
        "home_dangerous_attacks",
        "away_dangerous_attacks",
        fu.normalize_dang_attacks,
    )
    mf.away_avg_dang_attacks_last_20 = fu.avg_stat_last_n(
        team_index_league,
        match.away_team.id,
        match,
        20,
        "home_dangerous_attacks",
        "away_dangerous_attacks",
        fu.normalize_dang_attacks,
    )

    # ---- League table positions (assumes the earlier table init exists on cs)
    if cs is not None and hasattr(cs, "get_team_position_before_match"):
        mf.home_curr_position = cs.get_team_position_before_match(match.home_team.id, match)
        mf.away_curr_position = cs.get_team_position_before_match(match.away_team.id, match)
    else:
        mf.home_curr_position = sett.ALMOST_ZERO
        mf.away_curr_position = sett.ALMOST_ZERO

    # ---- Home/away-only scored/conceded features (from your old class)
    # These should be computed using only matches where team was home/away.
    mf.home_avg_goals_scored_home_last_5, mf.home_avg_goals_conceded_home_last_5 = (
        fu.avg_goals_scored_conceded_role_last_n(team_index_league, match.home_team.id, match, 5, "home")
    )
    mf.home_avg_goals_scored_home_last_20, mf.home_avg_goals_conceded_home_last_20 = (
        fu.avg_goals_scored_conceded_role_last_n(team_index_league, match.home_team.id, match, 20, "home")
    )

    mf.away_avg_goals_scored_away_last_5, mf.away_avg_goals_conceded_away_last_5 = (
        fu.avg_goals_scored_conceded_role_last_n(team_index_league, match.away_team.id, match, 5, "away")
    )
    mf.away_avg_goals_scored_away_last_20, mf.away_avg_goals_conceded_away_last_20 = (
        fu.avg_goals_scored_conceded_role_last_n(team_index_league, match.away_team.id, match, 20, "away")
    )

    # ---- Team strength calculation
    mf.home_team_strength = calculate_team_strength(match, match.home_team.id)
    mf.away_team_strength = calculate_team_strength(match, match.away_team.id)

    return mf
