from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Optional

from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import Global


def _safe_min_dt(matches) -> Optional[datetime]:
    dts = [m.datetime for m in matches if getattr(m, "datetime", None) is not None]
    return min(dts) if dts else None


def _safe_max_dt(matches) -> Optional[datetime]:
    dts = [m.datetime for m in matches if getattr(m, "datetime", None) is not None]
    return max(dts) if dts else None


def _season_key(dt: datetime, start_month: int) -> tuple[int, int, int, int, int]:
    """
    Key for comparing dates within a season that may cross calendar years.
    Example (start_month=8):
      Nov (11) -> 11
      May (5)  -> 17  (5 + 12)  => correctly treated as later than Nov.
    """
    m = dt.month
    m_adj = m if m >= start_month else m + 12
    return m_adj, dt.day, dt.hour, dt.minute, dt.second


def _project_end_date_for_season(
    season_start_year: int,
    start_month: int,
    template_end_dt: datetime,
) -> datetime:
    """
    Uses month/day/time from template_end_dt, but chooses year based on whether the end month
    falls before the season start month (=> next calendar year).
    """
    end_year = season_start_year + 1 if template_end_dt.month < start_month else season_start_year
    return template_end_dt.replace(year=end_year)


def populate_comp_season_first_last_dates() -> None:
    """
    Populates:
      - first_match_date: earliest match datetime in that season
      - last_match_date:
          * for seasons < LAST_SEASON: latest match datetime in that season
          * for LAST_SEASON: estimated using the latest end date across all previous seasons
            of the same competition (same (country, name)), but never earlier than what already
            present for the current season.
    Always overwrites existing attributes.
    """
    global_instance = Global.get_instance()
    comp_seasons = list(global_instance.all_comp_seasons.values())

    # 1) Per-season first/last + season start_month (from first match)
    per_first: dict[int, Optional[datetime]] = {}
    per_last: dict[int, Optional[datetime]] = {}
    per_start_month: dict[int, Optional[int]] = {}

    for cs in comp_seasons:
        first_dt = _safe_min_dt(cs.matches)
        last_dt = _safe_max_dt(cs.matches)

        per_first[cs.id] = first_dt
        per_last[cs.id] = last_dt
        per_start_month[cs.id] = first_dt.month if first_dt is not None else None

    # 2) Best previous-season "end template" per competition (country,name),
    #    using season-aware ordering (so May beats Nov for Aug-start seasons).
    best_prev_end_by_comp: dict[tuple[str, str], Optional[datetime]] = defaultdict(lambda: None)

    for cs in comp_seasons:
        if cs.season is None:
            continue
        if not (sett.FIRST_SEASON <= cs.season < sett.LAST_SEASON):
            continue

        last_dt = per_last.get(cs.id)
        start_month = per_start_month.get(cs.id)
        if last_dt is None or start_month is None:
            continue

        comp_key = (cs.country, cs.name)
        prev_best = best_prev_end_by_comp[comp_key]

        if prev_best is None:
            best_prev_end_by_comp[comp_key] = last_dt
        else:
            # Compare using THIS season's start_month
            if _season_key(last_dt, start_month) > _season_key(prev_best, start_month):
                best_prev_end_by_comp[comp_key] = last_dt

    # 3) Assign first/last (overwrite always)
    for cs in comp_seasons:
        cs.first_match_date = per_first.get(cs.id)

        own_last = per_last.get(cs.id)

        # Normal seasons: just use actual last match.
        if cs.season != sett.LAST_SEASON:
            cs.last_match_date = own_last
            continue

        comp_key = (cs.country, cs.name)
        prev_best_end = best_prev_end_by_comp.get(comp_key)

        # If no template, fall back.
        if prev_best_end is None:
            cs.last_match_date = own_last
            continue

        # Determine start_month for LAST_SEASON comparisons and projection.
        start_month = per_start_month.get(cs.id)
        if start_month is None:
            # fallback: pick smallest non-None start_month from previous seasons of this comp
            prev_start_months = [
                per_start_month.get(other.id)
                for other in comp_seasons
                if (other.country, other.name) == comp_key
                and other.season is not None
                and sett.FIRST_SEASON <= other.season < sett.LAST_SEASON
                and per_start_month.get(other.id) is not None
            ]
            start_month = min(prev_start_months) if prev_start_months else prev_best_end.month

        estimated_end = _project_end_date_for_season(
            season_start_year=sett.LAST_SEASON,
            start_month=start_month,
            template_end_dt=prev_best_end,
        )

        # Choose later in SEASON ORDER (not raw month/day)
        if own_last is None:
            cs.last_match_date = estimated_end
        else:
            cs.last_match_date = (
                estimated_end
                if _season_key(estimated_end, start_month) >= _season_key(own_last, start_month)
                else own_last
            )


def ensure_comp_season_dates(force: bool = False) -> None:
    global_instance = Global.get_instance()
    if (not force) and all(
        (cs.first_match_date is not None and cs.last_match_date is not None)
        for cs in global_instance.all_comp_seasons.values()
    ):
        return
    populate_comp_season_first_last_dates()


def initialize_league_tables(precompute_positions: bool = True, force_rebuild: bool = False) -> None:
    global_instance = Global.get_instance()

    league_seasons = [cs for cs in global_instance.all_comp_seasons.values() if cs.name in sett.COMPS_LEAGUE]

    print(f"[tables] Initializing league tables for {len(league_seasons)} competition seasons...")

    for cs in league_seasons:
        already_ready = getattr(cs, "_table_initialized", False) and bool(getattr(cs, "team_stats", {}))
        already_cached = bool(getattr(cs, "_pre_match_positions", {}))

        if (not force_rebuild) and already_ready and ((not precompute_positions) or already_cached):
            continue

        cs.init_league_table()
        if precompute_positions:  # False when don’t need positions yet/plan to compute them later
            cs.build_pre_match_positions_cache()

    print("[tables] Done.")


def link_matches_to_comp_seasons() -> None:
    """Ensure each FSMatch has comp_season_id / comp_name / country set.

    Works purely from already-loaded Global objects (snapshot), no API calls needed.
    """
    from football_outcomes.config.fs_globals import Global

    g = Global.get_instance()

    # Build match_id -> comp season
    match_to_cs = {}
    for cs in g.all_comp_seasons.values():
        for m in cs.matches:
            match_to_cs[m.id] = cs

    missing = 0
    for m in g.all_matches:
        cs = match_to_cs.get(m.id)
        if cs is None:
            missing += 1
            continue
        m.comp_season_id = cs.id
        m.comp_name = cs.name
        m.country = cs.country

    print(f"[link] Linked {len(g.all_matches) - missing}/{len(g.all_matches)} matches to comp seasons.")
    if missing:
        print(f"[link] Warning: {missing} matches not found in any comp season.")


def is_excluded_comp_season(comp_name: str | None, season: int | None) -> bool:
    if comp_name is None or season is None:
        return False
    return (comp_name, int(season)) in sett.EXCLUDED_COMP_SEASONS


def filter_clean_league_matches(matches):
    """
    Keep only league matches that are not in excluded competition seasons.
    """
    return [
        m
        for m in matches
        if getattr(m, "comp_name", None) in sett.COMPS_LEAGUE
        and not is_excluded_comp_season(getattr(m, "comp_name", None), getattr(m, "season", None))
    ]


def get_allowed_match_stat_keys(stat_keys):
    """
    Drop stats that should remain in raw data but be ignored in cleaned analysis/modeling.
    """
    return [k for k in stat_keys if k not in sett.IGNORED_MATCH_STATS]


def _match_sort_key_for_regular_season_flag(m):
    dt = getattr(m, "datetime", None)
    hr = getattr(m, "hour_utc", None)
    hr = hr if isinstance(hr, int) else -1
    return dt, hr, m.id


def annotate_regular_season_matches(debug_non_regular: bool = True) -> None:
    """
    Mark league matches as regular season / non-regular season using round_id tracking.

    Assumption:
      - each competition season starts with the regular-season phase
      - round_id is constant inside that round type
      - when round_id first changes away from the initial one, the season has moved
        into a non-regular phase (playoffs / split group / final series, etc.)

    All matches are first reset to regular_season=False.
    Then only league matches in the configured season interval are annotated.
    """
    g = Global.get_instance()

    # Reset everything first
    for m in g.all_matches:
        m.regular_season = False

    league_matches = [
        m
        for m in g.all_matches
        if getattr(m, "comp_name", None) in sett.COMPS_LEAGUE
        and getattr(m, "season", None) is not None
        and sett.FIRST_SEASON <= m.season < sett.LAST_SEASON
    ]

    by_comp_season: dict[int, list] = defaultdict(list)
    for m in league_matches:
        comp_season_id = getattr(m, "comp_season_id", None)
        if comp_season_id is None:
            continue
        by_comp_season[comp_season_id].append(m)

    total_regular = 0
    total_non_regular = 0

    for comp_season_id, matches in sorted(by_comp_season.items()):
        matches.sort(key=_match_sort_key_for_regular_season_flag)

        first_match = matches[0]
        comp_name = getattr(first_match, "comp_name", "<unknown>")
        season = getattr(first_match, "season", None)

        initial_round_id = getattr(first_match, "round_id", None)
        phase_switched = False
        switch_reason = None

        print(
            f"[regular_season] Processing [{comp_name} {season}] "
            f"matches={len(matches)} initial_round_id={initial_round_id}"
        )

        for m in matches:
            rid = getattr(m, "round_id", None)

            if not phase_switched:
                # Switch signal: first round_id change away from initial regular-season round_id
                if initial_round_id is not None and rid is not None and rid != initial_round_id:
                    phase_switched = True
                    switch_reason = f"round_id changed from initial {initial_round_id} to {rid}"

            m.regular_season = not phase_switched

            if m.regular_season:
                total_regular += 1
            else:
                total_non_regular += 1
                if debug_non_regular:
                    home_name = m.home_team.name if m.home_team is not None else "?"
                    away_name = m.away_team.name if m.away_team is not None else "?"
                    print(
                        "[regular_season][NON-REGULAR] "
                        f"{m.comp_name} {m.season} "
                        f"date={m.datetime.date() if m.datetime else None} "
                        f"hour_utc={m.hour_utc} "
                        f"game_week={m.game_week} "
                        f"round_id={m.round_id} "
                        f"match_id={m.id} "
                        f"{home_name} vs {away_name} "
                        f"reason={switch_reason}"
                    )

        reg_cnt = sum(1 for m in matches if m.regular_season)
        nonreg_cnt = len(matches) - reg_cnt
        print(f"[regular_season] Done [{comp_name} {season}] " f"regular={reg_cnt} non_regular={nonreg_cnt}")

    print(
        "[regular_season] Summary: "
        f"regular={total_regular}, non_regular={total_non_regular}, "
        f"total={total_regular + total_non_regular}"
    )
