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
