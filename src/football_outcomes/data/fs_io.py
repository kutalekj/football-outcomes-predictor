"""Compatibility facade for legacy data I/O imports."""

from football_outcomes.data.snapshots import (
    SNAPSHOT_VERSION,
    load_snapshot,
    save_snapshot,
    try_load_snapshot,
)
from football_outcomes.data.sofifa_ingestion import (
    _clean_csv_cell,
    _extract_skill_block,
    _is_number_like,
    _is_number_or_empty,
    _is_text_non_numeric,
    _list_csv_snapshots,
    _parse_date_flexible,
    _safe_csv_cell,
    _should_shift_skills_left_by_2,
    load_avg_team_strength,
    load_sofifa_players,
)

__all__ = [
    "SNAPSHOT_VERSION",
    "load_snapshot",
    "try_load_snapshot",
    "save_snapshot",
    "load_avg_team_strength",
    "load_sofifa_players",
    "_parse_date_flexible",
    "_list_csv_snapshots",
    "_clean_csv_cell",
    "_safe_csv_cell",
    "_is_number_like",
    "_is_number_or_empty",
    "_is_text_non_numeric",
    "_extract_skill_block",
    "_should_shift_skills_left_by_2",
]
