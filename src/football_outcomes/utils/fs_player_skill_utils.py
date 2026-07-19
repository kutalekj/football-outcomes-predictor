"""Compatibility facade for legacy player-skill utilities."""

from football_outcomes.data import lineups as _lineups
from football_outcomes.data import sofifa_player_matching as _player_matching
from football_outcomes.data import sofifa_skills as _sofifa_skills
from football_outcomes.data import sofifa_team_matching as _team_matching
from football_outcomes.data import team_strength_matrix as _team_strength_matrix

# Lineup compatibility exports
_FS_POS_ORDER = _lineups.FS_POSITION_ORDER
_pos_rank = _lineups.position_rank
_select_and_sort_lineup = _lineups.select_and_sort_lineup
calculate_team_position_indices = _lineups.calculate_team_position_indices

# Temporal skill compatibility exports
_ordered_snapshot_candidates = _sofifa_skills.ordered_snapshot_candidates
_merge_skills_from_snapshots = _sofifa_skills.merge_skills_from_snapshots

# Player-matching compatibility exports
_get_team_strength_log_path = _player_matching._get_team_strength_log_path
_dbg = _player_matching._dbg
_norm_name = _player_matching._norm_name
_player_display_name = _player_matching._player_display_name

MatchCandidate = _player_matching.MatchCandidate
MatchResult = _player_matching.MatchResult

_similarity = _player_matching._similarity
_name_key_last_firstinit = _player_matching._name_key_last_firstinit
_ensure_sofifa_namekey_index = _player_matching._ensure_sofifa_namekey_index
_build_name_bucket = _player_matching._build_name_bucket
_match_fs_to_sofifa = _player_matching._match_fs_to_sofifa

# Team-index and team-matching exports
_norm_country = _team_matching._norm_country
_norm_league = _team_matching._norm_league
_norm_team = _team_matching._norm_team
_try_get_first = _team_matching._try_get_first
_extract_sofifa_team_info = _team_matching._extract_sofifa_team_info
build_sofifa_team_indexes = _team_matching.build_sofifa_team_indexes
match_fs_teams_to_sofifa_teams = _team_matching.match_fs_teams_to_sofifa_teams

# Matrix compatibility exports
_gk_role_score = _team_strength_matrix.goalkeeper_role_score
_ensure_one_goalkeeper_row = _team_strength_matrix.ensure_one_goalkeeper_row
calculate_team_strength = _team_strength_matrix.calculate_team_strength
