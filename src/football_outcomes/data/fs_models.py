from __future__ import annotations

import http.client
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from football_outcomes.config import settings


@dataclass
class FSDataBundle:
    """
    Snapshot container.
    """

    comp_seasons: Dict[int, "FSCompSeason"] = field(default_factory=dict)
    teams: Dict[int, "FSTeam"] = field(default_factory=dict)
    players: Dict[int, "FSPlayer"] = field(default_factory=dict)
    matches: List["FSMatch"] = field(default_factory=list)

    leagues_list: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __setstate__(self, state: dict) -> None:  # called by pickle on load (if present)
        self.__dict__.update(state)


class FSCompSeason:
    def __init__(self, id_: int, season_: int, country_: str, name_: str):
        self.id = id_
        self.season = season_
        self.country = country_
        self.name = name_
        self.format: Optional[str] = None
        self.domestic_scale: Optional[int] = None
        self.division: Optional[int] = None
        self.total_game_week: Optional[int] = None
        self.matches: List["FSMatch"] = []
        self.first_match_date = None
        self.last_match_date = None

        # League table (for competitions in sett.COMPS_LEAGUE)
        self.teams: List["FSTeam"] = []
        self.team_stats: Dict[int, Dict[str, float]] = {}  # keyed by team_id
        # Pre-match normalized positions cache: match_id -> {team_id: pos in [0,1]}
        self._pre_match_positions: Dict[int, Dict[int, float]] = {}
        self._table_initialized: bool = False

        self.conn = http.client.HTTPSConnection(settings.HOST)  # transient (not pickled)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("conn", None)  # transient
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.conn = http.client.HTTPSConnection(settings.HOST)

    # --------------------------
    # League table functionality
    # --------------------------

    def init_league_table(self) -> None:
        """Initialize teams + empty table stats for this competition season.

        Teams are inferred from self.matches (home/away participants).
        """
        team_by_id: Dict[int, "FSTeam"] = {}
        for m in self.matches:
            if m.home_team is not None:
                team_by_id[m.home_team.id] = m.home_team
            if m.away_team is not None:
                team_by_id[m.away_team.id] = m.away_team

        self.teams = sorted(team_by_id.values(), key=lambda t: t.id)
        self.team_stats = {
            t.id: {
                "points": 0.0,
                "games_played": 0.0,
                "goals_for": 0.0,
                "goals_against": 0.0,
                "avg_points_per_game": 0.0,
            }
            for t in self.teams
        }
        self._pre_match_positions = {}
        self._table_initialized = True

    def _ensure_table(self) -> None:
        if not self._table_initialized:
            self.init_league_table()

    def reset_table(self) -> None:
        """Reset all stats to 0 but keep the team roster."""
        self._ensure_table()
        for tid in self.team_stats:
            self.team_stats[tid].update(
                points=0.0,
                games_played=0.0,
                goals_for=0.0,
                goals_against=0.0,
                avg_points_per_game=0.0,
            )

    def _apply_match_to_table(self, match: "FSMatch") -> None:
        """Update stats using a single completed match."""
        self._ensure_table()
        if match.home_team is None or match.away_team is None:
            raise ValueError("League table update failed: Team not found.")
        if match.home_goals is None or match.away_goals is None:
            raise ValueError("League table update failed: Team goals not found.")

        hid = match.home_team.id
        aid = match.away_team.id

        if hid not in self.team_stats or aid not in self.team_stats:
            raise ValueError("League table update failed: Team ID not found.")

        hg = float(match.home_goals)
        ag = float(match.away_goals)

        self.team_stats[hid]["games_played"] += 1.0
        self.team_stats[hid]["goals_for"] += hg
        self.team_stats[hid]["goals_against"] += ag

        self.team_stats[aid]["games_played"] += 1.0
        self.team_stats[aid]["goals_for"] += ag
        self.team_stats[aid]["goals_against"] += hg

        if hg > ag:
            self.team_stats[hid]["points"] += 3.0
        elif hg < ag:
            self.team_stats[aid]["points"] += 3.0
        else:
            self.team_stats[hid]["points"] += 1.0
            self.team_stats[aid]["points"] += 1.0

    def _recompute_avg_points(self) -> None:
        self._ensure_table()
        for s in self.team_stats.values():
            gp = s["games_played"]
            s["avg_points_per_game"] = (s["points"] / gp) if gp > 0 else 0.0

    def _sorted_team_ids(self) -> List[int]:
        """Sort teams by current table rules (deterministic tie-break)."""
        self._ensure_table()
        self._recompute_avg_points()

        def key(tid: int):
            s = self.team_stats[tid]
            gd = s["goals_for"] - s["goals_against"]
            # last component: -tid ensures deterministic ordering for exact ties
            return s["avg_points_per_game"], gd, s["goals_for"], -tid

        return sorted(self.team_stats.keys(), key=key, reverse=True)

    @staticmethod
    def _rank_to_position01(rank_1based: int, n_teams: int) -> float:
        """Best team => 1.0, worst => 0.0 (unlike the old 1 - pos/len behavior)."""
        if n_teams <= 1:
            return 1.0
        return float(1.0 - ((rank_1based - 1) / (n_teams - 1)))

    @staticmethod
    def _match_time_key(m: "FSMatch"):
        """
        Ordering key for pre-match table states.
        datetime is UTC date at 00:00, hour_utc refines ordering within the date.
        Missing hour_utc falls back to -1 (will be treated as earliest on that day).
        """
        dt = getattr(m, "datetime", None)
        hr = getattr(m, "hour_utc", None)
        hr = int(hr) if isinstance(hr, int) else -1
        return dt, hr, m.id

    def build_pre_match_positions_cache(self) -> None:
        """Precompute positions for every match in this season using (date, hour_utc).

        Positions are computed *pre-match* based on all matches strictly before that match's (date, hour_utc).

        Matches with the same (date, hour_utc) are treated as a batch:
        - compute positions from table state before the batch
        - cache positions for each match in the batch
        - apply the batch to advance the table
        """
        self._ensure_table()
        self.reset_table()

        matches_sorted = sorted(
            [m for m in self.matches if getattr(m, "datetime", None) is not None],
            key=self._match_time_key,
        )

        self._pre_match_positions = {}

        i = 0
        while i < len(matches_sorted):
            dt_i, hr_i, _ = self._match_time_key(matches_sorted[i])

            batch = []
            while i < len(matches_sorted):
                dt_j, hr_j, _ = self._match_time_key(matches_sorted[i])
                if dt_j != dt_i or hr_j != hr_i:
                    break
                batch.append(matches_sorted[i])
                i += 1

            # Positions before the batch
            ordered = self._sorted_team_ids()
            n = len(ordered)
            rank_by_team = {tid: r for r, tid in enumerate(ordered, start=1)}

            for m in batch:
                if m.home_team is None or m.away_team is None:
                    raise ValueError("Precomputing positions failed: Match contains a None team.")
                hid = m.home_team.id
                aid = m.away_team.id
                self._pre_match_positions[m.id] = {
                    hid: self._rank_to_position01(rank_by_team.get(hid, n), n),
                    aid: self._rank_to_position01(rank_by_team.get(aid, n), n),
                }

            # apply batch
            for m in batch:
                self._apply_match_to_table(m)

        self._recompute_avg_points()

    def get_team_position_before_match(self, team_id: int, match: "FSMatch") -> float:
        """O(1) lookup of normalized position for team_id before the given match."""
        self._ensure_table()
        pos_map = self._pre_match_positions.get(match.id)
        if pos_map is not None and team_id in pos_map:
            return pos_map[team_id]

        # Fallback if cache missing
        return self.get_team_position_up_to_date(team_id, getattr(match, "datetime", None))

    def get_team_position_up_to_date(self, team_id: int, date) -> float:
        """Slow fallback: recompute from matches strictly before date."""
        self._ensure_table()
        if date is None:
            return 0.0

        self.reset_table()
        for m in sorted(self.matches, key=lambda m: (getattr(m, "datetime", None) or 0, m.id)):
            if getattr(m, "datetime", None) is None:
                continue
            if m.datetime >= date:
                break
            self._apply_match_to_table(m)

        ordered = self._sorted_team_ids()
        n = len(ordered)
        for rank, tid in enumerate(ordered, start=1):
            if tid == team_id:
                return self._rank_to_position01(rank, n)

        raise ValueError(
            f"Team [{team_id}] not found in table for competition season [{self.name}, {self.season}] (id={self.id})."
        )


class FSTeam:
    def __init__(
        self, id_: int, name_: str, clean_n: str, english_n: str, full_n: str, shorthand_n: str, country_: str
    ):
        self.id = id_
        self.name = name_
        self.comp_seasons: Dict[int, list["FSPlayer"]] = {}
        self.clean_name = clean_n
        self.english_name = english_n
        self.full_name = full_n
        self.shorthand_name = shorthand_n
        self.country = country_

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FSTeam) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class FSPlayer:
    def __init__(self, id_: int, name: str, first: str, last: str, short: str, known_as_: str):
        self.id = id_
        self.full_name = name
        self.first_name = first
        self.last_name = last
        self.shorthand = short
        self.known_as = known_as_
        self.position: Optional[str] = None
        self.birthday = None
        self.nationality: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FSPlayer) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class FSMatch:
    def __init__(self, id_: int):
        self.id = id_
        self.home_team: Optional[FSTeam] = None
        self.away_team: Optional[FSTeam] = None
        self.season: Optional[int] = None
        self.round_id: Optional[int] = None
        self.game_week: Optional[int] = None
        self.datetime = None
        self.month: Optional[int] = None
        self.hour_utc: Optional[int] = None
        self.hour_local: Optional[int] = None

        self.home_points: Optional[int] = None
        self.away_points: Optional[int] = None
        self.home_goals: Optional[int] = None
        self.away_goals: Optional[int] = None

        self.stats = {
            "home_corners": -1,
            "away_corners": -1,
            "home_offsides": -1,
            "away_offsides": -1,
            "home_red_cards": -1,
            "away_red_cards": -1,
            "home_yellow_cards": -1,
            "away_yellow_cards": -1,
            "home_shots_on_target": -1,
            "away_shots_on_target": -1,
            "home_shots_off_target": -1,
            "away_shots_off_target": -1,
            "home_total_shots": -1,
            "away_total_shots": -1,
            "home_fouls": -1,
            "away_fouls": -1,
            "home_possession": -1,
            "away_possession": -1,
            "home_attacks": -1,
            "away_attacks": -1,
            "home_dangerous_attacks": -1,
            "away_dangerous_attacks": -1,
            "home_xg": -1,
            "away_xg": -1,
            "home_prematch_xg": -1,
            "away_prematch_xg": -1,
        }

        self.referee_id: Optional[int] = None
        self.home_coach_id: Optional[int] = None
        self.away_coach_id: Optional[int] = None

        self.odds = {
            "home_win": -1,
            "away_win": -1,
            "draw": -1,
            "over05": -1,
            "over15": -1,
            "over25": -1,
            "over35": -1,
            "over45": -1,
            "under05": -1,
            "under15": -1,
            "under25": -1,
            "under35": -1,
            "under45": -1,
            "btts_yes": -1,
            "btts_no": -1,
        }

        self.home_lineup: List[FSPlayer] = []
        self.away_lineup: List[FSPlayer] = []

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FSMatch) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
