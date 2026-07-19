from __future__ import annotations

import http.client
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data import league_table as _league_table


def _conn_host() -> str:
    return sett.FS_HOST.replace("https://", "").replace("http://", "")


@dataclass
class FSDataBundle:
    comp_seasons: Dict[int, "FSCompSeason"] = field(default_factory=dict)
    teams: Dict[int, "FSTeam"] = field(default_factory=dict)
    players: Dict[int, "FSPlayer"] = field(default_factory=dict)
    matches: List["FSMatch"] = field(default_factory=list)

    leagues_list: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    sofifa_snapshots: List[Tuple[date, Dict[int, Dict[str, Any]]]] = field(default_factory=list)
    sofifa_player_occurrences: Dict[int, List[Tuple[int, date]]] = field(default_factory=dict)
    sofifa_players_by_dob: Dict[date, List[Tuple[int, str, str]]] = field(default_factory=dict)

    fs_to_sofifa_cache: Dict[int, Tuple[Optional[int], float, float, bool, str]] = field(default_factory=dict)

    sofifa_team_meta: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    sofifa_players_by_team: Dict[int, List[Tuple[int, str, str, Optional[date]]]] = field(default_factory=dict)
    sofifa_teams_by_league: Any = field(default_factory=dict)
    fs_team_to_sofifa_team: Dict[int, int] = field(default_factory=dict)

    def __setstate__(self, state: dict) -> None:
        # Backward compatible load: populate missing keys with defaults
        self.__dict__.update(state)

        if "sofifa_snapshots" not in self.__dict__:
            self.sofifa_snapshots = []
        if "sofifa_player_occurrences" not in self.__dict__:
            self.sofifa_player_occurrences = {}
        if "sofifa_players_by_dob" not in self.__dict__:
            self.sofifa_players_by_dob = {}
        if "fs_to_sofifa_cache" not in self.__dict__:
            self.fs_to_sofifa_cache = {}
        if "sofifa_team_meta" not in self.__dict__:
            self.sofifa_team_meta = {}
        if "sofifa_players_by_team" not in self.__dict__:
            self.sofifa_players_by_team = {}
        if "sofifa_teams_by_league" not in self.__dict__:
            self.sofifa_teams_by_league = {}
        if "fs_team_to_sofifa_team" not in self.__dict__:
            self.fs_team_to_sofifa_team = {}


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

        # League table state
        self.teams: List["FSTeam"] = []
        self.team_stats: Dict[int, Dict[str, float]] = {}  # keyed by team_id
        # Pre-match normalized positions cache: match_id -> {team_id: pos in [0,1]}
        self._pre_match_positions: Dict[int, Dict[int, float]] = {}
        self._table_initialized: bool = False

        self.conn = http.client.HTTPSConnection(_conn_host())  # transient (not pickled)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("conn", None)  # transient
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.conn = http.client.HTTPSConnection(_conn_host())

    # ------------------------------------------------------------------
    # Helpers for table-eligible matches
    # ------------------------------------------------------------------

    def _valid_round_ids_for_this_season(
        self,
    ) -> Optional[set[int]]:
        return _league_table.valid_round_ids_for_season(self)

    def _is_match_table_eligible(
        self,
        match: FSMatch,
    ) -> bool:
        return _league_table.is_match_table_eligible(
            self,
            match,
        )

    def _get_table_matches(
        self,
    ) -> List[FSMatch]:
        return _league_table.get_table_matches(self)

    @staticmethod
    def _match_time_key(
        match: FSMatch,
    ):
        return _league_table.match_time_key(match)

    def init_league_table(
        self,
    ) -> None:
        _league_table.init_league_table(self)

    def _ensure_table(
        self,
    ) -> None:
        _league_table.ensure_table(self)

    def reset_table(
        self,
    ) -> None:
        _league_table.reset_table(self)

    def _apply_match_to_table(
        self,
        match: FSMatch,
    ) -> None:
        _league_table.apply_match_to_table(
            self,
            match,
        )

    def _recompute_avg_points(
        self,
    ) -> None:
        (_league_table.recompute_average_points(self))

    def _sorted_team_ids(
        self,
    ) -> List[int]:
        return _league_table.sorted_team_ids(self)

    @staticmethod
    def _rank_to_position01(
        rank_1based: int,
        n_teams: int,
    ) -> float:
        return _league_table.rank_to_position01(
            rank_1based,
            n_teams,
        )

    def build_pre_match_positions_cache(
        self,
    ) -> None:
        (_league_table.build_pre_match_positions_cache(self))

    def get_team_position_before_match(
        self,
        team_id: int,
        match: FSMatch,
    ) -> float:
        return _league_table.get_team_position_before_match(
            self,
            team_id,
            match,
        )

    def get_team_position_up_to_match(
        self,
        team_id: int,
        match: FSMatch,
    ) -> float:
        return _league_table.get_team_position_up_to_match(
            self,
            team_id,
            match,
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

        # True = regular season league match, False = post-season/playoff/split-stage/ir-reg. match
        self.regular_season: bool = False

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

        # Competition-season link (needed for table position + comp_id feature)
        self.comp_season_id: Optional[int] = None
        self.comp_name: Optional[str] = None
        self.country: Optional[str] = None

        # Features (pre-match)
        self.features_before_match: Optional["FSMatchFeatures"] = None

        # ELO (raw value only used for propagation)
        self.home_elo_after_match_raw: Optional[float] = None
        self.away_elo_after_match_raw: Optional[float] = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FSMatch) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def calculate_match_features(self, team_index_league, team_index_all) -> FSMatchFeatures:
        from football_outcomes.config import fs_settings as sett
        from football_outcomes.utils import fs_feature_utils as fu
        from football_outcomes.utils.fs_player_skill_utils import calculate_team_strength

        if self.home_team is None or self.away_team is None:
            raise ValueError("Match missing teams.")

        comp_season_id = self.comp_season_id
        if comp_season_id is None:
            raise ValueError(f"Match {self.id} missing comp_season_id.")

        # comp_id must be integer category index (stable)
        comp_name = self.comp_name or ""
        try:
            comp_id = sett.COMPS_LEAGUE.index(comp_name)
        except ValueError:
            # non-league comps can still exist in globals; we just keep -1
            comp_id = -1

        hour = int(self.hour_utc or 0)
        month = int(self.month or 1)
        hs, hc, ms, mc = fu.hour_month_cyclic(hour, month)

        mf = FSMatchFeatures(
            comp_id=comp_id,
            season=fu.normalize_season(self.season),
            home_team_id=self.home_team.id,
            away_team_id=self.away_team.id,
            hours_sin=hs,
            hours_cos=hc,
            month_sin=ms,
            month_cos=mc,
        )

        # ---- ELO (pre-match, computed from previous matches only, then stored on the match for next matches)
        mf.home_elo, mf.away_elo = fu.calculate_elo_for_match(
            team_index_league=team_index_league,
            curr_match=self,
        )

        # ---- Match position in season (requires populated first/last dates)
        g = Global.get_instance()
        cs = g.all_comp_seasons.get(comp_season_id)
        if cs is not None and cs.first_match_date is not None and cs.last_match_date is not None:
            total_seconds = (cs.last_match_date - cs.first_match_date).total_seconds()
            curr_seconds = (self.datetime - cs.first_match_date).total_seconds()
            # hour-level tie-break
            curr_seconds += float(hour) * 3600.0
            mf.match_position_in_season = (
                fu.clip01(curr_seconds / total_seconds) if total_seconds > 0 else sett.ALMOST_ZERO
            )
        else:
            mf.match_position_in_season = sett.ALMOST_ZERO

        # ---- xG averages
        mf.home_avg_xg_last_5 = fu.avg_stat_last_n(
            team_index_league, self.home_team.id, self, 5, "home_xg", "away_xg", fu.normalize_team_xg
        )
        mf.home_avg_xg_last_20 = fu.avg_stat_last_n(
            team_index_league, self.home_team.id, self, 20, "home_xg", "away_xg", fu.normalize_team_xg
        )
        mf.away_avg_xg_last_5 = fu.avg_stat_last_n(
            team_index_league, self.away_team.id, self, 5, "home_xg", "away_xg", fu.normalize_team_xg
        )
        mf.away_avg_xg_last_20 = fu.avg_stat_last_n(
            team_index_league, self.away_team.id, self, 20, "home_xg", "away_xg", fu.normalize_team_xg
        )

        mf.home_avg_xg_total_last_5 = fu.avg_total_stat_last_n(
            team_index_league, self.home_team.id, self, 5, fu.total_xg, fu.normalize_total_xg
        )
        mf.home_avg_xg_total_last_20 = fu.avg_total_stat_last_n(
            team_index_league, self.home_team.id, self, 20, fu.total_xg, fu.normalize_total_xg
        )
        mf.away_avg_xg_total_last_5 = fu.avg_total_stat_last_n(
            team_index_league, self.away_team.id, self, 5, fu.total_xg, fu.normalize_total_xg
        )
        mf.away_avg_xg_total_last_20 = fu.avg_total_stat_last_n(
            team_index_league, self.away_team.id, self, 20, fu.total_xg, fu.normalize_total_xg
        )

        # ---- pre-match xG averages
        mf.home_avg_pre_match_xg_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            5,
            "home_prematch_xg",
            "away_prematch_xg",
            fu.normalize_team_pre_match_xg,
        )
        mf.home_avg_pre_match_xg_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            20,
            "home_prematch_xg",
            "away_prematch_xg",
            fu.normalize_team_pre_match_xg,
        )
        mf.away_avg_pre_match_xg_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            5,
            "home_prematch_xg",
            "away_prematch_xg",
            fu.normalize_team_pre_match_xg,
        )
        mf.away_avg_pre_match_xg_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            20,
            "home_prematch_xg",
            "away_prematch_xg",
            fu.normalize_team_pre_match_xg,
        )

        mf.home_avg_pre_match_xg_total_last_5 = fu.avg_total_stat_last_n(
            team_index_league, self.home_team.id, self, 5, fu.total_pre_match_xg, fu.normalize_total_pre_match_xg
        )
        mf.home_avg_pre_match_xg_total_last_20 = fu.avg_total_stat_last_n(
            team_index_league, self.home_team.id, self, 20, fu.total_pre_match_xg, fu.normalize_total_pre_match_xg
        )
        mf.away_avg_pre_match_xg_total_last_5 = fu.avg_total_stat_last_n(
            team_index_league, self.away_team.id, self, 5, fu.total_pre_match_xg, fu.normalize_total_pre_match_xg
        )
        mf.away_avg_pre_match_xg_total_last_20 = fu.avg_total_stat_last_n(
            team_index_league, self.away_team.id, self, 20, fu.total_pre_match_xg, fu.normalize_total_pre_match_xg
        )

        # ---- match load
        mf.home_match_load_per_day_last_10_days = fu.match_load_per_day_last_n_days(
            team_index_all, self.home_team.id, self, 10
        )
        mf.home_match_load_per_day_last_25_days = fu.match_load_per_day_last_n_days(
            team_index_all, self.home_team.id, self, 25
        )
        mf.away_match_load_per_day_last_10_days = fu.match_load_per_day_last_n_days(
            team_index_all, self.away_team.id, self, 10
        )
        mf.away_match_load_per_day_last_25_days = fu.match_load_per_day_last_n_days(
            team_index_all, self.away_team.id, self, 25
        )

        # ---- points/goals
        mf.home_avg_points_last_5 = fu.avg_points_last_n(team_index_league, self.home_team.id, self, 5)
        mf.home_avg_points_last_20 = fu.avg_points_last_n(team_index_league, self.home_team.id, self, 20)
        mf.away_avg_points_last_5 = fu.avg_points_last_n(team_index_league, self.away_team.id, self, 5)
        mf.away_avg_points_last_20 = fu.avg_points_last_n(team_index_league, self.away_team.id, self, 20)

        mf.home_avg_goals_last_5 = fu.avg_goals_last_n(team_index_league, self.home_team.id, self, 5)
        mf.home_avg_goals_last_20 = fu.avg_goals_last_n(team_index_league, self.home_team.id, self, 20)
        mf.away_avg_goals_last_5 = fu.avg_goals_last_n(team_index_league, self.away_team.id, self, 5)
        mf.away_avg_goals_last_20 = fu.avg_goals_last_n(team_index_league, self.away_team.id, self, 20)

        # ---- shots/corners/etc.
        mf.home_avg_shots_on_target_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            5,
            "home_shots_on_target",
            "away_shots_on_target",
            fu.normalize_shots_on_g,
        )
        mf.home_avg_shots_on_target_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            20,
            "home_shots_on_target",
            "away_shots_on_target",
            fu.normalize_shots_on_g,
        )
        mf.away_avg_shots_on_target_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            5,
            "home_shots_on_target",
            "away_shots_on_target",
            fu.normalize_shots_on_g,
        )
        mf.away_avg_shots_on_target_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            20,
            "home_shots_on_target",
            "away_shots_on_target",
            fu.normalize_shots_on_g,
        )

        mf.home_avg_shots_off_target_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            5,
            "home_shots_off_target",
            "away_shots_off_target",
            fu.normalize_shots_off_g,
        )
        mf.home_avg_shots_off_target_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            20,
            "home_shots_off_target",
            "away_shots_off_target",
            fu.normalize_shots_off_g,
        )
        mf.away_avg_shots_off_target_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            5,
            "home_shots_off_target",
            "away_shots_off_target",
            fu.normalize_shots_off_g,
        )
        mf.away_avg_shots_off_target_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            20,
            "home_shots_off_target",
            "away_shots_off_target",
            fu.normalize_shots_off_g,
        )

        mf.home_avg_total_shots_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            5,
            "home_total_shots",
            "away_total_shots",
            fu.normalize_total_shots,
        )
        mf.home_avg_total_shots_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            20,
            "home_total_shots",
            "away_total_shots",
            fu.normalize_total_shots,
        )
        mf.away_avg_total_shots_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            5,
            "home_total_shots",
            "away_total_shots",
            fu.normalize_total_shots,
        )
        mf.away_avg_total_shots_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            20,
            "home_total_shots",
            "away_total_shots",
            fu.normalize_total_shots,
        )

        mf.home_avg_corner_kicks_last_5 = fu.avg_stat_last_n(
            team_index_league, self.home_team.id, self, 5, "home_corners", "away_corners", fu.normalize_corners
        )
        mf.home_avg_corner_kicks_last_20 = fu.avg_stat_last_n(
            team_index_league, self.home_team.id, self, 20, "home_corners", "away_corners", fu.normalize_corners
        )
        mf.away_avg_corner_kicks_last_5 = fu.avg_stat_last_n(
            team_index_league, self.away_team.id, self, 5, "home_corners", "away_corners", fu.normalize_corners
        )
        mf.away_avg_corner_kicks_last_20 = fu.avg_stat_last_n(
            team_index_league, self.away_team.id, self, 20, "home_corners", "away_corners", fu.normalize_corners
        )

        # possession/fouls/attacks/dang attacks are already in [0..100] or similar.
        # For now, we normalize by simple /100 for possession and /50 for fouls/attacks-ish later if needed.
        # To avoid inventing wrong caps, we keep them in [0..1] by clipping after /100 or /200 etc would be risky.
        # So: scale possession by /100; keep the rest min-max with conservative caps later (you can tune).
        def poss_norm(x: float) -> float:
            return fu.clip01(x / 100.0)

        mf.home_avg_ball_possession_last_5 = fu.avg_stat_last_n(
            team_index_league, self.home_team.id, self, 5, "home_possession", "away_possession", poss_norm
        )
        mf.home_avg_ball_possession_last_20 = fu.avg_stat_last_n(
            team_index_league, self.home_team.id, self, 20, "home_possession", "away_possession", poss_norm
        )
        mf.away_avg_ball_possession_last_5 = fu.avg_stat_last_n(
            team_index_league, self.away_team.id, self, 5, "home_possession", "away_possession", poss_norm
        )
        mf.away_avg_ball_possession_last_20 = fu.avg_stat_last_n(
            team_index_league, self.away_team.id, self, 20, "home_possession", "away_possession", poss_norm
        )

        # fouls/attacks/dangerous attacks left as scaled by conservative caps (TODO: tune later)
        mf.home_avg_fouls_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            5,
            "home_fouls",
            "away_fouls",
            fu.normalize_fouls,
        )
        mf.home_avg_fouls_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            20,
            "home_fouls",
            "away_fouls",
            fu.normalize_fouls,
        )
        mf.away_avg_fouls_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            5,
            "home_fouls",
            "away_fouls",
            fu.normalize_fouls,
        )
        mf.away_avg_fouls_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            20,
            "home_fouls",
            "away_fouls",
            fu.normalize_fouls,
        )

        mf.home_avg_attacks_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            5,
            "home_attacks",
            "away_attacks",
            fu.normalize_attacks,
        )
        mf.home_avg_attacks_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            20,
            "home_attacks",
            "away_attacks",
            fu.normalize_attacks,
        )
        mf.away_avg_attacks_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            5,
            "home_attacks",
            "away_attacks",
            fu.normalize_attacks,
        )
        mf.away_avg_attacks_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            20,
            "home_attacks",
            "away_attacks",
            fu.normalize_attacks,
        )

        mf.home_avg_dang_attacks_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            5,
            "home_dangerous_attacks",
            "away_dangerous_attacks",
            fu.normalize_dang_attacks,
        )
        mf.home_avg_dang_attacks_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.home_team.id,
            self,
            20,
            "home_dangerous_attacks",
            "away_dangerous_attacks",
            fu.normalize_dang_attacks,
        )
        mf.away_avg_dang_attacks_last_5 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            5,
            "home_dangerous_attacks",
            "away_dangerous_attacks",
            fu.normalize_dang_attacks,
        )
        mf.away_avg_dang_attacks_last_20 = fu.avg_stat_last_n(
            team_index_league,
            self.away_team.id,
            self,
            20,
            "home_dangerous_attacks",
            "away_dangerous_attacks",
            fu.normalize_dang_attacks,
        )

        # ---- League table positions (assumes the earlier table init exists on cs)
        if cs is not None and hasattr(cs, "get_team_position_before_match"):
            mf.home_curr_position = cs.get_team_position_before_match(self.home_team.id, self)
            mf.away_curr_position = cs.get_team_position_before_match(self.away_team.id, self)
        else:
            mf.home_curr_position = sett.ALMOST_ZERO
            mf.away_curr_position = sett.ALMOST_ZERO

        # ---- Home/away-only scored/conceded features (from your old class)
        # These should be computed using only matches where team was home/away.
        mf.home_avg_goals_scored_home_last_5, mf.home_avg_goals_conceded_home_last_5 = (
            fu.avg_goals_scored_conceded_role_last_n(team_index_league, self.home_team.id, self, 5, "home")
        )
        mf.home_avg_goals_scored_home_last_20, mf.home_avg_goals_conceded_home_last_20 = (
            fu.avg_goals_scored_conceded_role_last_n(team_index_league, self.home_team.id, self, 20, "home")
        )

        mf.away_avg_goals_scored_away_last_5, mf.away_avg_goals_conceded_away_last_5 = (
            fu.avg_goals_scored_conceded_role_last_n(team_index_league, self.away_team.id, self, 5, "away")
        )
        mf.away_avg_goals_scored_away_last_20, mf.away_avg_goals_conceded_away_last_20 = (
            fu.avg_goals_scored_conceded_role_last_n(team_index_league, self.away_team.id, self, 20, "away")
        )

        # ---- Team strength calculation
        mf.home_team_strength = calculate_team_strength(self, self.home_team.id)
        mf.away_team_strength = calculate_team_strength(self, self.away_team.id)

        return mf


class FSMatchFeatures:
    def __init__(
        self,
        comp_id,
        season,
        home_team_id,
        away_team_id,
        hours_sin,
        hours_cos,
        month_sin,
        month_cos,
    ):
        self.comp_id = comp_id
        self.season = season

        self.home_team_id = home_team_id
        self.away_team_id = away_team_id

        self.hours_sin = hours_sin
        self.hours_cos = hours_cos
        self.month_sin = month_sin
        self.month_cos = month_cos

        self.home_elo = None
        self.away_elo = None

        self.match_position_in_season = None

        self.home_avg_xg_last_5 = None
        self.home_avg_xg_last_20 = None
        self.away_avg_xg_last_5 = None
        self.away_avg_xg_last_20 = None

        self.home_avg_xg_total_last_5 = None
        self.home_avg_xg_total_last_20 = None
        self.away_avg_xg_total_last_5 = None
        self.away_avg_xg_total_last_20 = None

        self.home_avg_pre_match_xg_last_5 = None
        self.home_avg_pre_match_xg_last_20 = None
        self.away_avg_pre_match_xg_last_5 = None
        self.away_avg_pre_match_xg_last_20 = None

        self.home_avg_pre_match_xg_total_last_5 = None
        self.home_avg_pre_match_xg_total_last_20 = None
        self.away_avg_pre_match_xg_total_last_5 = None
        self.away_avg_pre_match_xg_total_last_20 = None

        self.home_match_load_per_day_last_10_days = None
        self.home_match_load_per_day_last_25_days = None
        self.away_match_load_per_day_last_10_days = None
        self.away_match_load_per_day_last_25_days = None

        self.home_avg_points_last_5 = None
        self.home_avg_points_last_20 = None
        self.away_avg_points_last_5 = None
        self.away_avg_points_last_20 = None

        self.home_avg_goals_last_5 = None
        self.home_avg_goals_last_20 = None
        self.away_avg_goals_last_5 = None
        self.away_avg_goals_last_20 = None

        self.home_avg_shots_on_target_last_5 = None
        self.home_avg_shots_on_target_last_20 = None
        self.away_avg_shots_on_target_last_5 = None
        self.away_avg_shots_on_target_last_20 = None

        self.home_avg_shots_off_target_last_5 = None
        self.home_avg_shots_off_target_last_20 = None
        self.away_avg_shots_off_target_last_5 = None
        self.away_avg_shots_off_target_last_20 = None

        self.home_avg_total_shots_last_5 = None
        self.home_avg_total_shots_last_20 = None
        self.away_avg_total_shots_last_5 = None
        self.away_avg_total_shots_last_20 = None

        self.home_avg_corner_kicks_last_5 = None
        self.home_avg_corner_kicks_last_20 = None
        self.away_avg_corner_kicks_last_5 = None
        self.away_avg_corner_kicks_last_20 = None

        self.home_avg_ball_possession_last_5 = None
        self.home_avg_ball_possession_last_20 = None
        self.away_avg_ball_possession_last_5 = None
        self.away_avg_ball_possession_last_20 = None

        self.home_avg_fouls_last_5 = None
        self.home_avg_fouls_last_20 = None
        self.away_avg_fouls_last_5 = None
        self.away_avg_fouls_last_20 = None

        self.home_avg_attacks_last_5 = None
        self.home_avg_attacks_last_20 = None
        self.away_avg_attacks_last_5 = None
        self.away_avg_attacks_last_20 = None

        self.home_avg_dang_attacks_last_5 = None
        self.home_avg_dang_attacks_last_20 = None
        self.away_avg_dang_attacks_last_5 = None
        self.away_avg_dang_attacks_last_20 = None

        self.home_curr_position = None
        self.away_curr_position = None

        self.home_avg_goals_scored_home_last_5 = 0
        self.home_avg_goals_scored_home_last_20 = 0
        self.away_avg_goals_scored_away_last_5 = 0
        self.away_avg_goals_scored_away_last_20 = 0

        self.home_avg_goals_conceded_home_last_5 = 0
        self.home_avg_goals_conceded_home_last_20 = 0
        self.away_avg_goals_conceded_away_last_5 = 0
        self.away_avg_goals_conceded_away_last_20 = 0

        self.home_team_strength = None
        self.away_team_strength = None

    @staticmethod
    def match_features_to_vector(f: "FSMatchFeatures") -> List[float]:
        """
        Convert FSMatchFeatures into a flat float vector.

        Notes:
        - Excludes categorical IDs (home_team_id, away_team_id). Keeps comp_id and season as numeric scalars
          (you can one-hot later).
        - Includes team strength matrices by direct flattening (temporary until autoencoder embedding is added).
        - Converts None to 0.0.
        - Keeps a fixed, explicit ordering (do NOT change lightly once you train models).
        """

        def v(x: Optional[float]) -> float:
            return 0.0 if x is None else float(x)

        vec: List[float] = []

        # --- Core categorical-as-numeric (stable small ints)
        vec.append(float(f.comp_id))
        vec.append(float(f.season))

        # --- Cyclic time features
        vec.append(v(f.hours_sin))
        vec.append(v(f.hours_cos))
        vec.append(v(f.month_sin))
        vec.append(v(f.month_cos))

        # --- Elo + season position
        vec.append(v(f.home_elo))
        vec.append(v(f.away_elo))
        vec.append(v(f.match_position_in_season))

        # --- xG features
        vec.append(v(f.home_avg_xg_last_5))
        vec.append(v(f.home_avg_xg_last_20))
        vec.append(v(f.away_avg_xg_last_5))
        vec.append(v(f.away_avg_xg_last_20))

        vec.append(v(f.home_avg_xg_total_last_5))
        vec.append(v(f.home_avg_xg_total_last_20))
        vec.append(v(f.away_avg_xg_total_last_5))
        vec.append(v(f.away_avg_xg_total_last_20))

        vec.append(v(f.home_avg_pre_match_xg_last_5))
        vec.append(v(f.home_avg_pre_match_xg_last_20))
        vec.append(v(f.away_avg_pre_match_xg_last_5))
        vec.append(v(f.away_avg_pre_match_xg_last_20))

        vec.append(v(f.home_avg_pre_match_xg_total_last_5))
        vec.append(v(f.home_avg_pre_match_xg_total_last_20))
        vec.append(v(f.away_avg_pre_match_xg_total_last_5))
        vec.append(v(f.away_avg_pre_match_xg_total_last_20))

        # --- match load
        vec.append(v(f.home_match_load_per_day_last_10_days))
        vec.append(v(f.home_match_load_per_day_last_25_days))
        vec.append(v(f.away_match_load_per_day_last_10_days))
        vec.append(v(f.away_match_load_per_day_last_25_days))

        # --- points/goals
        vec.append(v(f.home_avg_points_last_5))
        vec.append(v(f.home_avg_points_last_20))
        vec.append(v(f.away_avg_points_last_5))
        vec.append(v(f.away_avg_points_last_20))

        vec.append(v(f.home_avg_goals_last_5))
        vec.append(v(f.home_avg_goals_last_20))
        vec.append(v(f.away_avg_goals_last_5))
        vec.append(v(f.away_avg_goals_last_20))

        # --- shots/corners/possession/fouls/attacks
        vec.append(v(f.home_avg_shots_on_target_last_5))
        vec.append(v(f.home_avg_shots_on_target_last_20))
        vec.append(v(f.away_avg_shots_on_target_last_5))
        vec.append(v(f.away_avg_shots_on_target_last_20))

        vec.append(v(f.home_avg_shots_off_target_last_5))
        vec.append(v(f.home_avg_shots_off_target_last_20))
        vec.append(v(f.away_avg_shots_off_target_last_5))
        vec.append(v(f.away_avg_shots_off_target_last_20))

        vec.append(v(f.home_avg_total_shots_last_5))
        vec.append(v(f.home_avg_total_shots_last_20))
        vec.append(v(f.away_avg_total_shots_last_5))
        vec.append(v(f.away_avg_total_shots_last_20))

        vec.append(v(f.home_avg_corner_kicks_last_5))
        vec.append(v(f.home_avg_corner_kicks_last_20))
        vec.append(v(f.away_avg_corner_kicks_last_5))
        vec.append(v(f.away_avg_corner_kicks_last_20))

        vec.append(v(f.home_avg_ball_possession_last_5))
        vec.append(v(f.home_avg_ball_possession_last_20))
        vec.append(v(f.away_avg_ball_possession_last_5))
        vec.append(v(f.away_avg_ball_possession_last_20))

        vec.append(v(f.home_avg_fouls_last_5))
        vec.append(v(f.home_avg_fouls_last_20))
        vec.append(v(f.away_avg_fouls_last_5))
        vec.append(v(f.away_avg_fouls_last_20))

        vec.append(v(f.home_avg_attacks_last_5))
        vec.append(v(f.home_avg_attacks_last_20))
        vec.append(v(f.away_avg_attacks_last_5))
        vec.append(v(f.away_avg_attacks_last_20))

        vec.append(v(f.home_avg_dang_attacks_last_5))
        vec.append(v(f.home_avg_dang_attacks_last_20))
        vec.append(v(f.away_avg_dang_attacks_last_5))
        vec.append(v(f.away_avg_dang_attacks_last_20))

        # --- league table position
        vec.append(v(f.home_curr_position))
        vec.append(v(f.away_curr_position))

        # --- home/away-only scored/conceded
        vec.append(v(f.home_avg_goals_scored_home_last_5))
        vec.append(v(f.home_avg_goals_scored_home_last_20))
        vec.append(v(f.away_avg_goals_scored_away_last_5))
        vec.append(v(f.away_avg_goals_scored_away_last_20))

        vec.append(v(f.home_avg_goals_conceded_home_last_5))
        vec.append(v(f.home_avg_goals_conceded_home_last_20))
        vec.append(v(f.away_avg_goals_conceded_away_last_5))
        vec.append(v(f.away_avg_goals_conceded_away_last_20))

        # --- team strength (temporary: raw flatten)
        def flatten_strength(mat) -> List[float]:
            if mat is None:
                return [0.0] * (11 * 34)
            out: List[float] = []
            # Expect list[list[float]] of size 11x34
            for row in mat:
                for x in row:
                    out.append(float(x))
            # pad if malformed
            need = 11 * 34
            if len(out) < need:
                out.extend([0.0] * (need - len(out)))
            return out[:need]

        vec.extend(flatten_strength(f.home_team_strength))
        vec.extend(flatten_strength(f.away_team_strength))

        return vec
