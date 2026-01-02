from __future__ import annotations

import http.client
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from football_outcomes.config import settings


@dataclass
class FSDataBundle:
    """
    Snapshot container.

    Backward compatibility:
    - Old snapshots used attribute name: fs_leagues_list
    - New code uses: leagues_list
    """

    comp_seasons: Dict[int, "FSCompSeason"] = field(default_factory=dict)
    teams: Dict[int, "FSTeam"] = field(default_factory=dict)
    players: Dict[int, "FSPlayer"] = field(default_factory=dict)
    matches: List["FSMatch"] = field(default_factory=list)

    leagues_list: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- Backward compatibility helpers ----
    @property
    def fs_leagues_list(self) -> Any:
        # legacy alias
        return self.leagues_list

    @fs_leagues_list.setter
    def fs_leagues_list(self, value: Any) -> None:
        self.leagues_list = value

    def __setstate__(self, state: dict) -> None:  # called by pickle on load (if present)
        if "leagues_list" not in state and "fs_leagues_list" in state:
            state["leagues_list"] = state["fs_leagues_list"]  # rename handle
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

        self.conn = http.client.HTTPSConnection(settings.HOST)  # transient (not pickled)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("conn", None)  # transient
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.conn = http.client.HTTPSConnection(settings.HOST)


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
