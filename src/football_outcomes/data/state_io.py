# state_io.py
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from football_outcomes.config import settings
from football_outcomes.config.globals import Global

# Bump if you change the saved structure format
SNAPSHOT_VERSION = 1

# Default snapshot path (you can customize in settings if you prefer)
DEFAULT_SNAPSHOT_PATH = settings.PROCESSED_DIR / f"global_state_v{SNAPSHOT_VERSION}.fop"

# What to persist from Global (explicit, future-proof, easy to review)
GLOBAL_ATTRS_TO_SAVE = [
    "all_matches",
    "all_comps",
    "all_tables",
    "all_teams",
    "start_end_dates_per_country_season",
    "fs_leagues_list",
    "fs_leagues_matches",
    "sf_avg_team_strength",
    "sofifa_players_data",
    "sofifa_player_index_dict",
    "sofifa_players_by_dob",
]


@dataclass
class Manifest:
    version: int
    created_utc: str
    python_version: str
    compress: bool
    sha256_hex: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"), ensure_ascii=False)


def _build_payload_dict() -> Dict[str, Any]:
    g = Global.get_instance()
    payload: Dict[str, Any] = {}
    for attr in GLOBAL_ATTRS_TO_SAVE:
        # If the attribute doesn’t exist yet, store None (keeps loads resilient)
        payload[attr] = getattr(g, attr, None)
    return payload


def _atomic_write_bytes(target_path: Path, data: bytes) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(data)
    # Replace atomically (Windows: falls back to non-atomic but still safe enough here)
    if target_path.exists():
        os.replace(tmp_path, target_path)
    else:
        shutil.move(tmp_path, target_path)


def save_global_state(path: Optional[Path] = None, *, compress: bool = True) -> Path:
    """
    Serialize the entire Global state (as defined by GLOBAL_ATTRS_TO_SAVE) into a single snapshot file.
    Preserves object identity and cross-links via pickle.
    """
    # raise recursion ceiling for deep nested containers
    if sys.getrecursionlimit() < 100_000:
        sys.setrecursionlimit(100_000)

    snapshot_path = Path(path) if path else DEFAULT_SNAPSHOT_PATH

    # Build payload (a dict of Global fields) – not the Global instance itself
    # This avoids any singleton edge cases while still capturing the full object graph
    payload = _build_payload_dict()

    # Pickle with highest protocol (Python 3.10 supports protocol 5)
    pickled = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    # Optional compression
    if compress:
        pickled = gzip.compress(pickled)

    # Integrity hash
    sha256_hex = hashlib.sha256(pickled).hexdigest()

    manifest = (
        Manifest(
            version=SNAPSHOT_VERSION,
            created_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            python_version=sys.version.split()[0],
            compress=compress,
            sha256_hex=sha256_hex,
        )
        .to_json()
        .encode("utf-8")
    )

    # File layout:
    # [8-byte magic][4-byte manifest_len][manifest JSON][payload bytes]
    magic = b"FOPSTATE"
    manifest_len = len(manifest).to_bytes(4, "big")

    blob = magic + manifest_len + manifest + pickled
    _atomic_write_bytes(snapshot_path, blob)

    return snapshot_path


def load_global_state(path: Optional[Path] = None) -> None:
    """
    Load a previously saved snapshot and restore it into the existing Global singleton instance.
    Recreates HTTP connections via __setstate__ on Comp/Round/SeasonCompTable automatically.
    """
    # raise recursion ceiling for deep unpickling
    if sys.getrecursionlimit() < 100_000:
        sys.setrecursionlimit(100_000)

    snapshot_path = Path(path) if path else DEFAULT_SNAPSHOT_PATH
    with open(snapshot_path, "rb") as f:
        magic = f.read(8)
        if magic != b"FOPSTATE":
            raise ValueError("Not a valid FOP snapshot (magic header mismatch).")

        manifest_len = int.from_bytes(f.read(4), "big")
        manifest_raw = f.read(manifest_len)
        m = json.loads(manifest_raw.decode("utf-8"))

        # Basic sanity checks
        if m.get("version") != SNAPSHOT_VERSION:
            raise ValueError(f"Snapshot version mismatch: got {m.get('version')}, expected {SNAPSHOT_VERSION}.")
        compress = bool(m.get("compress", True))
        expected_sha = str(m.get("sha256_hex"))

        payload_bytes = f.read()

    # Verify integrity
    actual_sha = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("Snapshot integrity check failed (SHA-256 mismatch).")

    # Decompress
    if compress:
        payload_bytes = gzip.decompress(payload_bytes)

    # Unpickle
    payload = pickle.loads(payload_bytes)
    if not isinstance(payload, dict):
        raise ValueError("Unexpected payload format; expected a dict of Global attributes.")

    # Restore into the *existing* singleton (preserves singleton identity)
    g = Global.get_instance()

    # Clear existing lists/maps to avoid stale crosslinks, then assign loaded ones
    # (Order matters: objects in lists reference each other, so assign the whole graph first.)
    for attr in GLOBAL_ATTRS_TO_SAVE:
        setattr(g, attr, payload.get(attr, None))


def export_summary_csvs(out_dir: Optional[Path] = None, *, max_rows: int = 500) -> None:
    out = Path(out_dir) if out_dir else (settings.PROCESSED_DIR / "snapshot_exports")
    out.mkdir(parents=True, exist_ok=True)
    g = Global.get_instance()

    # 1) basic counts
    with open(out / "counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "count"])
        w.writerow(["all_matches", len(getattr(g, "all_matches", []) or [])])
        w.writerow(["all_comps", len(getattr(g, "all_comps", []) or [])])
        w.writerow(["all_teams", len(getattr(g, "all_teams", []) or [])])
        w.writerow(["all_tables", len(getattr(g, "all_tables", []) or [])])
        w.writerow(["fs_leagues_list", len(getattr(g, "fs_leagues_list", []) or [])])
        w.writerow(["fs_leagues_matches", len(getattr(g, "fs_leagues_matches", {}) or {})])
        w.writerow(["sofifa_players_data", len(getattr(g, "sofifa_players_data", []) or [])])
        w.writerow(["sofifa_player_index_dict", len(getattr(g, "sofifa_player_index_dict", {}) or {})])
        w.writerow(["sofifa_players_by_dob", len(getattr(g, "sofifa_players_by_dob", {}) or {})])

    # 2) tiny samples (IDs, names, datetimes): enough to spot issues quickly
    # matches
    with open(out / "matches_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "datetime", "home_team_id", "away_team_id", "comp_id", "season", "round"])
        for m in (getattr(g, "all_matches", []) or [])[:max_rows]:
            w.writerow(
                [
                    getattr(m, "id", ""),
                    getattr(m, "datetime", ""),
                    getattr(m.home_team, "id", "") if getattr(m, "home_team", None) else "",
                    getattr(m.away_team, "id", "") if getattr(m, "away_team", None) else "",
                    getattr(m.comp, "id", "") if getattr(m, "comp", None) else "",
                    getattr(m, "season", ""),
                    getattr(m.round, "name", "") if getattr(m, "round", None) else "",
                ]
            )

    # teams
    with open(out / "teams_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "fs_id", "fs_clean_name"])
        for t in (getattr(g, "all_teams", []) or [])[:max_rows]:
            w.writerow(
                [getattr(t, "id", ""), getattr(t, "name", ""), getattr(t, "fs_id", ""), getattr(t, "fs_clean_name", "")]
            )

    # comps
    with open(out / "comps_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "country", "regular_round_keywords"])
        for c in (getattr(g, "all_comps", []) or [])[:max_rows]:
            w.writerow(
                [
                    getattr(c, "id", ""),
                    getattr(c, "name", ""),
                    getattr(c, "country", ""),
                    ";".join(getattr(c, "regular_round_keywords", []) or []),
                ]
            )
