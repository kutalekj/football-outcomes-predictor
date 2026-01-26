from __future__ import annotations

import csv
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import football_outcomes.config.fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_models import FSDataBundle

SNAPSHOT_VERSION = 1


def load_snapshot(path: Path = sett.LOAD_SNAPSHOT_PATH) -> FSDataBundle:
    with path.open("rb") as f:
        bundle: FSDataBundle = pickle.load(f)

    # Simple version check
    version = bundle.meta.get("snapshot_version", 0)
    if version != SNAPSHOT_VERSION:
        raise RuntimeError(f"Incompatible snapshot version {version}; expected {SNAPSHOT_VERSION}.")

    return bundle


def try_load_snapshot(path: Path = sett.LOAD_SNAPSHOT_PATH) -> Optional[FSDataBundle]:
    if not path.exists():
        return None
    try:
        return load_snapshot(path)
    except Exception as e:
        print(f"Warning: failed to load snapshot ({e}). Rebuilding from API…")
        return None


def save_snapshot(bundle: FSDataBundle, path: Path = sett.SAVE_SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle.meta["snapshot_version"] = SNAPSHOT_VERSION
    print(f"Saving snapshot to: {path.resolve()}")

    with path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Snapshot saved.")


def load_avg_team_strength():
    global_instance = Global.get_instance()

    with open(sett.AVG_TEAM_STRENGTH_PATH, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        _ = next(reader)  # header row

        for row in reader:
            season = int(row[0])
            team_id = int(row[1])
            position_category = row[3]
            skill_values = list(map(float, row[4:]))  # convert skill values to floats

            if len(skill_values) != len(sett.PLAYER_SKILLS):
                raise ValueError(
                    f"Unexpected length of SOFIFA player skill values list for "
                    f"season [{season}], team ID [{team_id}], {position_category}: "
                    f"{len(skill_values)} loaded, but {len(sett.PLAYER_SKILLS)} we expected"
                )

            global_instance.sf_avg_team_strength[(season, team_id, position_category)] = skill_values

    print(f"[0] Successfully loaded team strength data from {sett.AVG_TEAM_STRENGTH_PATH}")


def _parse_date_flexible(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def _list_csv_snapshots(csv_dir: Path) -> List[Tuple[date, Path]]:
    files = [p for p in csv_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    out: List[Tuple[date, Path]] = []
    for p in files:
        try:
            d = datetime.strptime(p.stem, sett.SOFIFA_FILENAME_DATE_FORMAT).date()
        except Exception:
            continue
        out.append((d, p))
    out.sort(key=lambda x: x[0])
    return out


def load_sofifa_players(*, rebuild: bool = False) -> None:
    """
    Loads SOFIFA player snapshots from CSV files.

    Missing or invalid skill values are stored as -1.0 (no imputation here).
    Any later imputation/averaging should be done during feature building.

    Populates Global:
      - sofifa_snapshots: list[(snapshot_date, {sofifa_id: record})]
      - sofifa_player_occurrences: {sofifa_id: [(snapshot_index, snapshot_date), ...]}
      - sofifa_players_by_dob: {dob: [(sofifa_id, name, full_name), ...]}

    Record stores minimal fields needed for matching + skills:
      - sofifa_id, name, full_name, dob, skills[list[float]]
    """
    g = Global.get_instance()

    # ---- Guard / rebuild semantics ----
    if rebuild:
        # hard reset sofifa-related globals (so we don't accumulate duplicates)
        g.sofifa_snapshots = []
        g.sofifa_player_occurrences = {}
        g.sofifa_players_by_dob = {}

        # indexes/mappings derived from snapshots
        g.sofifa_team_meta = {}
        g.sofifa_players_by_team = {}
        g.sofifa_teams_by_league = {}
        g.fs_team_to_sofifa_team = {}
    else:
        # If sofifa_snapshots already exists (e.g., loaded from snapshot), do NOT append again.
        if getattr(g, "sofifa_snapshots", None):
            if len(g.sofifa_snapshots) > 0:
                print(f"[sofifa] Using {len(g.sofifa_snapshots)} snapshots from cache; not reloading CSV.")
                return

    snapshots = _list_csv_snapshots(Path(sett.SOFIFA_CSV_DIR))
    if not snapshots:
        print(f"[sofifa] No CSV files found in {sett.SOFIFA_CSV_DIR}")
        return

    for snap_idx, (snap_date, path) in enumerate(snapshots):
        players_by_id: Dict[int, dict] = {}

        total_rows = 0
        num_skipped = 0
        num_loaded = 0
        num_missing_cells = 0

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, start=2):
                # Guard 1: CSV module detected overflow columns or malformed parsing
                if None in row:
                    num_skipped += 1
                    continue

                # Guard 2: ensure the CSV actually provides all expected skill headers
                # (prevents silent shifting if header set doesn't match PLAYER_SKILLS)
                missing_headers = [k for k in sett.PLAYER_SKILLS if k not in row]
                if missing_headers:
                    num_skipped += 1
                    continue

                total_rows += 1
                sofifa_id_raw = (row.get("player_id", "") or "").strip()
                name = (row.get("name", "") or "").strip()
                full_name = (row.get("full_name", "") or "").strip()
                dob = _parse_date_flexible(row.get("dob", ""))
                club_id = int(row["club_id"]) if row["club_id"] else None
                club_name = row["club_name"] or None
                club_league_id = int(row["club_league_id"]) if row["club_league_id"] else None
                club_league_name = row["club_league_name"] or None

                # Minimal required data for later matching
                if not sofifa_id_raw or not name or not full_name or dob is None:
                    num_skipped += 1
                    continue

                try:
                    sofifa_id = int(sofifa_id_raw)
                except Exception:
                    num_skipped += 1
                    continue

                # Normalize name similarly to old code
                name_norm = name.split(" - FIFA")[0].split(" -")[0].strip()

                skills: List[float] = []
                for skill in sett.PLAYER_SKILLS:
                    v = (row.get(skill, "") or "").strip()
                    if v == "":
                        skills.append(-1.0)
                        num_missing_cells += 1
                        continue
                    try:
                        skills.append(float(v))
                    except Exception:
                        skills.append(-1.0)
                        num_missing_cells += 1

                record = {
                    "sofifa_id": sofifa_id,
                    "name": name_norm,
                    "full_name": full_name,
                    "dob": dob,
                    "skills": skills,  # length == len(sett.PLAYER_SKILLS)
                    "snapshot_date": snap_date,
                    "club_id": club_id,
                    "club_name": club_name,
                    "club_league_id": club_league_id,
                    "club_league_name": club_league_name,
                }

                players_by_id[sofifa_id] = record
                num_loaded += 1

                g.sofifa_player_occurrences.setdefault(sofifa_id, []).append((snap_idx, snap_date))

                g.sofifa_players_by_dob.setdefault(dob, [])
                if sofifa_id not in {t[0] for t in g.sofifa_players_by_dob[dob]}:
                    g.sofifa_players_by_dob[dob].append((sofifa_id, name_norm, full_name))

        g.sofifa_snapshots.append((snap_date, players_by_id))

        total_cells = total_rows * len(sett.PLAYER_SKILLS)
        print(
            f"[sofifa] Loaded {num_loaded} players from {path.name} "
            f"(skipped {num_skipped}/{total_rows} ({num_skipped/total_rows:.2%}), missing skill "
            f"cells: {num_missing_cells}/{total_cells} ({num_missing_cells/total_cells:.2%}))."
        )

    # Sort dicts for stable iteration/debug
    g.sofifa_player_occurrences = dict(sorted(g.sofifa_player_occurrences.items()))
    g.sofifa_players_by_dob = dict(sorted(g.sofifa_players_by_dob.items(), key=lambda x: x[0]))
