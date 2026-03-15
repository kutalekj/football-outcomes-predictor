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


def _clean_csv_cell(v: object) -> str:
    if v is None:
        return ""
    return str(v).replace("\ufeff", "").replace("\xa0", " ").strip()


def _safe_csv_cell(cells: List[str], idx: int) -> str:
    if 0 <= idx < len(cells):
        return _clean_csv_cell(cells[idx])
    return ""


def _is_number_like(v: str) -> bool:
    v = _clean_csv_cell(v)
    if v == "":
        return False
    try:
        float(v)
        return True
    except Exception:
        return False


def _is_number_or_empty(v: str) -> bool:
    v = _clean_csv_cell(v)
    return v == "" or _is_number_like(v)


def _is_text_non_numeric(v: str) -> bool:
    v = _clean_csv_cell(v)
    return v != "" and not _is_number_like(v)


def _extract_skill_block(cells: List[str], start_idx: int) -> List[float]:
    skills: List[float] = []
    for i in range(len(sett.PLAYER_SKILLS)):
        raw = _safe_csv_cell(cells, start_idx + i)
        if raw == "":
            skills.append(-1.0)
            continue
        try:
            skills.append(float(raw))
        except Exception:
            skills.append(-1.0)
    return skills


def _should_shift_skills_left_by_2(cells: List[str], header_idx: Dict[str, int]) -> bool:
    """
    Detect ONLY the known corruption mode:
    the whole 34-skill block is shifted by 2 cells to the left.

    We require strong evidence so normal rows with missing trailing GK skills
    are NOT "fixed" accidentally.
    """
    crossing_idx = header_idx["crossing"]
    gk_positioning_idx = header_idx["gk_positioning"]
    gk_reflexes_idx = header_idx["gk_reflexes"]
    play_styles_idx = header_idx["play_styles"]

    # The two cells immediately before the nominal skill block.
    # In shifted rows these often contain the displaced first two skill values:
    # crossing, finishing.
    pre_1 = _safe_csv_cell(cells, crossing_idx - 2)
    pre_2 = _safe_csv_cell(cells, crossing_idx - 1)

    # Tail of the nominal skill block.
    # In shifted rows, textual play_styles often leaks into one of these.
    gk_pos = _safe_csv_cell(cells, gk_positioning_idx)
    gk_ref = _safe_csv_cell(cells, gk_reflexes_idx)

    # What would become play_styles after correcting by -2
    corrected_play_styles = _safe_csv_cell(cells, play_styles_idx - 2)

    # Condition 1:
    # both cells before crossing look like displaced numeric skill values (or empty).
    pre_block_looks_like_skills = _is_number_or_empty(pre_1) and _is_number_or_empty(pre_2)

    # Condition 2:
    # strong sign that textual play_styles leaked into the skill block.
    text_leaked_into_skill_tail = _is_text_non_numeric(gk_pos) or _is_text_non_numeric(gk_ref)

    # Condition 3:
    # after shifting, the play_styles cell should be text-like or empty.
    corrected_play_styles_looks_valid = corrected_play_styles == "" or _is_text_non_numeric(corrected_play_styles)

    return pre_block_looks_like_skills and text_leaked_into_skill_tail and corrected_play_styles_looks_valid


def load_sofifa_players(*, rebuild: bool = False, debug_shifts: bool = False) -> None:
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
        g.sofifa_snapshots = []
        g.sofifa_player_occurrences = {}
        g.sofifa_players_by_dob = {}

        g.sofifa_team_meta = {}
        g.sofifa_players_by_team = {}
        g.sofifa_teams_by_league = {}
        g.fs_team_to_sofifa_team = {}
    else:
        if getattr(g, "sofifa_snapshots", None):
            if len(g.sofifa_snapshots) > 0:
                print(f"[sofifa] Using {len(g.sofifa_snapshots)} snapshots from cache; not reloading CSV.")
                return

    snapshots = _list_csv_snapshots(Path(sett.SOFIFA_CSV_DIR))
    if not snapshots:
        print(f"[sofifa] No CSV files found in {sett.SOFIFA_CSV_DIR}")
        return

    required_headers = [
        "player_id",
        "name",
        "full_name",
        "dob",
        "club_id",
        "club_name",
        "club_league_id",
        "club_league_name",
        *sett.PLAYER_SKILLS,
        "play_styles",
    ]

    for snap_idx, (snap_date, path) in enumerate(snapshots):
        players_by_id: Dict[int, dict] = {}

        total_rows = 0
        num_skipped = 0
        num_loaded = 0
        num_missing_cells = 0
        num_shifted_left_2 = 0

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)

            try:
                header = next(reader)
            except StopIteration:
                print(f"[sofifa] Empty CSV: {path.name}")
                continue

            header = [_clean_csv_cell(h) for h in header]
            header_idx = {col: i for i, col in enumerate(header)}

            missing_headers = [h for h in required_headers if h not in header_idx]
            if missing_headers:
                print(f"[sofifa] Skipping {path.name}: missing headers {missing_headers}")
                continue

            crossing_idx = header_idx["crossing"]

            for row_num, row in enumerate(reader, start=2):
                row = [_clean_csv_cell(x) for x in row]
                total_rows += 1

                sofifa_id_raw = _safe_csv_cell(row, header_idx["player_id"])
                name = _safe_csv_cell(row, header_idx["name"])
                full_name = _safe_csv_cell(row, header_idx["full_name"])
                dob = _parse_date_flexible(_safe_csv_cell(row, header_idx["dob"]))

                club_id_raw = _safe_csv_cell(row, header_idx["club_id"])
                club_name = _safe_csv_cell(row, header_idx["club_name"]) or None
                club_league_id_raw = _safe_csv_cell(row, header_idx["club_league_id"])
                club_league_name = _safe_csv_cell(row, header_idx["club_league_name"]) or None

                if not sofifa_id_raw or not name or not full_name or dob is None:
                    num_skipped += 1
                    continue

                try:
                    sofifa_id = int(sofifa_id_raw)
                except Exception:
                    num_skipped += 1
                    continue

                try:
                    club_id = int(club_id_raw) if club_id_raw else None
                except Exception:
                    club_id = None

                try:
                    club_league_id = int(club_league_id_raw) if club_league_id_raw else None
                except Exception:
                    club_league_id = None

                name_norm = name.split(" - FIFA")[0].split(" -")[0].strip()

                shifted = _should_shift_skills_left_by_2(row, header_idx)
                skill_start_idx = crossing_idx - 2 if shifted else crossing_idx
                skills = _extract_skill_block(row, skill_start_idx)

                if shifted:
                    num_shifted_left_2 += 1
                    if debug_shifts:
                        print(
                            f"[sofifa][warn] {path.name}:{row_num} player_id={sofifa_id} "
                            f"shifted by 2 to the left (name={name_norm})"
                        )

                num_missing_cells += sum(1 for x in skills if x == -1.0)

                record = {
                    "sofifa_id": sofifa_id,
                    "name": name_norm,
                    "full_name": full_name,
                    "dob": dob,
                    "skills": skills,
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

        total_cells = max(1, total_rows * len(sett.PLAYER_SKILLS))
        print(
            f"[sofifa] Loaded {num_loaded} players from {path.name} "
            f"(skipped {num_skipped}/{max(1, total_rows)} ({num_skipped/max(1, total_rows):.2%}), "
            f"missing skill cells: {num_missing_cells}/{total_cells} "
            f"({num_missing_cells/total_cells:.2%}), "
            f"shifted-left-by-2 fixed rows: {num_shifted_left_2})."
        )

    g.sofifa_player_occurrences = dict(sorted(g.sofifa_player_occurrences.items()))
    g.sofifa_players_by_dob = dict(sorted(g.sofifa_players_by_dob.items(), key=lambda x: x[0]))
