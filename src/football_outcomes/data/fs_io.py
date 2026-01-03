from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Optional

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
