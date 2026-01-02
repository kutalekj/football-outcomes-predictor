from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from football_outcomes.config.fs_settings import LOAD_SNAPSHOT_PATH, SAVE_SNAPSHOT_PATH
from football_outcomes.data.fs_models import FSDataBundle

SNAPSHOT_VERSION = 1


def load_snapshot(path: Path = LOAD_SNAPSHOT_PATH) -> FSDataBundle:
    with path.open("rb") as f:
        bundle: FSDataBundle = pickle.load(f)

    # Simple version check
    version = bundle.meta.get("snapshot_version", 0)
    if version != SNAPSHOT_VERSION:
        raise RuntimeError(f"Incompatible snapshot version {version}; expected {SNAPSHOT_VERSION}.")

    return bundle


def try_load_snapshot(path: Path = LOAD_SNAPSHOT_PATH) -> Optional[FSDataBundle]:
    if not path.exists():
        return None
    try:
        return load_snapshot(path)
    except Exception as e:
        print(f"Warning: failed to load snapshot ({e}). Rebuilding from API…")
        return None


def save_snapshot(bundle: FSDataBundle, path: Path = SAVE_SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle.meta["snapshot_version"] = SNAPSHOT_VERSION
    print(f"Saving snapshot to: {path.resolve()}")

    with path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Snapshot saved.")
