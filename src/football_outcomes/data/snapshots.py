from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import (
    FSDataBundle,
)

SNAPSHOT_VERSION = 1


def load_snapshot(
    path: Path = sett.LOAD_SNAPSHOT_PATH,
) -> FSDataBundle:
    with path.open("rb") as file:
        bundle: FSDataBundle = pickle.load(file)

    version = bundle.meta.get(
        "snapshot_version",
        0,
    )

    if version != SNAPSHOT_VERSION:
        raise RuntimeError("Incompatible snapshot version " f"{version}; expected " f"{SNAPSHOT_VERSION}.")

    return bundle


def try_load_snapshot(
    path: Path = sett.LOAD_SNAPSHOT_PATH,
) -> Optional[FSDataBundle]:
    if not path.exists():
        return None

    try:
        return load_snapshot(path)
    except Exception as error:
        print("Warning: failed to load " f"snapshot ({error}). " "Rebuilding from API…")
        return None


def save_snapshot(
    bundle: FSDataBundle,
    path: Path = sett.SAVE_SNAPSHOT_PATH,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    bundle.meta["snapshot_version"] = SNAPSHOT_VERSION

    print("Saving snapshot to: " f"{path.resolve()}")

    with path.open("wb") as file:
        pickle.dump(
            bundle,
            file,
            protocol=(pickle.HIGHEST_PROTOCOL),
        )

    print("Snapshot saved.")
