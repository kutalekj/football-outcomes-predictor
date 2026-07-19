from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from football_outcomes.data.fs_models import (
    FSDataBundle,
)

SNAPSHOT_VERSION = 1


def load_snapshot(
    path: Path,
) -> FSDataBundle:
    """
    Load and validate one explicitly selected
    snapshot.
    """

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
    path: Path,
) -> Optional[FSDataBundle]:
    """
    Attempt to load one explicitly selected
    snapshot.
    """

    if not path.exists():
        return None

    try:
        return load_snapshot(path)
    except Exception as error:
        print("Warning: failed to load " f"snapshot ({error}). " "Rebuilding from API…")
        return None


def save_snapshot(
    bundle: FSDataBundle,
    path: Path,
) -> None:
    """
    Store a bundle at one explicitly selected path.
    """

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
