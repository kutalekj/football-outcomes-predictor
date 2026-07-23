from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import (
    asdict,
    dataclass,
    is_dataclass,
)
from datetime import (
    date,
    datetime,
    timezone,
)
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any

SHA256_PATTERN = re.compile(r"^[0-9A-F]{64}$")
RUN_KIND_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class SnapshotIdentity:
    filename: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.filename:
            raise ValueError("Snapshot filename must not " "be empty.")

        if self.size_bytes < 0:
            raise ValueError("Snapshot size must be " "non-negative.")

        _validate_sha256(
            self.sha256,
            name="snapshot SHA-256",
        )


@dataclass(frozen=True)
class GitIdentity:
    commit: str
    branch: str | None
    is_dirty: bool
    dirty_entries: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.commit:
            raise ValueError("Git commit must not be " "empty.")

        if self.is_dirty != bool(self.dirty_entries):
            raise ValueError("Git dirty flag does not " "match dirty entries.")


@dataclass(frozen=True)
class EnvironmentIdentity:
    python_version: str
    python_implementation: str
    platform: str
    operating_system: str
    operating_system_release: str
    machine: str
    numpy_version: str | None
    scikit_learn_version: str | None
    tensorflow_version: str | None
    tensorflow_built_with_cuda: bool | None
    visible_devices: tuple[str, ...]
    tensorflow_runtime_error: str | None


@dataclass(frozen=True)
class ArtifactIdentity:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.relative_path)

        if not self.relative_path or path.is_absolute():
            raise ValueError("Artifact paths must be " "non-empty and relative.")

        if self.size_bytes < 0:
            raise ValueError("Artifact size must be " "non-negative.")

        _validate_sha256(
            self.sha256,
            name="artifact SHA-256",
        )


def _validate_sha256(
    value: str,
    *,
    name: str,
) -> None:
    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must contain 64 " "uppercase hexadecimal " "characters.")


def sha256_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest().upper()


def _normalise_json(
    value: Any,
) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _normalise_json(asdict(value))

    if isinstance(value, Enum):
        return _normalise_json(value.value)

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Mapping):
        keys = list(value)

        if any(not isinstance(key, str) for key in keys):
            raise TypeError("JSON mapping keys must be " "strings.")

        return {key: _normalise_json(value[key]) for key in sorted(keys)}

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [_normalise_json(item) for item in value]

    if isinstance(
        value,
        (
            set,
            frozenset,
        ),
    ):
        items = [_normalise_json(item) for item in value]

        return sorted(
            items,
            key=lambda item: (
                json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            ),
        )

    if value is None or isinstance(
        value,
        (
            str,
            bool,
            int,
        ),
    ):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must be " "finite.")

        return value

    raise TypeError("Unsupported manifest value " f"type: {type(value)!r}.")


def canonical_json_bytes(
    value: Any,
) -> bytes:
    normalised = _normalise_json(value)

    return json.dumps(
        normalised,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_payload_sha256(
    value: Any,
) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def write_canonical_json(
    path: Path,
    payload: Any,
) -> None:
    normalised = _normalise_json(payload)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(f"{path.name}.tmp")

    text = (
        json.dumps(
            normalised,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )

    temporary_path.write_text(
        text,
        encoding="utf-8",
    )
    temporary_path.replace(path)


def collect_snapshot_identity(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> SnapshotIdentity:
    resolved = path.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"Snapshot does not exist: " f"{resolved}")

    digest = sha256_file(resolved)

    if expected_sha256 is not None:
        expected = expected_sha256.upper()
        _validate_sha256(
            expected,
            name="expected snapshot SHA-256",
        )

        if digest != expected:
            raise RuntimeError("Snapshot SHA-256 mismatch: " f"expected {expected}, " f"found {digest}.")

    return SnapshotIdentity(
        filename=resolved.name,
        size_bytes=(resolved.stat().st_size),
        sha256=digest,
    )


def _run_git(
    repository_root: Path,
    *arguments: str,
) -> str:
    command = [
        "git",
        "-C",
        str(repository_root),
        *arguments,
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        raise RuntimeError("Unable to collect Git " "identity.") from error

    return completed.stdout.strip()


def collect_git_identity(
    repository_root: Path,
) -> GitIdentity:
    root = repository_root.resolve()

    commit = _run_git(
        root,
        "rev-parse",
        "HEAD",
    )
    branch_text = _run_git(
        root,
        "branch",
        "--show-current",
    )
    status_text = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    dirty_entries = tuple(sorted(line for line in status_text.splitlines() if line))

    return GitIdentity(
        commit=commit,
        branch=(branch_text or None),
        is_dirty=bool(dirty_entries),
        dirty_entries=dirty_entries,
    )


def _package_version(
    distribution_name: str,
) -> str | None:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_identity() -> EnvironmentIdentity:
    tensorflow_version = _package_version("tensorflow")
    tensorflow_built_with_cuda: bool | None = None
    visible_devices: tuple[
        str,
        ...,
    ] = ()
    tensorflow_runtime_error: str | None = None

    try:
        import tensorflow as tf

        tensorflow_built_with_cuda = bool(tf.test.is_built_with_cuda())
        visible_devices = tuple(sorted(device.name for device in tf.config.list_physical_devices()))
    except Exception as error:
        tensorflow_runtime_error = f"{type(error).__name__}: " f"{error}"

    return EnvironmentIdentity(
        python_version=(platform.python_version()),
        python_implementation=(platform.python_implementation()),
        platform=platform.platform(),
        operating_system=(platform.system()),
        operating_system_release=(platform.release()),
        machine=platform.machine(),
        numpy_version=(_package_version("numpy")),
        scikit_learn_version=(_package_version("scikit-learn")),
        tensorflow_version=(tensorflow_version),
        tensorflow_built_with_cuda=(tensorflow_built_with_cuda),
        visible_devices=visible_devices,
        tensorflow_runtime_error=(tensorflow_runtime_error),
    )


def collect_artifact_identities(
    paths: Sequence[Path],
    *,
    root: Path,
) -> tuple[ArtifactIdentity, ...]:
    resolved_root = root.resolve()
    identities: list[ArtifactIdentity] = []

    for path in paths:
        resolved = path.resolve()

        if not resolved.is_file():
            raise FileNotFoundError("Artifact does not exist: " f"{resolved}")

        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError("Artifact is outside the " "declared output root: " f"{resolved}") from error

        identities.append(
            ArtifactIdentity(
                relative_path=(relative.as_posix()),
                size_bytes=(resolved.stat().st_size),
                sha256=(sha256_file(resolved)),
            )
        )

    identities.sort(key=lambda item: (item.relative_path))

    relative_paths = [item.relative_path for item in identities]

    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Duplicate artifact paths are " "not allowed.")

    return tuple(identities)


def derive_run_id(
    *,
    run_kind: str,
    git_commit: str,
    snapshot_sha256: str,
    seed: int | None,
    configuration: Mapping[str, Any],
) -> str:
    if not RUN_KIND_PATTERN.fullmatch(run_kind):
        raise ValueError("run_kind must use lowercase " "letters, digits and hyphens.")

    _validate_sha256(
        snapshot_sha256,
        name="snapshot SHA-256",
    )

    if isinstance(seed, bool):
        raise TypeError("seed must be an integer or " "None.")

    identity_payload = {
        "schema_version": 1,
        "run_kind": run_kind,
        "git_commit": git_commit,
        "snapshot_sha256": (snapshot_sha256),
        "seed": seed,
        "configuration": (configuration),
    }

    suffix = canonical_payload_sha256(identity_payload)[:16].lower()

    return f"{run_kind}-{suffix}"


def _utc_text(
    value: datetime,
) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("created_at_utc must be " "timezone-aware.")

    utc_value = value.astimezone(timezone.utc)

    return utc_value.isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def build_experiment_manifest(
    *,
    run_kind: str,
    command: Sequence[str],
    created_at_utc: datetime,
    git: GitIdentity,
    snapshot: SnapshotIdentity,
    environment: EnvironmentIdentity,
    seed: int | None,
    configuration: Mapping[str, Any],
    artifacts: Sequence[ArtifactIdentity],
) -> dict[str, Any]:
    if not command or any(not isinstance(argument, str) or not argument for argument in command):
        raise ValueError("command must contain " "non-empty strings.")

    normalised_configuration = _normalise_json(configuration)

    ordered_artifacts = sorted(
        artifacts,
        key=lambda item: (item.relative_path),
    )

    artifact_paths = [artifact.relative_path for artifact in ordered_artifacts]

    if len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError("Duplicate artifact paths are " "not allowed.")

    run_id = derive_run_id(
        run_kind=run_kind,
        git_commit=git.commit,
        snapshot_sha256=(snapshot.sha256),
        seed=seed,
        configuration=(normalised_configuration),
    )

    artifact_payload = [asdict(artifact) for artifact in ordered_artifacts]

    return {
        "schema_version": 1,
        "run_id": run_id,
        "run_kind": run_kind,
        "created_at_utc": _utc_text(created_at_utc),
        "command": list(command),
        "git": asdict(git),
        "snapshot": asdict(snapshot),
        "environment": asdict(environment),
        "seed": seed,
        "target": {
            "name": "under_2_5_goals",
            "positive_class": 1,
            "positive_class_meaning": ("total goals below 2.5"),
            "negative_class": 0,
            "negative_class_meaning": ("total goals at least 2.5"),
            "prediction_field": ("probability_under_2_5"),
        },
        "configuration": (normalised_configuration),
        "configuration_sha256": (canonical_payload_sha256(normalised_configuration)),
        "artifacts": artifact_payload,
        "artifact_index_sha256": (canonical_payload_sha256(artifact_payload)),
    }


def write_experiment_manifest(
    path: Path,
    manifest: Mapping[str, Any],
) -> None:
    write_canonical_json(
        path,
        manifest,
    )
