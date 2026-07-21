from __future__ import annotations

import ast
import json
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

import pytest

from football_outcomes.experiments import (
    ArtifactIdentity,
    EnvironmentIdentity,
    GitIdentity,
    SnapshotIdentity,
    build_experiment_manifest,
    canonical_payload_sha256,
    collect_artifact_identities,
    collect_git_identity,
    collect_snapshot_identity,
    derive_run_id,
)
from football_outcomes.experiments import manifest as manifest_module
from football_outcomes.experiments import (
    write_canonical_json,
    write_experiment_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_HASH = "9F86D081884C7D659A2FEAA0C" "55AD015A3BF4F1B2B0B822CD" "15D6C15B0F00A08"


def environment() -> EnvironmentIdentity:
    return EnvironmentIdentity(
        python_version="3.10.8",
        python_implementation="CPython",
        platform="Test Platform",
        operating_system="Test OS",
        operating_system_release="1",
        machine="x86_64",
        numpy_version="1.23.5",
        scikit_learn_version="1.7.2",
        tensorflow_version="2.10.1",
        tensorflow_built_with_cuda=False,
        visible_devices=("/physical_device:CPU:0",),
        tensorflow_runtime_error=None,
    )


def git_identity() -> GitIdentity:
    return GitIdentity(
        commit="a" * 40,
        branch="test-branch",
        is_dirty=False,
        dirty_entries=(),
    )


def snapshot_identity() -> SnapshotIdentity:
    return SnapshotIdentity(
        filename="snapshot.pkl",
        size_bytes=123,
        sha256="B" * 64,
    )


def artifact(
    path: str = "configuration.json",
) -> ArtifactIdentity:
    return ArtifactIdentity(
        relative_path=path,
        size_bytes=10,
        sha256="C" * 64,
    )


def test_manifest_module_has_no_training_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "experiments" / "manifest.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ImportFrom,
        )
        and node.module is not None
    }

    assert not any(module.startswith("football_outcomes.training") for module in imported_modules)


def test_canonical_hash_ignores_mapping_order() -> None:
    first = {
        "b": 2,
        "a": {
            "y": 4,
            "x": 3,
        },
    }
    second = {
        "a": {
            "x": 3,
            "y": 4,
        },
        "b": 2,
    }

    assert canonical_payload_sha256(first) == canonical_payload_sha256(second)


def test_nonfinite_payload_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        canonical_payload_sha256(
            {
                "value": float("nan"),
            }
        )


def test_snapshot_identity_and_expected_hash(
    tmp_path,
) -> None:
    path = tmp_path / "snapshot.pkl"
    path.write_bytes(b"test")

    identity = collect_snapshot_identity(
        path,
        expected_sha256=TEST_HASH,
    )

    assert identity == SnapshotIdentity(
        filename="snapshot.pkl",
        size_bytes=4,
        sha256=TEST_HASH,
    )


def test_snapshot_hash_mismatch_is_rejected(
    tmp_path,
) -> None:
    path = tmp_path / "snapshot.pkl"
    path.write_bytes(b"test")

    with pytest.raises(
        RuntimeError,
        match="mismatch",
    ):
        collect_snapshot_identity(
            path,
            expected_sha256="A" * 64,
        )


def test_artifact_identities_are_relative_and_sorted(
    tmp_path,
) -> None:
    first = tmp_path / "b.csv"
    second = tmp_path / "nested" / "a.json"

    second.parent.mkdir()
    first.write_text(
        "b",
        encoding="utf-8",
    )
    second.write_text(
        "a",
        encoding="utf-8",
    )

    identities = collect_artifact_identities(
        [
            first,
            second,
        ],
        root=tmp_path,
    )

    assert [value.relative_path for value in identities] == [
        "b.csv",
        "nested/a.json",
    ]


def test_artifact_outside_root_is_rejected(
    tmp_path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside.txt"

    root.mkdir()
    outside.write_text(
        "outside",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="outside",
    ):
        collect_artifact_identities(
            [outside],
            root=root,
        )


def test_git_identity_captures_dirty_entries(
    monkeypatch,
    tmp_path,
) -> None:
    responses = {
        (
            "rev-parse",
            "HEAD",
        ): "1"
        * 40,
        (
            "branch",
            "--show-current",
        ): "feature/test",
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ): ("?? z.txt\n" " M a.py\n"),
    }

    def fake_run_git(
        repository_root,
        *arguments,
    ):
        assert repository_root == tmp_path.resolve()
        return responses[arguments]

    monkeypatch.setattr(
        manifest_module,
        "_run_git",
        fake_run_git,
    )

    identity = collect_git_identity(tmp_path)

    assert identity.commit == "1" * 40
    assert identity.branch == ("feature/test")
    assert identity.is_dirty is True
    assert identity.dirty_entries == (
        " M a.py",
        "?? z.txt",
    )


def test_run_id_changes_with_configuration() -> None:
    common = {
        "run_kind": "canary",
        "git_commit": "a" * 40,
        "snapshot_sha256": "B" * 64,
        "seed": 123,
    }

    first = derive_run_id(
        **common,
        configuration={
            "epochs": 1,
        },
    )
    second = derive_run_id(
        **common,
        configuration={
            "epochs": 2,
        },
    )

    assert first.startswith("canary-")
    assert first != second


def test_manifest_is_deterministic_with_fixed_inputs() -> None:
    arguments = {
        "run_kind": "canary",
        "command": (
            "python",
            "run_canary.py",
        ),
        "created_at_utc": datetime(
            2026,
            7,
            21,
            10,
            30,
            tzinfo=timezone.utc,
        ),
        "git": git_identity(),
        "snapshot": (snapshot_identity()),
        "environment": environment(),
        "seed": 123,
        "configuration": {
            "window_rounds": 25,
            "epochs": 1,
        },
    }

    first = build_experiment_manifest(
        **arguments,
        artifacts=[
            artifact("z.csv"),
            artifact("a.json"),
        ],
    )
    second = build_experiment_manifest(
        **arguments,
        artifacts=[
            artifact("a.json"),
            artifact("z.csv"),
        ],
    )

    assert first == second
    assert [item["relative_path"] for item in first["artifacts"]] == [
        "a.json",
        "z.csv",
    ]
    assert first["target"]["positive_class"] == 1
    assert first["target"]["prediction_field"] == "probability_under_2_5"


def test_naive_creation_time_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        build_experiment_manifest(
            run_kind="canary",
            command=("python",),
            created_at_utc=datetime(
                2026,
                7,
                21,
            ),
            git=git_identity(),
            snapshot=(snapshot_identity()),
            environment=environment(),
            seed=123,
            configuration={},
            artifacts=[],
        )


def test_canonical_writers_are_byte_deterministic(
    tmp_path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first_payload = {
        "b": 2,
        "a": 1,
    }
    second_payload = {
        "a": 1,
        "b": 2,
    }

    write_canonical_json(
        first_path,
        first_payload,
    )
    write_experiment_manifest(
        second_path,
        second_payload,
    )

    assert first_path.read_bytes() == second_path.read_bytes()

    loaded = json.loads(first_path.read_text(encoding="utf-8"))
    assert loaded == {
        "a": 1,
        "b": 2,
    }
