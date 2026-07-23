from __future__ import annotations

from pathlib import Path

import pytest

from football_outcomes.experiments import benchmark
from football_outcomes.experiments.benchmark import (
    BenchmarkConfig,
    choose_benchmark_fold_indices,
    is_full_benchmark_schedule,
    run_neural_benchmark,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_module_reuses_canary_boundary() -> None:
    path = PROJECT_ROOT / "src" / "football_outcomes" / "experiments" / "benchmark.py"
    source = path.read_text(encoding="utf-8")

    assert "run_modeling_canary" in source
    assert "fs_globals" not in source
    assert "Global" not in source
    assert "train_rolling" not in source
    assert "requests" not in source


def test_full_schedule_includes_every_eligible_fold() -> None:
    indices = choose_benchmark_fold_indices(
        round_count=320,
        window_rounds=25,
    )

    assert len(indices) == 295
    assert indices[0] == 25
    assert indices[-1] == 319
    assert is_full_benchmark_schedule(
        indices,
        round_count=320,
        window_rounds=25,
    )


def test_partial_schedule_is_explicit() -> None:
    indices = choose_benchmark_fold_indices(
        round_count=320,
        window_rounds=25,
        start_fold_offset=3,
        fold_count=2,
    )

    assert indices == (28, 29)
    assert not is_full_benchmark_schedule(
        indices,
        round_count=320,
        window_rounds=25,
    )


def test_schedule_rejects_out_of_range_request() -> None:
    with pytest.raises(ValueError, match="exceed"):
        choose_benchmark_fold_indices(
            round_count=30,
            window_rounds=25,
            fold_count=6,
        )


def test_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="fold_count"):
        BenchmarkConfig(fold_count=0)

    with pytest.raises(ValueError, match="start_fold_offset"):
        BenchmarkConfig(start_fold_offset=-1)

    with pytest.raises(ValueError, match="model_version"):
        BenchmarkConfig(model_version="v3")


def test_full_benchmark_uses_benchmark_identity(
    monkeypatch,
    tmp_path,
) -> None:
    captured = {}

    def fake_run_modeling_canary(**kwargs):
        captured.update(kwargs)
        return tmp_path / "benchmark-test"

    monkeypatch.setattr(
        benchmark,
        "run_modeling_canary",
        fake_run_modeling_canary,
    )

    result = run_neural_benchmark(
        repository_root=tmp_path,
        snapshot_path=tmp_path / "snapshot.pkl",
        output_root=tmp_path / "output",
        config=BenchmarkConfig(),
        command=("python", "benchmark"),
    )

    assert result == tmp_path / "benchmark-test"
    assert captured["run_kind"] == "benchmark"
    assert captured["experiment_tier"] == "full-neural-benchmark"
    assert captured["model_name"] == "v2-benchmark"
    assert captured["config"].fold_count == 295
    assert captured["config"].start_fold_offset == 0


def test_partial_benchmark_has_distinct_identity(
    monkeypatch,
    tmp_path,
) -> None:
    captured = {}

    def fake_run_modeling_canary(**kwargs):
        captured.update(kwargs)
        return tmp_path / "benchmark-partial-test"

    monkeypatch.setattr(
        benchmark,
        "run_modeling_canary",
        fake_run_modeling_canary,
    )

    run_neural_benchmark(
        repository_root=tmp_path,
        snapshot_path=tmp_path / "snapshot.pkl",
        output_root=tmp_path / "output",
        config=BenchmarkConfig(fold_count=2),
        command=("python", "benchmark"),
    )

    assert captured["run_kind"] == "benchmark-partial"
    assert captured["experiment_tier"] == "partial-neural-benchmark"
    assert captured["config"].fold_count == 2
