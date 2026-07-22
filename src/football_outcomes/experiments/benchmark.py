from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from football_outcomes.experiments.canary import (
    EXPECTED_ROUND_COUNT,
    CanaryConfig,
    run_modeling_canary,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    window_rounds: int = 25
    fold_count: int | None = None
    start_fold_offset: int = 0
    epochs_per_fold: int = 1
    batch_size: int = 64
    learning_rate: float = 0.0001
    seed: int = 123
    minimum_group_support: int = 20
    neutral_value: float = 50.0
    model_version: str = "v2"

    def __post_init__(self) -> None:
        for name in (
            "window_rounds",
            "epochs_per_fold",
            "batch_size",
            "minimum_group_support",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if self.fold_count is not None and (type(self.fold_count) is not int or self.fold_count <= 0):
            raise ValueError("fold_count must be a positive integer or None.")

        if type(self.start_fold_offset) is not int or self.start_fold_offset < 0:
            raise ValueError("start_fold_offset must be a non-negative integer.")

        if type(self.seed) is not int:
            raise ValueError("seed must be an integer.")

        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive finite number.")

        if not math.isfinite(self.neutral_value) or self.neutral_value < 0.0 or self.neutral_value > 100.0:
            raise ValueError("neutral_value must be finite and in [0, 100].")

        if self.model_version not in {"v1", "v2"}:
            raise ValueError("model_version must be 'v1' or 'v2'.")


def choose_benchmark_fold_indices(
    *,
    round_count: int,
    window_rounds: int,
    start_fold_offset: int = 0,
    fold_count: int | None = None,
) -> tuple[int, ...]:
    for name, value in (
        ("round_count", round_count),
        ("window_rounds", window_rounds),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if type(start_fold_offset) is not int or start_fold_offset < 0:
        raise ValueError("start_fold_offset must be a non-negative integer.")

    first = window_rounds + start_fold_offset
    available = round_count - first

    if available <= 0:
        raise ValueError("No chronological benchmark folds are available.")

    if fold_count is None:
        resolved_count = available
    else:
        if type(fold_count) is not int or fold_count <= 0:
            raise ValueError("fold_count must be a positive integer or None.")
        resolved_count = fold_count

    stop = first + resolved_count
    if stop > round_count:
        raise ValueError("Requested benchmark folds exceed the available rounds.")

    return tuple(range(first, stop))


def is_full_benchmark_schedule(
    indices: Sequence[int],
    *,
    round_count: int,
    window_rounds: int,
) -> bool:
    expected = tuple(range(window_rounds, round_count))
    return tuple(indices) == expected


def run_neural_benchmark(
    *,
    repository_root: Path,
    snapshot_path: Path,
    output_root: Path,
    config: BenchmarkConfig,
    command: Sequence[str],
    overwrite: bool = False,
) -> Path:
    fold_indices = choose_benchmark_fold_indices(
        round_count=EXPECTED_ROUND_COUNT,
        window_rounds=config.window_rounds,
        start_fold_offset=config.start_fold_offset,
        fold_count=config.fold_count,
    )
    full_schedule = is_full_benchmark_schedule(
        fold_indices,
        round_count=EXPECTED_ROUND_COUNT,
        window_rounds=config.window_rounds,
    )

    run_kind = "benchmark" if full_schedule else "benchmark-partial"
    experiment_tier = "full-neural-benchmark" if full_schedule else "partial-neural-benchmark"
    summary_title = "Step 8 full neural benchmark" if full_schedule else "Step 8 partial neural benchmark"

    canary_config = CanaryConfig(
        window_rounds=config.window_rounds,
        fold_count=len(fold_indices),
        start_fold_offset=config.start_fold_offset,
        epochs_per_fold=config.epochs_per_fold,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
        minimum_group_support=config.minimum_group_support,
        neutral_value=config.neutral_value,
        model_version=config.model_version,
    )

    return run_modeling_canary(
        repository_root=repository_root,
        snapshot_path=snapshot_path,
        output_root=output_root,
        config=canary_config,
        command=command,
        overwrite=overwrite,
        run_kind=run_kind,
        experiment_tier=experiment_tier,
        model_name=f"{config.model_version}-benchmark",
        summary_title=summary_title,
    )
