from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from football_outcomes.config.models import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    validate_experiment_config,
)


def make_valid_config() -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            source="offline_snapshot",
            load_snapshot_path=Path("example-snapshot.pkl"),
            allow_network=False,
        )
    )


def test_valid_offline_configuration_is_accepted() -> None:
    validate_experiment_config(make_valid_config())


def test_configuration_objects_are_immutable() -> None:
    config = make_valid_config()

    with pytest.raises(FrozenInstanceError):
        config.data.source = "footystats_api"


def test_offline_source_requires_snapshot_path() -> None:
    config = ExperimentConfig(
        data=DataConfig(
            source="offline_snapshot",
        )
    )

    with pytest.raises(
        ValueError,
        match="load_snapshot_path",
    ):
        validate_experiment_config(config)


def test_auxiliary_head_requires_task() -> None:
    config = ExperimentConfig(
        data=DataConfig(load_snapshot_path=Path("example-snapshot.pkl")),
        model=ModelConfig(
            use_team_aux_head=True,
            aux_task=None,
        ),
    )

    with pytest.raises(
        ValueError,
        match="auxiliary head",
    ):
        validate_experiment_config(config)


def test_child_strength_flags_are_allowed_when_branch_is_disabled() -> None:
    config = ExperimentConfig(
        data=DataConfig(load_snapshot_path=Path("example-snapshot.pkl")),
        model=ModelConfig(
            use_team_strength=False,
            use_position_embedding=True,
            use_strength_masks=True,
        ),
    )

    validate_experiment_config(config)
