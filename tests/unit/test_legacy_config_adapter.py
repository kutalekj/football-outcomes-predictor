from pathlib import Path
from types import SimpleNamespace

import pytest

from football_outcomes.config.legacy import (
    experiment_config_from_legacy_settings,
)
from football_outcomes.config.models import (
    FeatureConfig,
    ModelConfig,
    OutputConfig,
    TargetConfig,
    TrainingConfig,
)


def make_legacy_settings(
    **overrides,
) -> SimpleNamespace:
    values = {
        "ALL_LOAD": True,
        "ALL_GET_NEW": False,
        "ALL_STORE": False,
        "SUBMISSION_MODE": False,
        "LOAD_SNAPSHOT_PATH": Path("full-snapshot.pkl"),
        "SAVE_SNAPSHOT_PATH": Path("saved-snapshot.pkl"),
        "SOFIFA_CSV_DIR": "sofifa",
        "FIRST_SEASON": 2021,
        "LAST_SEASON": 2025,
        "COMPS_LEAGUE": [
            "League A",
            "League B",
        ],
        "EXCLUDED_COMP_SEASONS": {
            ("League B", 2022),
        },
        "DATA_DIR": Path("data"),
    }
    values.update(overrides)

    return SimpleNamespace(**values)


def test_current_settings_are_not_treated_as_legacy() -> None:
    with pytest.raises(
        ValueError,
        match=("enable neither snapshot loading " "nor FootyStats retrieval"),
    ):
        experiment_config_from_legacy_settings()


def test_full_offline_settings_are_translated() -> None:
    settings = make_legacy_settings()

    config = experiment_config_from_legacy_settings(settings)

    assert config.data.source == ("offline_snapshot")
    assert config.data.allow_network is False
    assert config.data.load_snapshot_path == Path("full-snapshot.pkl")
    assert config.data.sofifa_csv_dir == Path("sofifa")

    assert config.selection.competitions == (
        "League A",
        "League B",
    )
    assert config.selection.seasons == (
        2021,
        2022,
        2023,
        2024,
    )
    assert config.selection.exclude_competition_seasons == (("League B", 2022),)


def test_api_retrieval_requires_explicit_network() -> None:
    settings = make_legacy_settings(
        ALL_LOAD=False,
        ALL_GET_NEW=True,
    )

    config = experiment_config_from_legacy_settings(settings)

    assert config.data.source == ("footystats_api")
    assert config.data.allow_network is True
    assert config.data.load_snapshot_path is None


def test_ambiguous_legacy_data_mode_is_rejected() -> None:
    settings = make_legacy_settings(
        ALL_LOAD=True,
        ALL_GET_NEW=True,
    )

    with pytest.raises(
        ValueError,
        match="ALL_LOAD=True",
    ):
        experiment_config_from_legacy_settings(settings)


def test_disabled_legacy_data_modes_are_rejected() -> None:
    settings = make_legacy_settings(
        ALL_LOAD=False,
        ALL_GET_NEW=False,
    )

    with pytest.raises(
        ValueError,
        match="neither snapshot loading",
    ):
        experiment_config_from_legacy_settings(settings)


def test_explicit_config_sections_are_preserved() -> None:
    settings = make_legacy_settings()

    features = FeatureConfig(use_team_strength=False)
    target = TargetConfig(mode="goals_reg")
    model = ModelConfig(version="v1")
    training = TrainingConfig(window_rounds=10)
    output = OutputConfig(root_dir=Path("custom-output"))

    config = experiment_config_from_legacy_settings(
        settings,
        features=features,
        target=target,
        model=model,
        training=training,
        output=output,
    )

    assert config.features is features
    assert config.target is target
    assert config.model is model
    assert config.training is training
    assert config.output is output
