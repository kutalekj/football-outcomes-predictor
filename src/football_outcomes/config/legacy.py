from __future__ import annotations

from pathlib import Path
from typing import Any

from football_outcomes.config import fs_settings
from football_outcomes.config.models import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    OutputConfig,
    SelectionConfig,
    TargetConfig,
    TrainingConfig,
    validate_experiment_config,
)


def _optional_path(
    value: str | Path | None,
) -> Path | None:
    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    return Path(value)


def _legacy_data_source(
    settings: Any,
) -> tuple[str, bool]:
    load_existing = bool(getattr(settings, "ALL_LOAD", False))
    retrieve_new = bool(getattr(settings, "ALL_GET_NEW", False))

    if load_existing and retrieve_new:
        raise ValueError(
            "The legacy combination ALL_LOAD=True and "
            "ALL_GET_NEW=True cannot yet be represented "
            "by one explicit data source."
        )

    if load_existing:
        return "offline_snapshot", False

    if retrieve_new:
        return "footystats_api", True

    raise ValueError("The legacy settings enable neither snapshot " "loading nor FootyStats retrieval.")


def experiment_config_from_legacy_settings(
    settings: Any = fs_settings,
    *,
    features: FeatureConfig | None = None,
    target: TargetConfig | None = None,
    model: ModelConfig | None = None,
    training: TrainingConfig | None = None,
    output: OutputConfig | None = None,
) -> ExperimentConfig:
    """Translate active fs_settings values into explicit configuration.

    The legacy settings module contains data, selection, and path
    choices. Model and training choices currently live elsewhere, so
    callers may supply those sections explicitly. Until they do, the
    corresponding new-configuration defaults are used.
    """

    source, allow_network = _legacy_data_source(settings)

    first_season = int(settings.FIRST_SEASON)
    last_season = int(settings.LAST_SEASON)

    if last_season <= first_season:
        raise ValueError("LAST_SEASON must be greater than " "FIRST_SEASON.")

    if bool(getattr(settings, "SUBMISSION_MODE", False)):
        competitions = ("England Premier League",)
    else:
        competitions = tuple(settings.COMPS_LEAGUE)

    excluded_competition_seasons = tuple(
        sorted(
            (
                str(competition),
                int(season),
            )
            for competition, season in (settings.EXCLUDED_COMP_SEASONS)
        )
    )

    load_snapshot_path = None

    if source == "offline_snapshot":
        load_snapshot_path = _optional_path(settings.LOAD_SNAPSHOT_PATH)

    config = ExperimentConfig(
        data=DataConfig(
            source=source,
            load_snapshot_path=load_snapshot_path,
            save_snapshot_path=_optional_path(settings.SAVE_SNAPSHOT_PATH),
            sofifa_csv_dir=_optional_path(settings.SOFIFA_CSV_DIR),
            allow_network=allow_network,
        ),
        selection=SelectionConfig(
            competitions=competitions,
            seasons=tuple(
                range(
                    first_season,
                    last_season,
                )
            ),
            exclude_competition_seasons=(excluded_competition_seasons),
            include_postseason=False,
        ),
        features=features or FeatureConfig(),
        target=target or TargetConfig(),
        model=model or ModelConfig(),
        training=training or TrainingConfig(),
        output=output or OutputConfig(root_dir=Path(settings.DATA_DIR)),
    )

    validate_experiment_config(config)

    return config
