from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_DATA_SOURCES = {
    "offline_snapshot",
    "footystats_api",
}

SUPPORTED_TARGET_MODES = {
    "binary_u25",
    "goals_reg",
    "goals_dist",
}

SUPPORTED_MODEL_VERSIONS = {
    "v1",
    "v2",
}

SUPPORTED_LEARNING_RATE_SCHEDULES = {
    "constant",
    "exponential",
    "cosine",
}


@dataclass(frozen=True)
class DataConfig:
    source: str = "offline_snapshot"
    load_snapshot_path: Path | None = None
    save_snapshot_path: Path | None = None
    sofifa_csv_dir: Path | None = None
    allow_network: bool = False


@dataclass(frozen=True)
class SelectionConfig:
    competitions: tuple[str, ...] = ()
    seasons: tuple[int, ...] = ()
    exclude_competition_seasons: tuple[
        tuple[str, int],
        ...,
    ] = ()
    include_postseason: bool = False


@dataclass(frozen=True)
class FeatureConfig:
    use_team_strength: bool = True
    use_strength_masks: bool = True
    use_player_positions: bool = True
    missing_strength_value: float = 0.0


@dataclass(frozen=True)
class TargetConfig:
    mode: str = "binary_u25"
    max_goals_class: int = 10


@dataclass(frozen=True)
class ModelConfig:
    version: str = "v2"

    use_team_strength: bool = True
    use_team_ids: bool = True
    use_comp_embedding: bool = True
    use_position_embedding: bool = True
    use_strength_masks: bool = True

    use_team_aux_head: bool = False
    aux_task: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    window_rounds: int = 25
    epochs_per_step: int = 5
    learning_rate: float = 0.0001
    batch_size: int = 64
    seed: int | None = 42
    early_stopping_patience: int = 1
    learning_rate_schedule: str = "constant"


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path = Path("data")
    run_name: str | None = None
    save_oos_predictions: bool = True
    enable_tensorboard: bool = True
    enable_branch_diagnostics: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def validate_experiment_config(
    config: ExperimentConfig,
) -> None:
    errors: list[str] = []

    if config.data.source not in SUPPORTED_DATA_SOURCES:
        errors.append(f"Unknown data source: {config.data.source}")

    if config.data.source == "offline_snapshot" and config.data.load_snapshot_path is None:
        errors.append("Offline snapshot mode requires " "data.load_snapshot_path.")

    if config.data.source == "footystats_api" and not config.data.allow_network:
        errors.append("FootyStats API mode requires explicit " "network permission.")

    if config.data.source != "footystats_api" and config.data.allow_network:
        errors.append("Network access may only be enabled for " "the FootyStats API source.")

    if config.target.mode not in SUPPORTED_TARGET_MODES:
        errors.append(f"Unknown target mode: {config.target.mode}")

    if config.target.max_goals_class < 1:
        errors.append("target.max_goals_class must be positive.")

    if config.model.version not in SUPPORTED_MODEL_VERSIONS:
        errors.append(f"Unknown model version: {config.model.version}")

    if config.model.use_team_aux_head and config.model.aux_task is None:
        errors.append("An enabled auxiliary head requires " "model.aux_task.")

    if config.model.aux_task is not None and config.model.aux_task not in SUPPORTED_TARGET_MODES:
        errors.append(f"Unknown auxiliary task: " f"{config.model.aux_task}")

    if config.training.window_rounds < 1:
        errors.append("training.window_rounds must be positive.")

    if config.training.epochs_per_step < 1:
        errors.append("training.epochs_per_step must be positive.")

    if config.training.batch_size < 1:
        errors.append("training.batch_size must be positive.")

    if config.training.learning_rate <= 0:
        errors.append("training.learning_rate must be positive.")

    if config.training.learning_rate_schedule not in SUPPORTED_LEARNING_RATE_SCHEDULES:
        errors.append("Unknown learning-rate schedule: " f"{config.training.learning_rate_schedule}")

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError("Invalid experiment configuration:\n" f"{details}")
