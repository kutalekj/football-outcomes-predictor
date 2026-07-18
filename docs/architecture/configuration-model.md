# Configuration Model

## Problem with the legacy configuration

The legacy implementation combines several different concerns:

- static competition and position constants;
- local filesystem paths;
- data-loading behavior;
- submission-specific behavior;
- feature and model switches;
- training hyperparameters;
- output and logging choices.

Some runtime choices are evaluated during module import, and
`SUBMISSION_MODE` changes several behaviors indirectly.

The target model replaces hidden global decisions with explicit,
validated configuration objects.

## Root configuration

A complete run will be represented by one aggregate object:

```python
@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    selection: SelectionConfig
    features: FeatureConfig
    target: TargetConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig
```

The dataclasses should be immutable after validation.

### DataConfig

`DataConfig` describes where input data comes from.

Representative fields:

```python
@dataclass(frozen=True)
class DataConfig:
    source: str
    load_snapshot_path: Path | None
    save_snapshot_path: Path | None
    sofifa_csv_dir: Path | None
    allow_network: bool
```

Supported source values should eventually include:

- offline_snapshot;
- footystats_api;
- prepared_submission_snapshot.

Network access must be disabled unless explicitly requested.

### SelectionConfig

`SelectionConfig` determines which matches enter dataset preparation.

Representative fields:

```python
@dataclass(frozen=True)
class SelectionConfig:
    competitions: tuple[str, ...]
    seasons: tuple[int, ...]
    exclude_competition_seasons: tuple[
        tuple[str, int],
        ...,
    ]
    include_postseason: bool
```

Selection rules must not be hidden inside a submission-mode boolean.

### FeatureConfig

`FeatureConfig` controls feature preparation and missing-data handling.

Representative fields:

```python
@dataclass(frozen=True)
class FeatureConfig:
    use_team_strength: bool
    use_strength_masks: bool
    use_player_positions: bool
    missing_strength_value: float
```

Feature availability and model usage are related but different.
Feature preparation may construct strength arrays even when a model
configuration chooses not to use them.

### TargetConfig

`TargetConfig` defines the prediction target.

```python
@dataclass(frozen=True)
class TargetConfig:
    mode: str
    max_goals_class: int
```

Supported modes are initially:

- binary_u25;
- goals_reg;
- goals_dist.

The current `binary_u25` definition remains:

```
1 = total goals is 0, 1, or 2
0 = total goals is 3 or greater
```

### ModelConfig

`ModelConfig` contains architecture choices only.

Representative fields:

```python
@dataclass(frozen=True)
class ModelConfig:
    version: str
    use_team_strength: bool
    use_team_ids: bool
    use_comp_embedding: bool
    use_position_embedding: bool
    use_strength_masks: bool
    use_team_aux_head: bool
    aux_task: str | None
```

Architecture widths, dropout values, embedding dimensions, and
regularization values also belong here.

Training-window and output-path settings do not belong in
`ModelConfig`.

### TrainingConfig

`TrainingConfig` controls optimization and rolling evaluation.

Representative fields:

```python
@dataclass(frozen=True)
class TrainingConfig:
    window_rounds: int
    epochs_per_step: int
    learning_rate: float
    batch_size: int
    seed: int | None
    early_stopping_patience: int
    learning_rate_schedule: str
```

### OutputConfig

`OutputConfig` controls generated artifacts.

```python
@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path
    run_name: str | None
    save_oos_predictions: bool
    enable_tensorboard: bool
    enable_branch_diagnostics: bool
```

Output choices must not influence dataset selection or model
architecture.

## Named profiles

Profiles are explicit starting configurations, not hidden modes.

Initial profiles should be:

### full_offline

- loads a user-selected full local snapshot;
- selects the configured multi-league seasons;
- performs no network retrieval;
- writes normal experiment outputs.

### live_footystats

Reserved for the later retrieval work. This profile will not be
implemented during Steps 3 or 4.

## Legacy submission compatibility

The prepared EPL submission snapshot may temporarily be loaded through
the generic `offline_snapshot` data source for regression or historical
comparison.

It is not a permanent named profile because the thesis submission
workflow is no longer an active application requirement.

The legacy `SUBMISSION_MODE` switch will remain only until its remaining
callers are isolated. It will then be removed with the other
submission-specific code during Step 5.

## Configuration precedence

Configuration values will be resolved in this order, from lowest to
highest priority:

1. dataclass defaults;
2. selected named profile;
3. configuration file;
4. environment variables;
5. command-line arguments.

The final merged configuration is validated before any data is loaded.

## Validation rules

Central validation should reject at least:

- unknown target or model modes;
- a missing snapshot path for offline operation;
- network access without explicit permission;
- an auxiliary head without an auxiliary task;
- non-positive batch size or rolling-window size;
- invalid learning-rate schedule names;
- nonexistent required input directories.

Flags that are irrelevant because a parent branch is disabled may be
accepted but must have no effect.

For example, `use_strength_masks=True` is harmless when
`use_team_strength=False`.

## Migration from the legacy configuration

The migration will be incremental:

1. Keep static constants such as competition names and position indices.
2. Introduce the new dataclasses without changing existing callers.
3. Add a function that translates the current settings into an
`ExperimentConfig`.
4. Change one pipeline function at a time to accept an explicit config.
5. Keep compatibility wrappers while legacy scripts still call the old
signatures.
6. Remove `SUBMISSION_MODE` after the generic offline configuration can
load both the full snapshot and any explicitly selected historical
snapshot without changing global settings.
7. Remove runtime decisions from `fs_settings.py` after all consumers
receive explicit configuration.

## Configuration testing policy

Tests will cover:

- profile defaults;
- precedence between profile, file, environment, and CLI values;
- validation errors;
- path conversion;
- stable serialization to JSON-compatible dictionaries;
- compatibility translation from legacy settings.

Configuration tests must not load the full dataset or initialize a
TensorFlow model.
