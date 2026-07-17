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
