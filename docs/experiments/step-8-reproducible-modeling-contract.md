# Step 8 first reproducible modeling milestone contract

## Goal

Step 8 produces the first trusted end-to-end modeling benchmark after
the repository restructuring and data validation work.

The milestone must execute the complete path from the frozen snapshot to
chronological out-of-sample predictions through one documented command.

The milestone establishes a reproducible benchmark. It is not a broad
hyperparameter search and does not claim that the current neural model
is optimal.

## Frozen input

The benchmark uses only the validated frozen snapshot:

- filename:
  `fs_full_26-01-26_NO-25-26-MATCHES_v5.pkl`;
- SHA-256:
  `AEC8C575156346AB1C255433C3D1E92E8782A5A04666149B929EE336FE27A51C`;
- selected matches: 30,469;
- array-ready matches: 30,468;
- competition-seasons: 92;
- constructed chronological rounds: 320.

A benchmark command must fail before model construction when the
snapshot hash differs from the declared value.

The snapshot remains immutable.

## Prediction target

The milestone predicts whether the final match goal total is under or
over 2.5 goals.

The binary target semantics are:

- `1`: under 2.5 goals;
- `0`: over 2.5 goals.

Every prediction artifact must state these semantics explicitly.

## Evaluation unit

The authoritative evaluation unit is one out-of-sample match prediction.

Each prediction row must contain enough information to identify:

- experiment run;
- validation fold or round;
- match ID;
- match datetime;
- competition;
- season;
- true binary target;
- predicted probability of the positive class;
- model or baseline name.

A match may appear at most once in the authoritative out-of-sample
prediction artifact for a single experiment variant.

## Chronological protocol

The benchmark uses the repository's deterministic chronological round
construction.

For a validation round R:

1. the training window consists only of the configured rounds preceding
   R;
2. the validation set consists only of round R;
3. no validation target contributes to training, preprocessing,
   imputation, calibration or threshold selection;
4. no match from a later round contributes to the fold;
5. the final eligible round is included.

The initial benchmark uses a 25-round training window.

The model-state policy must be explicit in the experiment manifest. The
first neural benchmark uses the existing rolling carry-forward behavior
unless a later contract amendment declares a reset-per-fold experiment.

## Leakage-safe strength handling

When team-strength imputation is enabled:

1. SoFIFA values are reconstructed from snapshots dated no later than
   each match;
2. the statistical imputer is fitted only on the rolling training
   window;
3. the fitted imputer is applied to both training and validation
   matrices;
4. genuine past-only values retain observed mask `1`;
5. statistically generated values retain observed mask `0`;
6. no reconstructed or imputed value is written to the frozen snapshot.

The Step 7 provenance and temporal contracts remain authoritative.

## Experiment tiers

Step 8 contains three experiment tiers.

### Canary experiment

The canary exercises a small number of chronological folds through the
complete production path.

Its purpose is to verify:

- configuration loading;
- snapshot identity;
- match selection;
- categorical mappings;
- past-only reconstruction;
- fold-local imputation;
- array construction;
- model construction and compilation;
- optimization;
- prediction;
- metric calculation;
- artifact persistence.

Canary results are functional evidence, not final performance evidence.

### Baseline experiment

All baselines use exactly the same validation rows as the neural
benchmark.

The initial required baselines are:

1. training-window positive-class prevalence;
2. training-window majority class;
3. a deterministic classical linear classifier using the declared
   tabular feature subset.

A baseline must not read validation targets while fitting.

### Neural benchmark

The first neural benchmark uses the selected v2 binary model with
leakage-safe strength imputation enabled.

The complete model and training configuration must be stored in the
manifest. No configuration value may depend on validation performance
from the same benchmark run.

## Metrics

The primary model-ranking metric is pooled out-of-sample ROC AUC.

Required secondary metrics are:

- accuracy at threshold 0.5;
- Brier score;
- binary log loss;
- positive-class prevalence;
- prediction count.

Required reporting scopes are:

- pooled across all out-of-sample predictions;
- per validation fold;
- per competition;
- per season;
- per competition-season.

Calibration summaries and probability-distribution diagnostics are also
required in the final report.

Metrics for all model variants and baselines must be calculated from the
same authoritative prediction rows.

## Reproducibility identity

Every experiment receives an immutable manifest containing at least:

- schema version;
- run ID;
- command and arguments;
- UTC creation time;
- Git commit;
- Git dirty-state indicator;
- snapshot filename, size and SHA-256;
- Python version;
- operating-system information;
- NumPy version;
- scikit-learn version;
- TensorFlow version;
- TensorFlow CUDA-build status;
- visible devices;
- random seed;
- match-selection configuration;
- round and window configuration;
- categorical-map identity;
- imputation configuration;
- model configuration;
- optimizer and training configuration;
- output artifact names and hashes.

An experiment with a dirty Git working tree must be clearly marked. The
final Milestone 1 benchmark must use a clean committed revision.

## Reproducibility levels

Deterministic non-model artifacts must be byte-identical when generated
twice with the same inputs. These include:

- selected match IDs;
- round membership;
- targets;
- categorical maps;
- experiment configuration;
- report schemas.

On the same machine and software environment, repeated canary runs must
satisfy:

- identical fold membership;
- identical target arrays;
- identical prediction-row ordering;
- maximum absolute prediction difference no greater than `1e-6`;
- absolute aggregate-metric difference no greater than `1e-6`.

The manifest must expose environment differences rather than claiming
cross-platform byte identity for TensorFlow optimization.

## Required artifacts

Each completed experiment produces:

1. `manifest.json`;
2. `folds.csv`;
3. `predictions.csv`;
4. `fold_metrics.csv`;
5. `aggregate_metrics.json`;
6. `configuration.json`;
7. `runtime.json`;
8. a human-readable Markdown summary.

The final milestone additionally produces a tracked report under
`docs/experiments/results`.

Large model files, TensorBoard logs and temporary training artifacts are
not committed unless explicitly selected for long-term archival.

Tracked reports must contain hashes for any referenced untracked
artifacts.

## Failure policy

An experiment command must exit non-zero when:

- the snapshot identity is incorrect;
- selected or array-ready scope differs from the declared contract;
- a fold violates chronology;
- a validation match appears more than once;
- a target or prediction is missing or non-finite;
- a prediction lies outside `[0, 1]`;
- strength imputation leaves unresolved model values;
- required artifacts cannot be written;
- manifest and artifact identities disagree.

Warnings may record small validation folds, class imbalance or weak model
performance, but warnings do not override structural failures.

## Performance interpretation

The milestone is accepted even when the neural model does not outperform
a baseline.

A weak result must be reported honestly and becomes the starting point
for later architecture and feature experiments.

No model-selection claim may be based on the same validation predictions
used for the final benchmark comparison.

## Non-goals

Step 8 does not:

- perform broad hyperparameter optimization;
- tune a probabilitfrom football_outcomes.experiments.manifest import (
  ArtifactIdentity,
  EnvironmentIdentity,
  GitIdentity,
  SnapshotIdentity,
  build_experiment_manifest,
  canonical_payload_sha256,
  collect_artifact_identities,
  collect_environment_identity,
  collect_git_identity,
  collect_snapshot_identity,
  derive_run_id,
  write_canonical_json,
  write_experiment_manifest,
  )

__all__ = [
"ArtifactIdentity",
"EnvironmentIdentity",
"GitIdentity",
"SnapshotIdentity",
"build_experiment_manifest",
"canonical_payload_sha256",
"collect_artifact_identities",
"collect_environment_identity",
"collect_git_identity",
"collect_snapshot_identity",
"derive_run_id",
"write_canonical_json",
"write_experiment_manifest",
]y threshold on final validation predictions;

- modify the frozen snapshot;
- rewrite legacy historical experiment results;
- introduce GPU-specific requirements;
- restore retired data-source implementations;
- claim statistical significance from a single benchmark.

## Step 8 substeps

1. define the milestone and reproducibility contract;
2. implement experiment manifests and artifact identity;
3. implement and run the chronological canary;
4. implement common-fold baselines;
5. execute the first full neural benchmark;
6. aggregate metrics, calibration and comparisons;
7. validate, report, merge and tag Milestone 1.

## Step 8.1 acceptance criteria

Step 8.1 is complete when:

1. the branch starts from the completed Step 7 merge;
2. two independent Step 7 validator executions produce byte-identical
   JSON, CSV and Markdown artifacts;
3. the determinism evidence is stored as tracked JSON;
4. the frozen snapshot hash is recorded;
5. target semantics and chronological boundaries are explicit;
6. required experiment artifacts and metrics are defined;
7. reproducibility tolerances are explicit;
8. weak model performance is distinguished from pipeline failure;
9. documentation checks pass;
10. the Step 8.1 files are committed and pushed.

## Step 8.2 experiment identity and manifest

Experiment identity is implemented in
`football_outcomes.experiments.manifest`.

The module is independent of training and model construction. It
provides deterministic primitives for:

- canonical JSON normalization;
- canonical payload hashing;
- atomic canonical JSON writing;
- frozen snapshot identity and expected-hash validation;
- Git commit, branch and dirty-state capture;
- Python, operating-system and package environment capture;
- relative artifact path, size and SHA-256 capture;
- deterministic run-ID derivation;
- final experiment-manifest construction.

A run ID is derived from:

- run kind;
- Git commit;
- frozen snapshot SHA-256;
- random seed;
- normalized experiment configuration.

UTC creation time and runtime-environment information are recorded in
the manifest but do not alter the deterministic run ID.

Artifact paths are relative to the declared experiment-output root.
Artifacts outside that root and duplicate artifact paths are rejected.

The manifest records the binary target semantics explicitly:

- positive class `1`: total goals below 2.5;
- negative class `0`: total goals at least 2.5;
- prediction field: `probability_under_2_5`.

The manifest itself is not included in its artifact index, avoiding a
self-referential hash. Every other completed experiment artifact will be
hashed before the manifest is written.

The Git dirty-state record includes staged, modified and untracked
entries. A dirty run remains inspectable, but the final Milestone 1
benchmark must be executed from a clean committed revision.

Step 8.2 does not modify the current rolling trainer or its historical
TensorBoard output behavior. The Step 8.3 canary runner will become the
first consumer of this manifest API.
## Step 8.3 chronological modeling canary

The first manifest-backed modeling runner is implemented in
`football_outcomes.experiments.canary` and exposed through
`scripts/tools/run_modeling_canary.py`.

The default canary uses:

- the validated frozen snapshot and required SHA-256;
- the validated 30,469 selected-match scope;
- the 30,468 array-ready matches;
- the repository's deterministic 320 chronological rounds;
- a 25-round training window;
- the first two eligible validation rounds;
- one optimization epoch per fold;
- the v2 binary model;
- leakage-safe past-only strength reconstruction;
- fold-local statistical imputation;
- carry-forward model state across consecutive canary folds;
- deterministic seed `123`;
- disabled branch diagnostics and TensorBoard output.

For each canary fold, the runner:

1. slices only the preceding training rounds;
2. validates the training/validation datetime boundary;
3. reconstructs and completes strength arrays through the Step 7 adapter;
4. fits the strength imputer only on that fold's training matches;
5. builds the model once and carries its state into the next consecutive
   canary fold;
6. trains without batch shuffling;
7. predicts only the current validation round;
8. verifies finite binary targets and probabilities in `[0, 1]`;
9. rejects duplicate validation match IDs.

Every canary run writes a deterministic run directory containing:

- `configuration.json`;
- `folds.csv`;
- `predictions.csv`;
- `fold_metrics.csv`;
- `aggregate_metrics.json`;
- `runtime.json`;
- `summary.md`;
- `manifest.json`.

The manifest hashes every artifact except itself. The run ID is derived
from the Git commit, frozen snapshot identity, seed and normalized
configuration. Runtime timestamps and durations are recorded but do not
alter the run ID.

The canary is accepted as functional evidence when the complete command
exits zero, the manifest reports the expected snapshot and scope, all
prediction rows are unique and valid, and every required artifact is
present. Canary performance is not used for model selection.


## Step 8.4 common-fold baselines

The first benchmark baselines are implemented in
`football_outcomes.experiments.baselines` and executed through
`scripts/tools/run_common_fold_baselines.py`.

A baseline run consumes a completed manifest-backed canary or neural
benchmark run as its reference. Before fitting, it verifies the
reference manifest hashes for:

- `configuration.json`;
- `folds.csv`;
- `predictions.csv`.

The frozen snapshot SHA-256 must match the reference run. The baseline
runner reconstructs the validated selected scope and chronological
rounds, then requires exact ordered agreement with every reference
validation match ID and target.

Three binary baselines are produced for each reference fold:

1. `training-prevalence`: every validation probability equals the
   positive-class prevalence of the fold's training window;
2. `training-majority`: every validation probability is the hard
   majority class of the training window, with ties assigned to the
   positive class;
3. `logistic-regression`: a standardized deterministic logistic model
   fitted only on the numerical pre-match feature array.

The logistic baseline excludes team IDs, competition IDs, SoFIFA
strength tensors, masks and player-position arrays. It uses solver
`liblinear`, explicit random state, no class weighting and no validation
information during fitting. A single-class training window falls back to
that training class and records the fallback status.

All three variants write prediction rows in the same schema as the
reference neural run. For each model, the ordered tuple of fold, round,
match ID and target must match the reference exactly.

The baseline artifact set is:

- `configuration.json`;
- `folds.csv`;
- `predictions.csv`;
- `fold_metrics.csv`;
- `aggregate_metrics.json`;
- `runtime.json`;
- `summary.md`;
- `manifest.json`.

The manifest records the reference run ID and hashes, the exact
validation-match identity, feature-subset policy and complete logistic
configuration. Baseline outputs remain separate from neural outputs so
later reporting can compare them without rewriting either run.


## Step 8.5 first full neural benchmark

The chronological canary execution boundary is generalized so the same
validated implementation can identify either a small canary or the full
neural benchmark without duplicating model-training logic.

The full benchmark uses:

- all 295 eligible validation folds from round 26 through round 320;
- a 25-round rolling training window;
- one optimization epoch per fold;
- batch size `64`;
- learning rate `0.0001` with a constant schedule;
- random seed `123`;
- the v2 binary architecture;
- leakage-safe past-only SoFIFA reconstruction;
- fold-local strength imputation with minimum grouped support `20`;
- neutral fallback value `50.0`;
- model state carried forward between consecutive folds;
- batch shuffling disabled.

One epoch per fold is the frozen first-milestone configuration. It is a
reproducible reference point, not a claim that the optimization budget is
optimal.

The benchmark includes the first and final eligible validation rounds.
Every validation match appears at most once in the authoritative neural
prediction file.

The default benchmark command omits a fold count and therefore consumes
every eligible fold. Supplying a fold count creates a separately
identified `benchmark-partial` run and must not be presented as the full
Milestone 1 neural result.

The benchmark runner reuses the canary's snapshot, scope, chronology,
imputation, metric, artifact and manifest checks. The run identity is
changed to:

- run kind: `benchmark`;
- experiment tier: `full-neural-benchmark`;
- model name: `v2-benchmark`.

The final full benchmark must be launched from a clean committed Git
revision. A detached clean Git worktree is used so preserved untracked
historical experiment files in the development worktree do not need to be
removed or modified.

The benchmark output remains outside the repository. Step 8.6 will use
its manifest-backed predictions as the reference rows for all three
baseline models and will create the tracked comparison report.

