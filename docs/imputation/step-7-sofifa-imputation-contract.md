# Step 7 leakage-safe SoFIFA imputation contract

## Goal

Step 7 introduces a temporary and removable process for producing
complete team-strength tensors without using information that would
not have been available at prediction time.

The frozen Step 6 snapshot remains unchanged.

## Step 6 baseline

The selected model-development scope contains:

- 30,469 selected matches;
- 30,468 array-ready matches;
- 60,938 match-team strength matrices;
- 22,790,064 raw strength cells;
- 22,584,899 observed strength cells;
- 205,165 missing strength cells;
- 6,005 fully missing player rows;
- 231 partially missing player rows;
- 2 absent strength matrices;
- 6 validly shaped but fully missing matrices;
- 306 unmatched unique lineup players.

All 509 selected teams are mapped to SoFIFA teams.

## Existing temporal limitation

The legacy SoFIFA skill lookup prioritizes snapshots dated on or before
the match, but it may fall back to a snapshot dated after the match.

Persisted strength values therefore remain the reproducibility baseline,
but they are not treated as proof of strict temporal safety.

Step 7 introduces a separate past-only reconstruction path.

## Temporal policy

For a match at time T:

1. no SoFIFA snapshot dated after T may be used;
2. candidate snapshots are ordered from the nearest past snapshot to
   older past snapshots;
3. each observed skill value records the snapshot date from which it
   was obtained;
4. missing skills may be merged from multiple past snapshots within an
   explicit maximum age;
5. validation or test matches never contribute values or statistics to
   the imputer fitted for their prediction;
6. target values are never used for imputation.

Future fallback is disabled in the leakage-safe path.

## Missing-value definition

A raw skill cell is missing when it is:

- absent;
- `None`;
- non-finite;
- numerically less than zero.

Observed SoFIFA skills remain on the original 0–100 scale until the
existing array-normalization boundary.

## Per-cell provenance

Every reconstructed skill cell receives exactly one provenance code:

- `0`: unresolved missing value;
- `1`: nearest available past SoFIFA snapshot;
- `2`: older past SoFIFA snapshot used to complete a missing skill;
- `3`: competition-and-position median fitted on training matches;
- `4`: position median fitted on training matches;
- `5`: global per-skill median fitted on training matches;
- `6`: fixed neutral fallback.

The provenance matrix has the same shape as the strength matrix.

## Statistical fallback hierarchy

Statistical imputation is fitted only on the rolling training window.

For each missing cell, the fallback order is:

1. competition-and-position median for the skill, when sufficient
   training observations exist;
2. position median for the skill;
3. global training median for the skill;
4. fixed neutral value `50.0`.

The minimum support for grouped medians is explicit configuration and
must be reported.

Medians are used because they are deterministic and less sensitive to
outlying player ratings than arithmetic means.

## Position handling

The existing deterministic lineup-position array is used.

Fully missing or padded rows may receive fallback values, but their
provenance remains non-observed. The model must therefore be able to
distinguish real SoFIFA observations from generated values.

## Model-input compatibility

Step 7 does not change the v1 or v2 architecture by default.

When imputation is enabled:

- imputed values replace raw missing values;
- the existing observed-value mask remains `1` only for genuine
  past-only SoFIFA observations;
- imputed cells retain observed mask `0`;
- the full provenance matrix remains available for reporting and later
  architecture experiments.

When imputation is disabled, array construction must remain byte-for-byte
equivalent to the Step 6 behavior.

## Mutation policy

The implementation must not:

- modify the frozen snapshot;
- modify serialized match or feature objects in place;
- overwrite persisted strength matrices;
- write imputed values back to the snapshot cache.

Reconstructed and imputed arrays are transient derived data.

## Required comparisons

Step 7 reporting must compare:

1. persisted legacy strength coverage;
2. reconstructed past-only strength coverage;
3. past-only plus statistical imputation coverage.

This separates temporal-policy losses from ordinary missing-data losses.

## Acceptance criteria

Step 7 is complete when:

1. future-dated SoFIFA snapshots are impossible in the new path;
2. per-cell temporal and imputation provenance is deterministic;
3. observed past-only values remain unchanged by statistical imputation;
4. every imputed validation value is based only on its rolling training
   window;
5. imputation-disabled behavior remains unchanged;
6. no snapshot object is mutated;
7. all validation, unit and characterization tests pass;
8. a deterministic full-dataset comparison report is produced.

## Step 7.2 past-only skill retrieval

A pure temporal retrieval service is implemented in
`football_outcomes.data.sofifa_temporal`.

The service:

- accepts snapshots and player occurrences explicitly;
- never reads global state or mutable settings;
- uses the actual date attached to each snapshot as the temporal
  authority;
- rejects every snapshot dated after the prediction date;
- orders eligible snapshots from nearest past to older past;
- merges valid skills at cell level;
- records one source date and provenance code per skill;
- distinguishes nearest-past observations, older-past observations and
  unresolved cells;
- returns immutable results;
- does not modify the legacy SoFIFA retrieval path or the frozen
  snapshot.

The complete provenance code range `0` through `6` is frozen here for
later statistical-imputation increments.

## Step 7.3 past-only team-strength reconstruction

A transient team-strength reconstruction service is implemented in
`football_outcomes.data.sofifa_strength`.

For every selected match-team side, the service produces aligned:

- `11 x 34` raw skill values;
- `11 x 34` provenance codes;
- `11 x 34` source-age values in days;
- 11 FootyStats player IDs;
- 11 matched SoFIFA player IDs or unresolved markers;
- 11 deterministic position indices.

Row ordering is delegated to the existing lineup service so the skill,
provenance, source-age and position arrays share the same goalkeeper,
defender, midfielder, forward and padding alignment.

Only successful cached FootyStats-to-SoFIFA player mappings are used.
Failed, absent or malformed mappings remain unresolved.

Source age is `-1` for unresolved cells. Every observed cell has a
non-negative source age bounded by the configured past-only window.

The service reconstructs both home and away matrices without modifying
matches, persisted features, lineups, snapshot records, occurrence
indexes or player-match cache records.

## Step 7.4 fold-local statistical fallback

A pure statistical imputation service is implemented in
`football_outcomes.data.sofifa_imputation`.

The service is separated into explicit fit and apply operations.

Fitting accepts only past-only reconstructed training matrices.
Statistically imputed values are rejected as fitting observations, which
prevents recursively learning from generated values.

Missing cells are resolved in this deterministic order:

1. competition-and-position per-skill median;
2. position per-skill median;
3. global training-window per-skill median;
4. fixed neutral value `50.0`.

Competition-and-position and position medians require the configured
minimum support. Global medians require at least one observed training
value.

Observed past-only values, provenance codes and source ages remain
unchanged. Their observed mask remains `1`.

Every statistically imputed cell:

- retains observed mask `0`;
- retains source age `-1`;
- receives provenance code `3`, `4`, `5` or `6`;
- is not written back to the frozen snapshot.

The fitter receives its training samples explicitly and has no access to
validation matches, targets, global state or mutable settings.
