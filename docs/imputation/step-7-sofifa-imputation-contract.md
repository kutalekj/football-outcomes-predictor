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
