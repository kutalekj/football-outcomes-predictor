# Step 6 full-dataset validation contract

## Scope

Step 6 validates the frozen full multi-league FootyStats and SoFIFA
snapshot without modifying its contents.

The validation process must be deterministic and offline.

## Snapshot identity

Record:

- snapshot filename;
- SHA-256 digest;
- file size;
- snapshot format version;
- collection counts;
- snapshot metadata keys.

Exact counts identify the frozen validation input. They are not general
requirements for future refreshed snapshots.

## Critical failures

Validation must fail for:

- incompatible snapshot format;
- duplicate primary IDs;
- broken required object relationships;
- invalid match teams, goals, dates, seasons, or competition links;
- unknown selected competitions;
- unusable target values;
- invalid model-input dimensions;
- non-finite model-input values after documented conversion;
- chronology or round-construction violations;
- nondeterministic validation results.

## Reported quality metrics

Validation must report, but not initially fail solely because of:

- missing lineups;
- incomplete player positions;
- failed SoFIFA mappings;
- missing SoFIFA skills;
- incomplete strength matrices before masking;
- competition and season imbalance;
- small rolling validation rounds.

Thresholds for these quality metrics will be chosen only after their
full-dataset distributions are measured.

## Required artifacts

The final validator must produce:

- one JSON summary;
- detailed CSV tables where appropriate;
- one concise Markdown report;
- a nonzero process exit code for critical failures.

## Step 6 acceptance

Step 6 is complete when:

1. the full snapshot passes all critical checks;
2. all quality metrics are explicitly quantified;
3. repeated runs produce identical report values;
4. the ordinary test suite and runtime smoke test remain green;
5. the report provides the evidence needed to design Step 7 imputation.

## Frozen validation snapshot

To be filled from the Step 6.1 inventory:

- Filename:
- SHA-256:
- Size in bytes:
- Snapshot version:
- Competition seasons:
- Teams:
- Players:
- Matches:
- SoFIFA snapshots:
- FS-to-SoFIFA cache entries:
