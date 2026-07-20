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
- invalid teams, goals, dates, seasons, or competition links among
  matches selected for model development;
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
- small rolling validation rounds;
- incomplete raw match team references outside the selected model scope;
- detached reverse match references retained by competition-season objects;
- duplicate or cross-team lineup player references;
- stale auxiliary team-roster references.

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

- Filename: `fs_full_26-01-26_NO-25-26-MATCHES_v5.pkl`
- SHA-256: `AEC8C575156346AB1C255433C3D1E92E8782A5A04666149B929EE336FE27A51C`
- Size in bytes: `700083593`
- Snapshot version: `1`
- Competition seasons: `224`
- Teams: `2995`
- Players: `63805`
- Matches: `43110`
- SoFIFA snapshots: `29`
- FS-to-SoFIFA cache entries: `20401`
- Metadata keys: `snapshot_version`

## Step 6.2 raw-domain findings

The authoritative top-level collections passed their primary-ID and
canonical-reference checks.

The raw snapshot also contains quality findings that are retained and
reported without modifying the snapshot:

- 6,797 detached competition-season match references;
- 919 raw matches with a missing away-team reference;
- 443 duplicate home-lineup player occurrences;
- 308 duplicate away-lineup player occurrences;
- 326 player occurrences present in both lineups;
- 18 roster player references absent from the canonical player index.

These are warnings at the raw-snapshot boundary. Step 6.3 revalidates
the selected model-development match scope and treats missing teams,
invalid chronology, and other unusable selected-match fields as
critical failures.
