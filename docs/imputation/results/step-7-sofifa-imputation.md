# Step 7 leakage-safe SoFIFA imputation report

## Result

**PASS**

- Critical observations: `0`
- Warning observations: `1126798`

## Frozen snapshot

- Filename: `fs_full_26-01-26_NO-25-26-MATCHES_v5.pkl`
- SHA-256: `AEC8C575156346AB1C255433C3D1E92E8782A5A04666149B929EE336FE27A51C`
- Size in bytes: `700083593`

## Validation scope

- Selected matches: `30469`
- Array-ready matches: `30468`
- Constructed rounds: `320`
- Strength cells: `22790064`

## Temporal and imputation policy

- Rolling window: `25` rounds
- Audited folds: `5`
- Maximum SoFIFA source age: `120` days
- Maximum snapshots scanned per player: `6`
- Minimum grouped-median support: `20`
- Neutral fallback: `50.0`

## Full selected-scope coverage

| Representation | Observed cells | Missing or unresolved cells | Observation rate |
|---|---:|---:|---:|
| Persisted legacy | 22584899 | 205165 | 99.10% |
| Strict past-only | 22244915 | 545149 | 97.61% |

## Legacy versus strict past-only

- Observed and equal: `22209469`
- Observed but changed: `34392`
- Legacy-only observations: `341038`
- Past-only-only observations: `1054`
- Missing in both paths: `204111`

## Strict temporal evidence

- Nearest-past cells: `21861603`
- Older-past completion cells: `383312`
- Maximum observed source age: `120` days
- Future-source cells: `0`

## Fold-local production-path audit

- Audited validation matches: `349`
- Audited validation cells: `261052`
- Genuine past-only cells: `250854`
- Statistically imputed cells: `10198`
- Cells left unresolved: `0`
- Non-strength array mismatches: `0`

### Audited provenance

- `COMPETITION_POSITION_MEDIAN`: `10130`
- `NEAREST_PAST_SOFIFA`: `230928`
- `OLDER_PAST_SOFIFA`: `19926`
- `POSITION_MEDIAN`: `68`

## Findings

- `legacy_missing_cells`: 205165 — Persisted legacy strength cells are missing.
- `past_only_unresolved_cells`: 545149 — Strict past-only reconstruction cannot resolve all selected-scope cells without imputation.
- `legacy_only_observed_cells`: 341038 — Legacy matrices contain observations that are not available under the strict temporal policy.
- `past_only_only_observed_cells`: 1054 — Past-only reconstruction recovers observations missing from persisted matrices.
- `changed_observed_cells`: 34392 — Legacy and strict past-only paths select different values for cells observed by both paths.

## Acceptance decision

Step 7 is accepted because strict past-only reconstruction used no future information and all audited fold-local arrays satisfied the production input contract.

The persisted snapshot remains unchanged. The new path is transient, explicitly configurable and disabled by default.
