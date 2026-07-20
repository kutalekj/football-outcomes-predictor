# Step 6 full-dataset validation report

## Result

**PASS**

- Critical observations: `0`
- Warning observations: `226414`

Warnings quantify affected data entities or cells and are not a count of distinct defect categories.

## Frozen snapshot

- Filename: `fs_full_26-01-26_NO-25-26-MATCHES_v5.pkl`
- SHA-256: `AEC8C575156346AB1C255433C3D1E92E8782A5A04666149B929EE336FE27A51C`
- Size in bytes: `700083593`

## Validation components

| Component | Status | Critical | Warnings |
|---|---:|---:|---:|
| Domain | PASS | 0 | 8811 |
| Selection | PASS | 0 | 0 |
| Readiness | PASS | 0 | 1 |
| Coverage | PASS | 0 | 217602 |

## Model-development scope

- Selected matches: `30469`
- Competitions: `24`
- Competition-seasons: `92`
- Constructed rounds: `320`
- Array-ready matches: `30468`
- Under 2.5 targets: `14879`
- Over 2.5 targets: `15589`

## Coverage summary

- Selected-team mapping: `100.00%`
- Unique-player matching: `98.23%`
- Lineup-reference matching: `99.18%`
- Observed strength cells: `99.10%`

## Finding categories

### Domain

- `detached_comp_season_match_reference`: 6797 (warning)
- `duplicate_away_lineup_player`: 308 (warning)
- `duplicate_home_lineup_player`: 443 (warning)
- `missing_away_team`: 919 (warning)
- `player_in_both_lineups`: 326 (warning)
- `unknown_roster_player`: 18 (warning)

### Readiness

- `missing_persisted_features`: 1 (warning)

### Coverage

- `competition_season_player_match_gap`: 79 (warning)
- `competition_season_strength_gap`: 86 (warning)
- `fully_missing_strength_matrices`: 6 (warning)
- `fully_missing_strength_rows`: 6005 (warning)
- `lineups_without_explicit_goalkeeper`: 203 (warning)
- `missing_strength_cells`: 205165 (warning)
- `missing_strength_matrices`: 2 (warning)
- `noncomplete_lineups`: 8 (warning)
- `partially_missing_strength_rows`: 231 (warning)
- `unmatched_lineup_player_references`: 5511 (warning)
- `unmatched_unique_lineup_players`: 306 (warning)

## Acceptance decision

Step 6 is accepted because all critical validation components passed.

The documented quality gaps remain inputs to Step 7 imputation and are not silently repaired in the snapshot.
