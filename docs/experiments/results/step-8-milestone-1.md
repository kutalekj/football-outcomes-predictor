# Step 8 first reproducible modeling milestone

## Acceptance

**Result: PASS**

The first manifest-backed full chronological benchmark completed on a clean committed revision.

## Scope

- Selected matches: `30,469`
- Array-ready matches: `30,468`
- Evaluated folds: `295`
- Validation rounds: `26` through `320`
- Out-of-sample matches: `28,029`
- Training window: `25` preceding rounds
- Model-state policy: `carry-forward`

## Pooled results

| Model | ROC AUC | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| logistic-regression | 0.558414 | 0.542367 | 0.252669 | 0.702557 | 0.057550 |
| v2-benchmark | 0.545390 | 0.530879 | 0.253065 | 0.701272 | 0.044219 |
| training-prevalence | 0.511609 | 0.516715 | 0.249824 | 0.692796 | 0.004979 |
| training-majority | 0.510773 | 0.516715 | 0.483285 | 7.789637 | 0.483285 |

The deterministic logistic-regression baseline achieved the strongest pooled ROC AUC. The neural benchmark exceeded the prevalence and majority baselines in ranking and threshold accuracy, but it did not outperform logistic regression.

The training-prevalence baseline achieved the lowest Brier score, log loss, and expected calibration error. This indicates that the current neural probability outputs require further calibration and modeling work.

Weak or mixed performance does not invalidate the milestone. Step 8 establishes a trusted, leakage-safe, reproducible benchmark against which later changes can be measured.

## Source identities

- Neural run: `benchmark-8eec79bacd3c9754`
- Baseline run: `baselines-889d2555756949cc`
- Comparison run: `comparison-24d9ee7428d5fb2e`
- Neural Git commit: `396b6698048985b42542b4f33fa217ca4c2835c0`
- Comparison Git commit: `8ae166276bc2582635f5b9378a5a85921182f827`
- Snapshot SHA-256: `AEC8C575156346AB1C255433C3D1E92E8782A5A04666149B929EE336FE27A51C`

## Reporting coverage

- Fold groups: `295`
- Competitions: `24`
- Seasons: `4`
- Competition-seasons: `92`

## Verification

- All source manifest artifact hashes matched.
- All comparison source hashes matched.
- Black, isort, and flake8 passed.
- 295 unit and characterization tests passed.
- v1 and v2 runtime smoke tests passed.

## Tracked detailed evidence

- `step-8-milestone-1.json`
- `step-8-comparison.json`
- `step-8-pooled-metrics.csv`
- `step-8-calibration.csv`
- `step-8-scope-metrics.csv`
