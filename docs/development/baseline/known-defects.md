# Known Legacy Defects Before Refactoring

The following defects are protected by strict expected-failure tests.

## FOP-001 — Main strength tensor is discarded

`build_arrays_for_matches` initializes the strength-input collection
but does not append the home and away strength values and masks.

The returned tensor therefore contains zeros even when valid lineup
strengths are present.

## FOP-002 — Final rolling round is omitted

The MLP trainer, strength-pretraining trainer, and classical-baseline
evaluator use an exclusive loop ending at `len(rounds) - 1`.

The final available round is therefore not evaluated.

## FOP-003 — v2 branch flags are ignored

The v2 model constructs the categorical and structured branches
regardless of the branch-disable settings in `TrainConfig`.

Changing supposedly disabled inputs can therefore change predictions.

## Test policy

Each issue has a `pytest.mark.xfail(strict=True)` test.

While the issue exists, the test must be reported as `XFAIL`.
After the issue is corrected, the marker must be removed in the same
commit as the correction.

## Resolution status

All three recorded defects were corrected during overall Step 2.

Their strict expected-failure markers were removed as each correction
was committed. The tests now remain as ordinary passing regression
tests.
