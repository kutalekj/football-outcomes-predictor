### Explanation of parameters

- `count` (both for `delta_days_*` and `snapshots_used_*`) is basically how many player-skill retrieval events were logged in that (league, 4-month) bucket. It tracks how many matches there were + how many lineups were present, not snapshot quality.

- `delta_days_mean/std` describe how far (in days) from the match date the snapshots that contributed skills were, on average, and how spread out that distance is.

- `snapshots_used_mean/std` describe how many distinct snapshots were needed to assemble a full skills vector - the best proxy for "snapshot completeness / missingness pressure":
	- near 1.0 -> usually one snapshot is enough (good coverage / low missingness)
	- higher values -> often had to "patch" skills from multiple snapshots (more missingness / more instability / more mixing past+future)

TLDR: `delta_days_*` is about time alignment; `snapshots_used_*` is about completeness / patching