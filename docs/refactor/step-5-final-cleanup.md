# Step 5 final legacy cleanup

This document records the final Step 5 compatibility-path inventory and removal decisions.

## Preserved compatibility boundaries

- `football_outcomes.data.fs_models` remains because serialized snapshots depend on its module and
  class paths.
- `football_outcomes.config.fs_globals` remains temporarily because extracted SoFIFA and
  match-preparation services still use the singleton adapter.
- Snapshot loading remains offline-first and requires an explicit path.

```
scripts/main_apifootball.py:13:from football_outcomes.data.match import Match
scripts/main_apifootball.py:29:if not settings.ALL_LOAD:
scripts/main_apifootball.py:127:if settings.ALL_STORE:
scripts/main_apifootball_app.py:18:from football_outcomes.data.match import Match
src/football_outcomes/data/fs_models.py:332:        from football_outcomes.data.match_features import calculate_match_features as build_match_features
src/football_outcomes/data/io.py:11:from football_outcomes.data.match import Match
src/football_outcomes/data/io.py:13:# from football_outcomes.features import features_utils as feature_ut
src/football_outcomes/data/io_mega.py:10:from football_outcomes.data.match import Match
src/football_outcomes/data/match.py:20:from football_outcomes.features import features_utils as feature_ut
```

## Removed compatibility paths

The following unreferenced FootyStats compatibility paths were removed:

- `src/football_outcomes/data/fs_io.py`
- `src/football_outcomes/utils/fs_player_skill_utils.py`
- `src/football_outcomes/config/legacy.py`
- `tests/unit/test_legacy_config_adapter.py`
- `scripts/tools/create_submission_epl_snapshot.py`
- `fill_globals_with_cache` from `data/fs_retrieve.py`

Tracked analysis and thesis tools now:

- resolve snapshot paths explicitly;
- restore bundles through `apply_bundle_to_global`;
- import SoFIFA team matching from its extracted service;
- avoid the retired FootyStats runtime switches.

Compatibility-only alias tests were removed while behavioral and
architectural boundary tests were retained.

## Deferred legacy API-Football subsystem

The following files are preserved as one internally connected legacy
API-Football subsystem:

- `scripts/main_apifootball.py`
- `scripts/main_apifootball_app.py`
- `src/football_outcomes/config/settings.py`
- `src/football_outcomes/data/match.py`
- `src/football_outcomes/data/io.py`
- `src/football_outcomes/data/io_mega.py`
- `src/football_outcomes/features/features_utils.py`

`data.match` still has tracked callers and imports
`features.features_utils`. Removing or relocating these files would be a
separate migration unrelated to the active FootyStats data-layer cleanup.

The subsystem is therefore retained unchanged and excluded from the
Step 5 removal criteria.
