# Step 5 Data-Layer Inventory

## Purpose

Step 5 will remove obsolete and submission-specific code, establish
clear data-layer boundaries, and preserve compatibility with existing
serialized snapshots.

No serialized domain class may be moved or renamed until its pickle
compatibility has been explicitly protected.

## Active modules

| Module | Current responsibilities | Planned treatment |
|---|---|---|
| `data/fs_models.py` | Serialized domain objects, league-table behavior, feature calculation, settings/global access | Preserve legacy pickle paths; separate serialized state from feature and table services incrementally |
| `data/fs_io.py` | Snapshot pickle I/O, snapshot versioning, average-strength CSV loading, SoFIFA CSV parsing | Split snapshot persistence from SoFIFA ingestion |
| `data/fs_retrieve.py` | FootyStats HTTP access, pagination, parsing, entity construction, lineup processing, throttling, global mutation | Isolate transport, parsers and state assembly; keep live redesign for Step 10 |
| `config/fs_globals.py` | Mutable process-wide dataset and matching state | Introduce explicit state/container boundaries and gradually remove active singleton dependence |
| `utils/fs_player_skill_utils.py` | SoFIFA indexes, team matching, player matching, snapshot selection, lineup normalization, strength construction | Split matching, lineup and strength responsibilities |
| `utils/fs_common.py` | Dataset linking, filtering, valid-round handling, season annotation, table initialization | Move domain-specific preparation into explicit dataset/application services |
| `config/fs_settings.py` | Static constants, runtime data-source flags, paths, submission behavior, matching policy | Retain true constants; migrate runtime choices to explicit configuration |
| `scripts/main_footystats.py` | Data loading, global initialization, filtering, feature calculation, model selection and training | Convert into a thin application entry point |
| `training/fs_training_utils.py` | Legacy compatibility wrappers | Remove after all callers and historical requirements are classified |
| `training/fs_classical_baselines.py` | Active classical baseline orchestration | Keep active; migrate only remaining legacy dependencies |

## Safety constraints

1. Existing pickle files must remain loadable throughout Step 5.
2. The historical module path `football_outcomes.data.fs_models`
   remains available until an explicit compatibility migration exists.
3. Default unit and characterization tests perform no network requests.
4. New data functions receive paths, configuration and state explicitly.
5. New code must not introduce additional dependencies on `Global`.
6. Runtime source selection must not depend on `SUBMISSION_MODE`.
7. The full offline snapshot remains the canonical data source during
   Steps 5 through 9.
8. Live FootyStats retrieval is isolated during Step 5 but redesigned
   only during Step 10.

## Planned Step 5 sequence

1. Characterize snapshot and data-layer contracts.
2. Split snapshot persistence from SoFIFA CSV ingestion.
3. Introduce an explicit dataset-state boundary around `Global`.
4. Separate serialized domain objects from feature/table services.
5. Split SoFIFA matching and team-strength responsibilities.
6. Isolate FootyStats transport and response parsing.
7. Move application orchestration out of `main_footystats.py`.
8. Remove submission-specific and confirmed dead modules.
9. Retire compatibility wrappers that have no remaining callers.
10. Run the complete offline and snapshot compatibility validation.
