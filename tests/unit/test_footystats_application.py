from pathlib import Path
from types import SimpleNamespace

from football_outcomes.application import (
    footystats_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_config(
    tmp_path,
):
    return footystats_pipeline.FootyStatsPipelineConfig(
        snapshot_path=(tmp_path / "snapshot.pkl"),
        allow_network=False,
        save_snapshot_path=None,
        competitions=("Test League",),
        first_season=2021,
        last_season_exclusive=2025,
        excluded_competition_seasons=(
            frozenset(
                {
                    (
                        "Test League",
                        2022,
                    )
                }
            )
        ),
        valid_round_ids_by_season={
            (
                "Test League",
                2021,
            ): {10},
            (
                "Test League",
                2023,
            ): {30},
        },
        validate_round_ids=True,
        rebuild_derived_state=False,
        log_dir=tmp_path / "logs",
        summary_path=(tmp_path / "dataset_summary.json"),
        run_name="test-run",
    )


def test_selection_is_explicit(
    tmp_path,
) -> None:
    config = make_config(tmp_path)

    matches = [
        SimpleNamespace(
            id=1,
            comp_name="Test League",
            season=2021,
            round_id=10,
        ),
        SimpleNamespace(
            id=2,
            comp_name="Test League",
            season=2021,
            round_id=99,
        ),
        SimpleNamespace(
            id=3,
            comp_name="Test League",
            season=2022,
            round_id=20,
        ),
        SimpleNamespace(
            id=4,
            comp_name="Other League",
            season=2021,
            round_id=10,
        ),
        SimpleNamespace(
            id=5,
            comp_name="Test League",
            season=2023,
            round_id=30,
        ),
    ]

    selected = footystats_pipeline.select_clean_league_matches(
        matches,
        config,
    )

    assert [match.id for match in selected] == [1, 5]


def test_offline_load_uses_explicit_path(
    monkeypatch,
    tmp_path,
) -> None:
    config = make_config(tmp_path)
    config.snapshot_path.write_bytes(b"placeholder")

    bundle = object()
    applied = []

    monkeypatch.setattr(
        footystats_pipeline,
        "load_snapshot",
        lambda path: bundle,
    )
    monkeypatch.setattr(
        footystats_pipeline,
        "retrieve_new_data",
        lambda: ((_ for _ in ()).throw(AssertionError("Network retrieval " "must not run."))),
    )

    target = SimpleNamespace(
        all_matches=[],
        all_teams={},
        all_players={},
        sofifa_snapshots=[],
    )

    def fake_apply(
        supplied_bundle,
    ):
        applied.append(supplied_bundle)
        return target

    monkeypatch.setattr(
        footystats_pipeline,
        "apply_bundle_to_global",
        fake_apply,
    )

    footystats_pipeline.load_data_into_globals(config)

    assert applied == [bundle]


def test_main_script_is_thin() -> None:
    source_path = PROJECT_ROOT / "scripts" / "main_footystats.py"
    source = source_path.read_text(encoding="utf-8")

    assert "run_pipeline" in source
    assert "default_pipeline_config" in source

    assert "fs_settings" not in source
    assert "fs_globals" not in source
    assert "fs_common" not in source
    assert "fs_feature_utils" not in source
    assert "SUBMISSION_MODE" not in source


def test_application_has_no_submission_mode() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "application" / "footystats_pipeline.py"
    source = source_path.read_text(encoding="utf-8")

    assert "SUBMISSION_MODE" not in source
    assert "submission_outputs" not in source
    assert "submission_epl" not in source
