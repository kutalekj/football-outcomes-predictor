import ast
from pathlib import Path

from football_outcomes.evaluation import (
    plots,
)
from football_outcomes.training import (
    train_mlp_rolling,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_plot_module_has_no_training_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "evaluation" / "plots.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ImportFrom,
        )
        and node.module is not None
    }

    assert not any(module.startswith("football_outcomes.training") for module in imported_modules)


def test_legacy_plot_name_is_direct_alias() -> None:
    assert train_mlp_rolling._save_pretrain_round_plot is plots.save_pretrain_round_plot


def test_empty_records_do_not_create_plot(
    tmp_path,
) -> None:
    plots.save_pretrain_round_plot(
        log_dir=tmp_path,
        round_records=[],
        title="Empty",
    )

    assert not (tmp_path / "round_overview.png").exists()


def test_round_overview_plot_is_created(
    tmp_path,
) -> None:
    plots.save_pretrain_round_plot(
        log_dir=tmp_path,
        round_records=[
            {
                "round_idx": 1,
                "val_accuracy": 0.60,
                "val_auc": 0.65,
                "val_brier": 0.24,
                "val_loss": 0.70,
            },
            {
                "round_idx": 2,
                "val_accuracy": 0.70,
                "val_auc": 0.75,
                "val_brier": 0.20,
                "val_loss": 0.60,
            },
        ],
        title="Pretraining test",
    )

    output_path = tmp_path / "round_overview.png"

    assert output_path.exists()
    assert output_path.stat().st_size > 0
