import ast
from dataclasses import asdict
from pathlib import Path

from football_outcomes.training import (
    train_mlp_rolling,
)
from football_outcomes.training.configs import (
    StrengthPretrainConfig,
    TrainConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_config_module_has_no_rolling_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "training" / "configs.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "football_outcomes.training." "train_mlp_rolling" not in imported_modules


def test_legacy_config_names_are_direct_exports() -> None:
    assert train_mlp_rolling.TrainConfig is TrainConfig
    assert train_mlp_rolling.StrengthPretrainConfig is StrengthPretrainConfig


def test_train_config_defaults_are_preserved() -> None:
    config = TrainConfig()

    assert asdict(config) == {
        "mode": "binary_u25",
        "window_rounds": 25,
        "epochs_per_step": 5,
        "learning_rate": 0.0001,
        "batch_size": 64,
        "team_emb_dim": 8,
        "comp_emb_dim": 5,
        "strength_emb_dim": 16,
        "position_emb_dim": 3,
        "max_goals_class": 10,
        "seed": 42,
        "model_version": "v2",
        "use_team_aux_head": False,
        "aux_task": None,
        "aux_weight": 0.15,
        "num_branch_dim": 48,
        "cat_branch_dim": 32,
        "team_branch_dim": 32,
        "player_row_hidden_dim": 32,
        "role_post_hidden_dim": 32,
        "fusion_hidden_dim_1": 64,
        "fusion_hidden_dim_2": 32,
        "tabular_dropout": 0.20,
        "cat_dropout": 0.15,
        "team_dropout": 0.25,
        "fusion_dropout_1": 0.45,
        "fusion_dropout_2": 0.30,
        "num_l2": 1e-5,
        "cat_l2": 1e-5,
        "team_l2": 5e-5,
        "fusion_l2": 5e-5,
        "early_stopping_patience": 1,
        "early_stopping_min_delta": 0.0,
        "freeze_pretrained_branch_rounds": 0,
        "run_name": None,
        "min_warning_val_size": 20,
        "save_oos_predictions": True,
        "enable_branch_diagnostics": True,
        "probe_matches": 32,
        "use_team_strength": True,
        "use_team_ids": True,
        "use_comp_embedding": True,
        "use_position_embedding": True,
        "representation": "full",
        "use_strength_masks": True,
        "enable_strength_imputation": False,
        "strength_imputation_minimum_support": 20,
        "strength_imputation_neutral_value": 50.0,
        "mlp_hidden_1": 128,
        "mlp_hidden_2": 64,
        "mlp_hidden_3": 32,
        "mlp_dropout_1": 0.50,
        "mlp_dropout_2": 0.40,
        "lr_schedule": "constant",
        "lr_decay_rate": 0.997,
        "min_learning_rate": 2e-5,
    }

    assert config.enable_strength_imputation is False
    assert config.strength_imputation_minimum_support == 20
    assert config.strength_imputation_neutral_value == 50.0


def test_strength_pretrain_defaults_are_preserved() -> None:
    config = StrengthPretrainConfig()

    assert config.branch_version == "v1"
    assert config.mode == "binary_u25"
    assert config.window_rounds == 25
    assert config.epochs_per_step == 3
    assert config.learning_rate == 5e-5
    assert config.strength_emb_dim == 16
    assert config.player_row_hidden_dim == 32
    assert config.compare_hidden_dim == 32
    assert config.representation == "full"
    assert config.use_strength_masks is True
    assert config.use_position_embedding is True
