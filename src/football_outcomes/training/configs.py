from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    mode: str = "binary_u25"
    window_rounds: int = 25
    epochs_per_step: int = 5
    learning_rate: float = 0.0001
    batch_size: int = 64

    team_emb_dim: int = 8
    comp_emb_dim: int = 5
    strength_emb_dim: int = 16
    position_emb_dim: int = 3

    max_goals_class: int = 10
    seed: int | None = 42

    model_version: str = "v2"
    use_team_aux_head: bool = False
    aux_task: str | None = None
    aux_weight: float = 0.15

    num_branch_dim: int = 48
    cat_branch_dim: int = 32
    team_branch_dim: int = 32
    player_row_hidden_dim: int = 32
    role_post_hidden_dim: int = 32
    fusion_hidden_dim_1: int = 64
    fusion_hidden_dim_2: int = 32

    tabular_dropout: float = 0.20
    cat_dropout: float = 0.15
    team_dropout: float = 0.25
    fusion_dropout_1: float = 0.45
    fusion_dropout_2: float = 0.30

    num_l2: float = 1e-5
    cat_l2: float = 1e-5
    team_l2: float = 5e-5
    fusion_l2: float = 5e-5

    early_stopping_patience: int = 1
    early_stopping_min_delta: float = 0.0
    freeze_pretrained_branch_rounds: int = 0

    run_name: str | None = None
    min_warning_val_size: int = 20
    save_oos_predictions: bool = True

    enable_branch_diagnostics: bool = True
    probe_matches: int = 32
    use_team_strength: bool = True
    use_team_ids: bool = True
    use_comp_embedding: bool = True
    use_position_embedding: bool = True

    representation: str = "full"
    use_strength_masks: bool = True

    mlp_hidden_1: int = 128
    mlp_hidden_2: int = 64
    mlp_hidden_3: int = 32
    mlp_dropout_1: float = 0.50
    mlp_dropout_2: float = 0.40

    lr_schedule: str = "constant"
    lr_decay_rate: float = 0.997
    min_learning_rate: float = 2e-5


@dataclass
class StrengthPretrainConfig:
    branch_version: str = "v1"
    mode: str = "binary_u25"

    window_rounds: int = 25
    epochs_per_step: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 64

    max_goals_class: int = 10
    seed: int | None = 42
    run_name: str | None = None

    early_stopping_patience: int = 1
    early_stopping_min_delta: float = 0.0

    strength_emb_dim: int = 16
    position_emb_dim: int = 3

    player_row_hidden_dim: int = 32
    role_post_hidden_dim: int = 32
    team_branch_dim: int = 32
    team_dropout: float = 0.25
    team_l2: float = 5e-5

    compare_hidden_dim: int = 32
    compare_dropout: float = 0.20

    save_oos_predictions: bool = True

    representation: str = "full"
    use_strength_masks: bool = True
    use_position_embedding: bool = True
