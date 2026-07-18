from __future__ import annotations

from tensorflow.keras.models import Model

from football_outcomes.modeling.v1 import (
    build_model_v1,
)
from football_outcomes.modeling.v2 import (
    build_model_v2,
)


def build_model(
    num_num: int,
    num_teams: int,
    num_comps: int,
    cfg,
) -> Model:
    """Build the configured full prediction model."""

    if cfg.model_version == "v1":
        return build_model_v1(
            num_num=num_num,
            num_teams=num_teams,
            num_comps=num_comps,
            cfg=cfg,
        )

    if cfg.model_version == "v2":
        return build_model_v2(
            num_num=num_num,
            num_teams=num_teams,
            num_comps=num_comps,
            cfg=cfg,
        )

    raise ValueError("Unknown model_version: " f"{cfg.model_version}")
