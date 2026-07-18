from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def save_pretrain_round_plot(
    log_dir: str | Path,
    round_records: Sequence[dict[str, Any]],
    title: str,
) -> None:
    """Save the structured-pretraining round overview."""

    if not round_records:
        return

    round_indices = [record["round_idx"] for record in round_records]

    figure = plt.figure(figsize=(12, 8))

    if "val_accuracy" in round_records[0]:
        accuracy_axis = figure.add_subplot(
            2,
            2,
            1,
        )
        accuracy_axis.plot(
            round_indices,
            [record["val_accuracy"] for record in round_records],
        )
        accuracy_axis.set_title("Round val accuracy")

    if "val_auc" in round_records[0]:
        auc_axis = figure.add_subplot(
            2,
            2,
            2,
        )
        auc_axis.plot(
            round_indices,
            [record["val_auc"] for record in round_records],
        )
        auc_axis.set_title("Round val AUC")

    if "val_brier" in round_records[0]:
        brier_axis = figure.add_subplot(
            2,
            2,
            3,
        )
        brier_axis.plot(
            round_indices,
            [record["val_brier"] for record in round_records],
        )
        brier_axis.set_title("Round val Brier")

    if "val_loss" in round_records[0]:
        loss_axis = figure.add_subplot(
            2,
            2,
            4,
        )
        loss_axis.plot(
            round_indices,
            [record["val_loss"] for record in round_records],
        )
        loss_axis.set_title("Round val loss")

    figure.suptitle(title)
    figure.tight_layout()

    output_path = Path(log_dir) / "round_overview.png"
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)
