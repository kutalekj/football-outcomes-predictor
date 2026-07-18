from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    Callback,
)
from tensorflow.keras.models import Model


class LayerDriftLogger(Callback):
    """
    Logs L2 drift from initialization for selected layers.
    Skips layers that are not present in the current model variant.
    """

    def __init__(self, layer_names: List[str], writer, every_epoch: bool = True):
        super().__init__()
        self.layer_names = layer_names
        self.writer = writer
        self.every_epoch = every_epoch
        self._initial = {}
        self._present_layer_names = []

    def on_train_begin(self, logs=None):
        self._present_layer_names = []
        for name in self.layer_names:
            try:
                layer = self.model.get_layer(name)
            except ValueError:
                continue
            self._initial[name] = [w.numpy().copy() for w in layer.weights]
            self._present_layer_names.append(name)

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for name in self._present_layer_names:
                layer = self.model.get_layer(name)
                init_ws = self._initial[name]
                curr_ws = layer.weights
                sq = 0.0
                for w0, w1 in zip(init_ws, curr_ws):
                    diff = w1.numpy() - w0
                    sq += float(np.sum(diff * diff))
                drift = float(np.sqrt(sq))
                tf.summary.scalar(f"diag_drift/{name}", drift, step=epoch + 1)
            self.writer.flush()


class BranchProbeLogger(Callback):
    """
    Logs activation variance on a fixed probe batch to detect dead branches.
    Skips layers that are not present in the current model variant.
    """

    def __init__(self, probe_inputs, writer, layer_names: List[str]):
        super().__init__()
        self.probe_inputs = probe_inputs
        self.writer = writer
        self.layer_names = layer_names
        self._submodels = {}

    def on_train_begin(self, logs=None):
        self._submodels = {}
        for name in self.layer_names:
            try:
                layer = self.model.get_layer(name)
            except ValueError:
                continue
            self._submodels[name] = Model(self.model.inputs, layer.output)

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for name, sm in self._submodels.items():
                out = sm.predict(self.probe_inputs, verbose=0)
                tf.summary.scalar(f"diag_probe_meanabs/{name}", float(np.mean(np.abs(out))), step=epoch + 1)
                tf.summary.scalar(f"diag_probe_std/{name}", float(np.std(out)), step=epoch + 1)
            self.writer.flush()


class BranchDiagnosticsCsvLogger(Callback):
    """
    Writes diagnostics to CSV so figures can be regenerated as vector plots.

    One row is written per layer and epoch.
    Drift rows store value=drift.
    Probe rows store value=mean(abs(activation)) and probe_std=std(activation).
    """

    def __init__(
        self,
        csv_path: Path,
        drift_layer_names: List[str],
        probe_layer_names: List[str],
        probe_inputs,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.drift_layer_names = drift_layer_names
        self.probe_layer_names = probe_layer_names
        self.probe_inputs = probe_inputs

        self._initial = {}
        self._present_drift_names = []
        self._submodels = {}

        self._global_step = 0
        self._round_idx = None
        self._train_size = None
        self._val_size = None

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "diag_step",
                    "round_idx",
                    "train_size",
                    "val_size",
                    "local_epoch",
                    "metric_family",
                    "layer",
                    "value",
                    "probe_std",
                ],
            )
            writer.writeheader()

    def set_round_context(
        self,
        round_idx: int,
        train_size: int,
        val_size: int,
        learning_rate: float | None = None,
    ) -> None:
        self._round_idx = round_idx
        self._train_size = train_size
        self._val_size = val_size

    def on_train_begin(self, logs=None):
        self._initial = {}
        self._present_drift_names = []

        for name in self.drift_layer_names:
            try:
                layer = self.model.get_layer(name)
            except ValueError:
                continue

            self._initial[name] = [w.numpy().copy() for w in layer.weights]
            self._present_drift_names.append(name)

        self._submodels = {}
        for name in self.probe_layer_names:
            try:
                layer = self.model.get_layer(name)
            except ValueError:
                continue

            self._submodels[name] = Model(self.model.inputs, layer.output)

    def on_epoch_end(self, epoch, logs=None):
        self._global_step += 1
        rows = []

        for name in self._present_drift_names:
            layer = self.model.get_layer(name)
            init_ws = self._initial[name]
            curr_ws = layer.weights

            sq = 0.0
            for w0, w1 in zip(init_ws, curr_ws):
                diff = w1.numpy() - w0
                sq += float(np.sum(diff * diff))

            rows.append(
                {
                    "diag_step": self._global_step,
                    "round_idx": self._round_idx,
                    "train_size": self._train_size,
                    "val_size": self._val_size,
                    "local_epoch": epoch + 1,
                    "metric_family": "drift",
                    "layer": name,
                    "value": float(np.sqrt(sq)),
                    "probe_std": "",
                }
            )

        for name, sm in self._submodels.items():
            out = sm.predict(self.probe_inputs, verbose=0)

            rows.append(
                {
                    "diag_step": self._global_step,
                    "round_idx": self._round_idx,
                    "train_size": self._train_size,
                    "val_size": self._val_size,
                    "local_epoch": epoch + 1,
                    "metric_family": "probe_meanabs",
                    "layer": name,
                    "value": float(np.mean(np.abs(out))),
                    "probe_std": float(np.std(out)),
                }
            )

        if rows:
            with self.csv_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writerows(rows)


class EpochMetricsCsvLogger(Callback):
    """
    Writes Keras per-epoch fit() metrics to CSV.

    This is intended for plots that need train/validation epoch-level
    accuracy and loss curves, not only round-level validation metrics.
    """

    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = Path(csv_path)
        self._global_epoch_step = 0
        self._round_idx = None
        self._train_size = None
        self._val_size = None
        self._learning_rate = None

        self.fieldnames = [
            "global_epoch_step",
            "round_idx",
            "train_size",
            "val_size",
            "local_epoch",
            "learning_rate",
            "loss",
            "accuracy",
            "auc",
            "mae",
            "val_loss",
            "val_accuracy",
            "val_auc",
            "val_mae",
            "output_main_loss",
            "output_main_accuracy",
            "output_main_auc",
            "output_main_mae",
            "val_output_main_loss",
            "val_output_main_accuracy",
            "val_output_main_auc",
            "val_output_main_mae",
        ]

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def set_round_context(
        self,
        round_idx: int,
        train_size: int,
        val_size: int,
        learning_rate: float | None = None,
    ) -> None:
        self._round_idx = round_idx
        self._train_size = train_size
        self._val_size = val_size
        self._learning_rate = learning_rate

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._global_epoch_step += 1

        row = {
            "global_epoch_step": self._global_epoch_step,
            "round_idx": self._round_idx,
            "train_size": self._train_size,
            "val_size": self._val_size,
            "local_epoch": epoch + 1,
            "learning_rate": self._learning_rate,
        }

        for key in self.fieldnames:
            if key in row:
                continue
            value = logs.get(key)
            row[key] = "" if value is None else float(value)

        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
