from __future__ import annotations

import platform
import sys
import traceback

import numpy as np
import sklearn
import tensorflow as tf

from football_outcomes.cli.main import build_parser
from football_outcomes.training.train_mlp_rolling import (
    TrainConfig,
    build_model,
)

NUM_NUMERICAL_FEATURES = 8
NUM_TEAMS = 4
NUM_COMPETITIONS = 2
BATCH_SIZE = 2


def build_inputs() -> list[np.ndarray]:
    numerical = np.zeros(
        (BATCH_SIZE, NUM_NUMERICAL_FEATURES),
        dtype=np.float32,
    )

    home_ids = np.asarray([[0], [1]], dtype=np.int32)
    away_ids = np.asarray([[1], [2]], dtype=np.int32)
    competition_ids = np.asarray([[0], [1]], dtype=np.int32)

    strength = np.zeros(
        (BATCH_SIZE, 4, 11, 34),
        dtype=np.float32,
    )

    # [home values, home mask, away values, away mask]
    strength[:, 0, :, :] = 0.60
    strength[:, 1, :, :] = 1.00
    strength[:, 2, :, :] = 0.50
    strength[:, 3, :, :] = 1.00

    position_row = np.asarray(
        [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
        dtype=np.int32,
    )
    home_positions = np.tile(position_row, (BATCH_SIZE, 1))
    away_positions = np.tile(position_row, (BATCH_SIZE, 1))

    return [
        numerical,
        home_ids,
        away_ids,
        competition_ids,
        strength,
        home_positions,
        away_positions,
    ]


def prediction_array(output) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        output = output[0]

    if hasattr(output, "numpy"):
        output = output.numpy()

    return np.asarray(output)


def run_architecture(model_version: str) -> None:
    tf.keras.backend.clear_session()
    np.random.seed(123)
    tf.random.set_seed(123)

    config = TrainConfig(
        mode="binary_u25",
        model_version=model_version,
        seed=123,
        enable_branch_diagnostics=False,
        use_team_aux_head=False,
        use_team_strength=True,
        use_team_ids=True,
        use_comp_embedding=True,
        use_position_embedding=True,
        use_strength_masks=True,
    )

    model = build_model(
        num_num=NUM_NUMERICAL_FEATURES,
        num_teams=NUM_TEAMS,
        num_comps=NUM_COMPETITIONS,
        cfg=config,
    )

    output = model(build_inputs(), training=False)
    prediction = prediction_array(output)

    if prediction.shape != (BATCH_SIZE, 1):
        raise AssertionError(f"{model_version}: unexpected prediction shape " f"{prediction.shape}")

    if not np.isfinite(prediction).all():
        raise AssertionError(f"{model_version}: predictions contain non-finite values")

    if np.any(prediction < 0.0) or np.any(prediction > 1.0):
        raise AssertionError(f"{model_version}: binary predictions are outside [0, 1]")

    print(f"{model_version}: PASS")
    print(f"{model_version}: parameters={model.count_params()}")
    input_names = [str(getattr(tensor, "name", tensor)).split(":")[0] for tensor in model.inputs]

    output_names = list(getattr(model, "output_names", []))
    if not output_names:
        output_names = [str(getattr(tensor, "name", tensor)).split(":")[0] for tensor in model.outputs]

    print(f"{model_version}: inputs={input_names}")
    print(f"{model_version}: outputs={output_names}")
    print(f"{model_version}: prediction_shape={prediction.shape}")
    print(f"{model_version}: prediction_range=" f"({prediction.min():.6f}, {prediction.max():.6f})")


def main() -> int:
    print("Executable:", sys.executable)
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    print("NumPy:", np.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("TensorFlow:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

    parser = build_parser()
    print("CLI parser:", parser.prog)

    failures = 0

    for model_version in ("v1", "v2"):
        print()
        print("=" * 72)
        print(f"Testing architecture: {model_version}")
        print("=" * 72)

        try:
            run_architecture(model_version)
        except Exception:
            failures += 1
            print(f"{model_version}: FAILED")
            traceback.print_exc()

    print()
    print("Failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
