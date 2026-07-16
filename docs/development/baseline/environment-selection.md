# Environment Selection During Refactoring

Two usable local TensorFlow environments were discovered.

## Environment A: `.venv`

- Selected by Poetry and PyCharm.
- TensorFlow 2.10.1, CPU-only.
- NumPy 1.23.5.
- scikit-learn 1.7.2.
- Contains one unrelated-looking `pip check` discrepancy involving
  `mumin` and `python-dotenv`.

## Environment B: `venv`

- Alternate local environment.
- TensorFlow 2.17.0, CPU-only.
- NumPy 1.25.2.
- scikit-learn 1.5.1.
- `pip check` reports no broken requirements.

## Current decision

No environment will be modified during Phase 0.

Environment A remains the default for initial source inspection because
it is selected by both Poetry and PyCharm.

Environment B is retained as a comparison environment.

Before running full baseline training, a small import and forward-pass
smoke test will be executed in both environments. The environment that
successfully executes the unchanged project with the fewest compatibility
issues will be selected for characterization runs.

Neither environment is currently CUDA-enabled.
