# Environment Selection During Refactoring

Two usable local TensorFlow environments were discovered.

## Environment A: `.venv`

- Selected by Poetry and PyCharm.
- TensorFlow 2.10.1, CPU-only.
- NumPy 1.23.5.
- scikit-learn 1.7.2.
- `pip check` reports no broken requirements.

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

## Runtime smoke-test result

Environment A successfully imported the project and performed forward
passes through both the v1 and v2 models.

Environment A is selected as the canonical runtime for Phase 0 and
Phase 1 characterization.

Environment B remains preserved only as a compatibility comparison.

No environment packages were changed.

Environment B also successfully built both models and performed finite
forward passes after the smoke-test introspection was made compatible
with newer Keras versions. It remains a secondary compatibility
environment because it is not selected by Poetry or PyCharm.
