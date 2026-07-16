# Legacy Environment Baseline

## Environment A — Current Poetry and PyCharm `.venv`

- PyCharm interpreter:
  `C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\.venv\Scripts\python.exe`

- `sys.executable`:
  `C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\.venv\Scripts\python.exe`

- Python:
  3.10.8, 64-bit, MSC v.1933

- Virtual-environment path:
  `C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\.venv`

- pip:
  25.2

- Poetry:
  2.1.1

- Poetry environment:
  `C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\.venv`

- Environment-management status:
  Poetry, PyCharm, the activated shell, and `poetry run`
  all resolve to this `.venv` interpreter.

### Principal packages

- NumPy:
  1.23.5

- scikit-learn:
  1.7.2

- TensorFlow import module:
  2.10.1

- TensorFlow distributions:
  - `tensorflow-intel`: 2.10.1
  - `tensorflow-cpu`: 2.10.1
  - Distribution named exactly `tensorflow`: not installed

- TensorBoard:
  2.10.1

### TensorFlow hardware support

- TensorFlow built with CUDA:
  No

- TensorFlow-visible GPUs:
  None

- TensorFlow build type:
  CPU-only Windows build using oneDNN optimizations.

### Dependency consistency

`python -m pip check` reports:

`mumin 1.8.0 has requirement python-dotenv~=0.19.0, but python-dotenv 1.2.1 is installed.`

This discrepancy was recorded but has not been corrected during
the environment-freezing phase.

## Environment B — Alternate local `venv`

- Interpreter:
  `C:\Users\kutalekj\PycharmProjects\football-outcomes-predictor\venv\Scripts\python.exe`

- Python:
  3.10.8, 64-bit, MSC v.1933

- NumPy:
  1.25.2

- scikit-learn:
  1.5.1

- TensorFlow:
  2.17.0

- TensorFlow distributions:
  - `tensorflow`: 2.17.0
  - `tensorflow-intel`: 2.17.0
  - `tensorflow-cpu`: not installed

- TensorBoard:
  2.17.1

- TensorFlow built with CUDA:
  No

- TensorFlow-visible GPUs:
  None

- Dependency consistency:
  `pip check` reported no broken requirements.

- Relationship to thesis experiments:
  Unknown. This environment is retained as an alternate candidate
  and must not yet be described as the confirmed thesis-training
  environment.

## Operating system and hardware

- Operating system:
  Microsoft Windows 11 Home

- Windows version:
  10.0.26200

- Windows build:
  26200

- Platform:
  Windows-11-10.0.26200-SP0

- GPU:
  NVIDIA GeForce RTX 3050 Laptop GPU

- Device Manager driver version:
  32.0.15.7283

- Device Manager driver date:
  2025-03-14

- DirectX:
  12, feature level 12.2

## CUDA and cuDNN

- `nvcc`:
  Not available on PATH.

- Standalone CUDA toolkit:
  Not detected through `nvcc`.

- TensorFlow CUDA support:
  Both inspected project environments are CPU-only builds.

- cuDNN:
  Not used by either inspected TensorFlow build.

- NVIDIA driver information:
  Captured separately in
  `docs/development/baseline/environment/nvidia-smi.txt`.

## Poetry lock status

The existing root `poetry.lock` was created before significant
changes to `pyproject.toml`.

`poetry check` reports that `pyproject.toml` changed significantly
after the lock file was generated.

The lock file is retained only as a historical artifact. It is not
currently considered an authoritative or reproducible environment
definition.

No `poetry lock`, `poetry update`, `poetry install`, or package
installation was performed during this phase.
