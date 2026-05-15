# football-outcomes-predictor

## Install (editable)
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -U pip
pip install -e ".[dev]"    # add ,[tf],[firebase],[scrape] as needed
pre-commit install
```

## Running (console scripts)
```bash
fop.main --help
fop.app --help
fop.train-ann --help
```

## Environment
Copy `.env.example` to `.env` and set:
- `FIREBASE_CREDENTIALS` -> path to your Firebase service account JSON (do **not** commit the file).
