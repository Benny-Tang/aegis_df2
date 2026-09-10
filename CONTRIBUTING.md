# Contributing

## Setup

```bash
git clone <this-repo>
cd aegis-amd
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env   # then fill in GROQ_API_KEY
```

## Running the app

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

## Running tests and lint

```bash
pytest -q
ruff check .
```

Both run automatically in CI on every push and pull request
(`.github/workflows/ci.yml`).

## Commit conventions

- Keep each feature or fix in its own small commit (or PR), including the
  tests that verify it — avoid bulk commits that mix formatting,
  refactors, and features together.
- If you're pairing or a teammate contributes, please commit under your
  own identity/email so the history reflects actual authorship.

## Project structure

See `README.md` for the architecture diagram and directory layout.
