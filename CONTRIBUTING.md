# Contributing

## Setup

```bash
git clone <this-repo>
cd aegis_df3
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # full hashed closure (runtime + dev tooling)
pip install -r requirements-dev.txt
cp .env.example .env                   # then fill in GROQ_API_KEY
```

This repo standardizes on Python 3.12. If you change `requirements.txt`,
regenerate the lockfile under Python 3.12:

```bash
pip-compile requirements.txt requirements-dev.txt --generate-hashes -o requirements-lock.txt
```

## Running the app

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Or via Docker:

```bash
docker compose up
```

## Running tests and lint

```bash
pytest -q --cov --cov-report=term-missing
ruff check .
pip-audit -r requirements-lock.txt
```

All three run automatically in CI on every push and pull request
(`.github/workflows/ci.yml`). The test step fails the build if coverage
drops below 65%.

## Commit conventions

- Keep each feature or fix in its own small commit (or PR), including the
  tests that verify it — avoid bulk commits that mix formatting,
  refactors, and features together.
- If you're pairing or a teammate contributes, please commit under your
  own identity/email so the history reflects actual authorship.

## Project structure

See `README.md` for the architecture diagram and directory layout.
