<!-- README.md -->

# AI Data Analyst Application

Typecheck: `pyright` (from the repo root; checks `src/` only).
Tests: `pytest`.

## Run

From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Set the four `OLLAMA_MODEL_*` names in `.env` (placeholders in `.env.example`). Then:

```powershell
streamlit run src/ui/app.py
```

Use the sidebar to choose a HITL mode (Guided / Standard / Auto; default
Standard), upload a data file (CSV / Excel / JSON), and optionally a
domain-context file (markdown / text / PDF). Context files are stored separately
and are not treated as datasets. Mode is session-only until the agent graph
exists. Use **Preview** on a data source to see column types and a short
sample (`SAMPLE_N_ROWS`); the full file is never loaded into the UI.

Optional Postgres: set `DB_NAME` and `DB_USER` in `.env` (and host/port/password
as needed). The database **role must be read-only** — grant that in Postgres;
the app only checks that a connection works. Use **Test connection** in the
sidebar. Check **Use configured database** to list it as a data source (no
password form; credentials stay in `.env`).
