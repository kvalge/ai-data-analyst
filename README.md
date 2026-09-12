<!-- README.md -->

# AI Data Analyst

A private, local alternative to prompt-to-analysis tools. You upload data or
point at a read-only Postgres database, inspect it, and ask questions in chat.
All LLM inference runs on your machine through **Ollama**. Sensitive rows are
not sent to hosted APIs, and Ollama `:cloud` models are rejected.

The app is a Streamlit shell around a LangGraph agent. Chat can ask the model
to list, sample, profile, or load a file. Query, generated code, and the
sandbox are not bound yet.

## What you can do now

- Upload CSV / Excel / JSON, or attach a configured Postgres source
- Upload domain-context files (markdown / text / PDF) — stored separately,
  never treated as datasets
- **Preview** a file: column types and a short sample (`SAMPLE_N_ROWS`)
- **Profile** a file: bounded-head schema, data-quality, and EDA (cached;
  not a whole-file profile; the dataset is not shown)
- **Chat** with the local primary model (it may call list / sample / profile /
  load_full_file)

## What is not here yet

Chat cannot query Postgres or run generated SQL/Python. The sandbox and RAG
retrieval come later. Context files are stored only.

## Requirements

- Python 3.11+
- A local [Ollama](https://ollama.com) instance with four models named in `.env`
  (see `.env.example`; do not put real model tags in that file)
- Optional: Postgres with a **read-only** role (granted in the database, not
  by this app)

## Setup

From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Set the four `OLLAMA_MODEL_*` names (and optional `DB_*`) in `.env`.

## Run

```powershell
streamlit run src/ui/app.py
```

**Sidebar:** HITL mode (Guided / Standard / Auto; default Standard), uploads,
source list, optional Postgres test. **Main:** preview and profile panels
(they can stay open together; close either from its heading), then chat.

In **Guided** mode, profile pauses after schema, then data quality, then EDA
(Continue / Skip remaining / Abort). Standard and Auto show all sections at
once.

Optional Postgres: set `DB_NAME` and `DB_USER` in `.env`. Use **Test
connection**, then **Use configured database** to list it as a source.
Credentials stay in `.env`; there is no password form.

## Develop

From the repo root:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\pyright.exe
```

Typecheck covers `src/` only. Tests stay flat under `tests/`.
