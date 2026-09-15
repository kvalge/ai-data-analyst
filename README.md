<!-- README.md -->

# AI Data Analyst

A private, local alternative to prompt-to-analysis tools. You upload data or
point at a read-only Postgres database, inspect it, and ask questions in chat.
All LLM inference runs on your machine through **Ollama**. Sensitive rows are
not sent to hosted APIs, and Ollama `:cloud` models are rejected.

The app is a Streamlit shell around a LangGraph agent. Chat can ask the model
to list, sample, profile, load a file, run a checked read-only SQL query,
or run checked Python in a local sandbox. Standard and Guided pause
before generated SQL or Python; you can edit the text, Approve runs
it in the sandbox, Reject does not.
Auto runs it without asking; the tool-use audit line is still written.

## What you can do now

- Upload CSV / Excel / JSON, or attach a configured Postgres source
- Upload domain-context files (markdown / text / PDF) — stored separately,
  never treated as datasets
- **Preview** a file: column types and a short sample (`SAMPLE_N_ROWS`)
- **Profile** a file: bounded-head schema, data-quality, and EDA (cached;
  not a whole-file profile; the dataset is not shown)
- **Chat** with the local primary model (it may call list / sample / profile /
  load_full_file, query_database, run_analysis_code, or
  retrieve_domain_context). Chat pauses to confirm the source when none are registered,
  several are present with none selected, or the sample/schema is empty. In
  **Guided** mode it also pauses after schema, data quality, and EDA.
  Standard and Guided also pause before `run_analysis_code` or
  `query_database` (editable textarea; Approve or Reject).
  **Auto** runs that generated SQL or Python without pausing; the audit
  line is still written. Every mode pauses before an over-limit
  `load_full_file` (Approve loads anyway; Reject does not). A
  malformed tool JSON or unknown tool is retried once, then shown as an
  error; the app does not invent a tool call. Successful tool use and graph
  profiling append one JSONL line under `data/logs/audit.jsonl` (tool,
  source_id, timestamp; generated SQL/Python also logs the code, whether
  that snippet was truncated, the HITL decision, and sandbox outcome; a
  rejected over-limit load logs reject; never file contents or dataset
  rows). An audit disk error is a warning, not a failed tool.

## What is not here yet

Chat still has the primary model write SQL/Python tool arguments; a
plan→coding-model helper validates generated code but is not in the
graph yet. Context retrieve is keyword/TF-IDF over CONTEXT_DIR; the
index rebuilds when a context file is uploaded. Postgres queries are one
checked SELECT/WITH. Sandbox Python is AST-checked, then run in a
subprocess; results are stdout plus csv/png paths, not row dumps.

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

**Sidebar:** HITL mode (Guided / Standard / Auto; default Standard), New
chat (a new graph thread; the previous thread stays unused on disk),
uploads, source list, optional Postgres test. **Main:** preview and
profile panels (they can stay open together; close either from its
heading), then chat. Restarting Streamlit resumes the last thread from
`CHECKPOINT_PATH` (`./data/checkpoints/graph.sqlite` by default).
Checkpoints store a short tool summary and artifact paths, not tables.
The model prompt keeps the last `MAX_PROMPT_TURNS` (default 8); the chat
UI still shows the full thread. Follow-up questions can reuse listed
artifact paths from earlier tools. A `.csv` (or `.parquet`) path under
`ARTIFACT_DIR` is shown as a table in chat. A `.png` path under
`ARTIFACT_DIR` is shown as an image. The agent is told to save those
files in the sandbox work dir and not paste large tables into the chat.
When a last-tool summary or artifact path exists, chat offers
**Download report.md** (summary line and paths only, not tables).
The file name includes a short thread id and UTC time.

In **Guided** mode, chat pauses after schema, then data quality, then EDA
(Continue / Skip remaining / Abort). Standard and Auto run those steps
without pausing. While a turn is running, chat shows **Stop**; the graph
does not start another tool (an in-flight tool may finish). The sidebar
Profile button always shows the full cached summary.

Optional Postgres: set `DB_NAME` and `DB_USER` in `.env`. Use **Test
connection**, then **Use configured database** to list it as a source.
Credentials stay in `.env`; there is no password form.

## Develop

From the repo root:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\pyright.exe
```

Typecheck covers `src/` only. Tests nest under `tests/` by `src/` package
(`tests/agent/`, `tests/tools/`, …). Shared fixtures stay in
`tests/conftest.py` and `tests/fixtures/`. `pytest` still runs everything.

## Docker

This repo has no Dockerfile or Compose file. Run the app in a host venv
with **Ollama on the same machine** (needed for local GPU/VRAM). Optional
Postgres is also on the host. Generated Python still runs in a subprocess,
not a container.

Revisit Docker later if you need to package the Streamlit app for another
machine, offer Compose for Postgres, or harden the sandbox into a
container. Even then, Ollama stays on the host; do not put `:cloud`
models in a compose file.
