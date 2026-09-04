# Implementation plan

Working rules: one step at a time; after a step, update this file, then ask for review; if approved, ask whether to commit; then start the next step only after permission.

**Status:** Phase 0 done. **1.1 done.** Next step: **1.2**.

Legend: `[x]` done · `[ ]` not started · `[~]` in progress

---

## Plan-time choices (spec left these open)

| Topic | Choice |
|---|---|
| Streamlit layout | Single page. Sidebar: HITL mode, data upload, context upload, source list. Main: profile/results panel, then chat (from 3.x). No multi-page app in v1. |
| Entry point | `src/ui/app.py`, run with `streamlit run src/ui/app.py`. |
| Checkpointer | `MemorySaver` in tests; `SqliteSaver` at `data/checkpoints/graph.sqlite` in the app (from 5.x; MemorySaver until then). |
| Chart library | Sandbox writes PNG (matplotlib). Streamlit shows the image. No Plotly server. |
| RAG v1 | Chunk context files + keyword/TF-IDF retrieval. No vector DB and no embedding model until 6.6 (optional). |
| MCP server | Not in v1. In-process MCP-shaped contracts only. |
| Docker | Not in this plan. Phase 8 only documents when to revisit. |
| New env vars (placeholders in `.env.example`) | `CONTEXT_DIR`, `CACHE_DIR`, `ARTIFACT_DIR`, `CHECKPOINT_PATH`, `MAX_UPLOAD_BYTES`, `MAX_FULL_LOAD_ROWS`, `SAMPLE_N_ROWS`, `SANDBOX_TIMEOUT_S`, `MAX_PROMPT_CHARS` |

**Default limits:** upload 50 MB; sample 50 rows; auto full-load pause above 100 000 rows or 50 MB; sandbox 30 s; prompt 8 000 characters.

**SQL static checks:** single statement; reject write/DDL (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `GRANT`, `COPY`, `INTO`).

**Python AST deny:** `eval`/`exec`/`__import__`; imports of `subprocess`, `socket`, `requests`, `http`, `urllib` (except if we later allow none); writes outside the sandbox working/artifact dir; `os.system` / `os.popen`.

**LangGraph topology (target; nodes are added across phases, not all at once):**

```
START → confirm_sources → detect_schema → run_dq → run_eda
      → agent ⇄ validate_output → execute_tool
      → summarize_result → END
```

Interrupts: `confirm_sources` (multi/broken source); after schema/DQ/EDA if Guided; before `run_analysis_code` / generated SQL if Standard or Guided; before `load_full_file` over threshold; after a failed retry.

---

## Phase 0 — Environment and model selection

- [x] **0.1** Repo skeleton: `src/`, `tests/`, `.env` / `.env.example` placeholders, `.gitignore` (`venv/`, `.env`, `data/`).
- [x] **0.2** Local Ollama lineup chosen; `scripts/compare_ollama_models.py` measures latency and compares outputs.
- [x] **0.3** `PROJECT_SPEC.md` and `.cursor/RULES.md` agreed.

---

## Phase 1 — Data sources + thin Streamlit shell

Do not add LangGraph, Ollama calls, or profiling yet.

### 1.1 Pytest and fixtures

- [x] Add `pytest` to `requirements.txt`.
- [x] Add `tests/conftest.py` and `tests/fixtures/sample_sales.csv` (tiny: date, region, revenue).
- [x] One test that the fixture loads with pandas.
- **Done when:** `pytest` passes from repo root.

### 1.2 Settings module

- Add `src/config.py`: load `.env` via dotenv; typed settings for Ollama host/model env vars, all path vars, limits above; never hardcode model names.
- Create missing path env vars in `.env.example` with placeholders.
- Tests: defaults resolve; `UPLOAD_DIR` is read from env.
- **Done when:** settings tests pass; `.env.example` lists the new keys.

### 1.3 Logging

- Add `src/logging_setup.py`: configure root logging from `LOG_LEVEL`; no `print` in library code.
- Test: logger emits at the configured level (caplog).
- **Done when:** test passes.

### 1.4 Runtime directories

- Add `src/storage/paths.py` (or similar): `ensure_runtime_dirs()` for uploads, context, cache, artifacts, checkpoints, logs.
- Test with `tmp_path`.
- **Done when:** dirs are created idempotently.

### 1.5 Data-file input validation

- Allowlist: `.csv`, `.xlsx`, `.xls`, `.json`.
- Reject empty, wrong suffix, over `MAX_UPLOAD_BYTES`.
- Tests for each reject path + one accept path (fixture).
- **Done when:** validator tests pass. No Streamlit yet.

### 1.6 Source record + file hash

- Dataclass/model: `source_id`, `kind=file`, `original_name`, `stored_path`, `sha256`, `created_at`.
- Hash the stored bytes; `source_id` derived from hash (stable).
- Tests with the fixture file.
- **Done when:** two copies of the same bytes get the same `source_id`.

### 1.7 Persist upload to disk

- Copy/save a validated file into `UPLOAD_DIR` using the source id in the filename; write a small sidecar JSON (or a registry JSON) so sources survive restart.
- Idempotent if the same hash is uploaded again.
- Tests on `tmp_path`.
- **Done when:** save + re-list from disk works without Streamlit.

### 1.8 Tool contract helper + `list_available_sources`

- Add `src/tools/contracts.py`: name, description, JSON Schema for args, structured result (MCP-shaped; in-process).
- Implement `list_available_sources` over the registry from 1.7.
- Unit test: empty registry; one file source.
- **Done when:** tests pass; no LLM.

### 1.9 Streamlit skeleton

- Add `streamlit` to `requirements.txt`.
- `src/ui/app.py`: title, sidebar placeholder, main placeholder. No uploads yet.
- README: how to create venv, copy `.env`, `streamlit run src/ui/app.py`.
- **Done when:** app starts (manual); README documents it.

### 1.10 Streamlit: data file upload + list

- Sidebar uploader → validate (1.5) → persist (1.7) → show `list_available_sources` in the sidebar.
- Show validation errors in the UI.
- **Done when:** uploading the fixture CSV shows one source; bad file shows an error. Manual UI check.

### 1.11 Context-file upload slot (store only)

- Allowlist: `.md`, `.txt`, `.pdf`.
- Same size limit; save under `CONTEXT_DIR`; do not profile, execute, or retrieve.
- List context files separately from data sources.
- Test validators; manual UI check.
- **Done when:** a `.md` file appears under context, not as a data source.

### 1.12 HITL mode in session state

- Sidebar select: Guided / Standard / Auto; default Standard.
- Store in `st.session_state` only (no graph yet).
- **Done when:** mode persists across reruns in one session.

### 1.13 `read_file_sample`

- Tool: path/`source_id` + `n_rows` (default `SAMPLE_N_ROWS`).
- Return dtypes, column names, head as records; never the full file.
- Tests with the fixture.
- **Done when:** sample tests pass; optional “preview” button in UI calling this tool (no LLM).

### 1.14 Postgres settings + read-only connection test

- Add `psycopg` (or `psycopg2`) when this step starts.
- Connect from env; fail if connection is not usable; document that the DB role must be read-only (enforced in DB, not in Python).
- Test with mocks (no real DB required in CI).
- **Done when:** mock tests pass; UI can show connection ok/fail if env is set.

### 1.15 Register Postgres as a source

- `kind=postgres`; do not store the password in the source registry (use env).
- `list_available_sources` includes it when config is present.
- Streamlit: short “use configured DB” toggle, not a password form in v1 (credentials stay in `.env`).
- **Done when:** list shows file sources and optionally one DB source.

---

## Phase 2 — Profiling, EDA, cache, Guided stepper (UI-level)

Profiling is functions + cache. Guided pauses here are **Streamlit Continue buttons**, not LangGraph `interrupt()` yet (that is Phase 3).

### 2.1 Schema detection

- From a sample: columns, pandas dtypes, null count, row count estimate (file: full row count if cheap; else “unknown”).
- Tests on the fixture.
- **Done when:** schema dict is stable and tested.

### 2.2 Profile cache

- Key: `source_id` + file hash (files) or a connection fingerprint (DB).
- Store JSON under `CACHE_DIR`; invalidate when hash changes.
- Tests: hit, miss, stale hash.
- **Done when:** cache tests pass.

### 2.3 DQ: nulls

- Per-column null count and %.
- Test with a fixture that has nulls.

### 2.4 DQ: duplicates

- Duplicate row count; optional key-column later — v1 = full-row duplicates.
- Test with a duplicated fixture.

### 2.5 DQ: type mismatches

- Numeric/date columns stored as strings, mixed types.
- Test with a messy CSV fixture.

### 2.6 DQ: inconsistent formatting

- Date formats, numeric thousands separators, whitespace in categoricals.
- Test with a messy fixture.

### 2.7 DQ: outliers

- Simple IQR (or z-score) on numeric columns; record bounds and counts.
- Test with an obvious outlier.

### 2.8 EDA: summary stats

- Count, mean/median/min/max for numerics; nunique for categoricals.
- Tests on the sales fixture.

### 2.9 EDA: distributions

- Hist bins or value_counts top-N; store compact JSON, not plots yet.
- Tests.

### 2.10 EDA: correlations

- Numeric-only Pearson; skip if <2 numeric columns.
- Tests.

### 2.11 `profile_source` tool

- Orchestrate schema + DQ + EDA; write cache; return a compact summary (not the dataset).
- Use sample or a bounded read; do not put the frame in logs.
- Tests: cached vs fresh.
- **Done when:** tool tests pass.

### 2.12 `load_full_file` with threshold

- Load only if under limits; otherwise return “needs approval” (UI confirm in this phase; graph interrupt in 4.x).
- Test under and over limit (`MAX_FULL_LOAD_ROWS` / bytes).

### 2.13 Streamlit: show profile

- Button “Profile selected source”; render schema / DQ / EDA from the tool result.
- Manual check.

### 2.14 Guided stepper in UI

- If mode is Guided: run schema → wait Continue → DQ → Continue → EDA.
- Standard/Auto: run all, then show.
- Skip remaining / abort buttons.
- **Done when:** Guided pauses; Standard does not. Still no LangGraph.

---

## Phase 3 — Agent orchestration core

No sandbox code execution yet. Tools: `list_available_sources`, `read_file_sample`, `profile_source`, `load_full_file` (threshold message only).

### 3.1 Wrap existing tools in contracts

- Register tools in a dict: name → contract + callable.
- Test: every registered tool has schema + description.

### 3.2 JSON output helpers

- Strip markdown fences; `json.loads`; schema validate.
- Retry policy function: one stricter retry, then fail (no guess).
- Tests: fenced JSON, bad JSON, schema miss.

### 3.3 Ollama client wrapper

- `src/agent/llm.py`: host + model from env; `think=false` for structured calls; no hardcoded model names.
- Role picker: primary by default.
- Tests: mock urllib/httpx; assert model comes from env.

### 3.4 Agent state

- `src/agent/state.py` TypedDict: `messages`, `hitl_mode`, `source_ids`, `profile_summary`, `pending_interrupt`, `last_tool_result`, `artifacts` (paths only), `error`.
- No dataframes in state.

### 3.5 System prompt

- `src/agent/prompts.py`: data safety, tool use, never dump full data, HITL rules.
- No Streamlit imports.

### 3.6 Graph stub + MemorySaver

- Add `langgraph` (and the smallest Ollama/langchain helper you need) to requirements.
- Graph: one `agent` node that returns a plain text reply (tools off).
- Test with mocked LLM.

### 3.7 Streamlit chat → graph

- Main area chat; send user text into the graph; show the reply.
- Do not put system prompts in the UI module.
- Manual check: one round-trip with Ollama.

### 3.8 Bind data tools to the agent

- Agent may call list / sample / profile / load_full_file.
- `execute_tool` node + allowlist.
- Mocked test: model asks `list_available_sources` → tool runs → reply.

### 3.9 Validate tool name and arguments

- Unknown tool or bad args → validation error path (3.2 retry then fail).
- Tests.

### 3.10 `confirm_sources` interrupt

- Interrupt if 0 sources, >1 source with none selected, or sample/schema empty.
- Streamlit: confirm / select / abort; resume graph.
- Mocked test for interrupt payload.

### 3.11 Move Guided profiling into the graph

- Nodes `detect_schema`, `run_dq`, `run_eda` + interrupts in Guided.
- Remove the Phase 2 UI stepper (or make it call the graph).
- Mocked tests: Standard runs three nodes without interrupt; Guided interrupts three times.

### 3.12 Visible failure after retry

- After one strict retry, surface the error in chat; do not invent a tool call.
- Test.

### 3.13 Audit log (minimal)

- Append-only JSONL under `logs/`: source_id, tool name, timestamp (no full file contents).
- Test: one line written on tool use.

---

## Phase 4 — Code generation, sandbox, code/SQL approval

### 4.1 Subprocess runner

- `src/execution/sandbox.py`: run a Python file in a subprocess; cwd = sandbox work dir; timeout `SANDBOX_TIMEOUT_S`; capture stdout/stderr/exit code.
- Never `exec`/`eval` in the app process.
- Test: child prints hello; timeout kills a sleeper.

### 4.2 Sandbox work and artifact dirs

- Per-run temp work dir; artifacts copied only into `ARTIFACT_DIR`.
- Test: write outside work dir fails or is not visible to the app as success.

### 4.3 Python AST checks

- Parse generated code; apply the deny-list; reject before spawn.
- Tests: allowed pandas snippet; rejected `subprocess` import.

### 4.4 SQL checks

- Parse/reject writes, multi-statement, empty.
- Tests.

### 4.5 `query_database` tool

- Parameterized read-only query; run only after SQL checks.
- Mock cursor tests; skip real DB in default pytest.

### 4.6 `run_analysis_code` tool

- AST check → sandbox → structured result: stdout + artifact paths (csv/png).
- Tests with a fixture CSV copied into the work dir.

### 4.7 Code-generation path

- Structured plan/JSON from **primary**; raw code from **coding** model only when the task is code (per spec).
- Strip fences; then AST/SQL validate.
- Mocked tests for role selection.

### 4.8 HITL before execute (Standard + Guided)

- Interrupt payload: code, source_id, short rationale.
- Resume: approve / edit+run / reject (reject must not execute).
- Mocked graph tests.

### 4.9 Auto mode skips code interrupt

- Still write audit log.
- Test: Auto executes without interrupt; Standard interrupts.

### 4.10 Full-file load interrupt over threshold

- Wire `load_full_file` to `interrupt()` (replace the Phase 2 UI confirm).
- Test.

### 4.11 Streamlit approval widget

- Show code in a text area (editable); Approve / Reject.
- UI must not run the code itself.
- Manual check.

### 4.12 Audit: generated code/SQL

- Log code/SQL text, decision (approve/edit/reject), sandbox outcome.
- Test.

---

## Phase 5 — Multi-turn memory

### 5.1 Thread id in Streamlit

- One thread per browser session (or an explicit “new chat”).
- Pass `thread_id` into the graph config.

### 5.2 SqliteSaver

- Persist checkpoints to `CHECKPOINT_PATH`.
- Restart Streamlit: resume the same thread.
- Test: MemorySaver for unit tests; optional sqlite round-trip test on tmp path.

### 5.3 Result summaries in state

- After tools, store a short text summary + artifact paths, not tables.
- Test: state JSON has no large record lists.

### 5.4 Truncate / last-N messages

- Keep last N turns + the system prompt; drop older raw tool dumps.
- Test.

### 5.5 Follow-up uses prior artifacts

- Prompt/state tells the agent which artifact paths exist (“break that down by region”).
- Mocked test: second turn mentions the prior summary.

---

## Phase 6 — Domain context / local RAG

### 6.1 Context file readers

- Load `.md`/`.txt`; PDF text extract if already uploaded.
- Test with a small markdown fixture.

### 6.2 Chunker

- Split on headings/paragraphs; record `source_file`, chunk id, text.
- Tests.

### 6.3 Keyword/TF-IDF retrieve

- `retrieve_domain_context(query)` → top-k chunks.
- Tests: query hits the relevant fixture chunk.

### 6.4 Bind RAG tool to the graph

- Agent may call it; inject snippets into context; never index CSVs.
- Test: data files are not in the corpus.

### 6.5 Reindex on context upload

- After a new context file, rebuild the index; idempotent.
- Test.

### 6.6 Optional embeddings

- Only if keyword retrieval is too weak: local Ollama embedding model + a local store.
- Skip unless we decide it is needed; would require a spec key-decision line.

---

## Phase 7 — Richer output

### 7.1 Table artifacts in UI

- If sandbox wrote csv/parquet under `ARTIFACT_DIR`, show `st.dataframe`.
- Manual + a unit test that a known csv path renders via a small helper.

### 7.2 Chart PNG artifacts

- Matplotlib in sandbox → PNG → `st.image`.
- Fixture test: helper accepts a PNG path.

### 7.3 Agent instructions for artifacts

- Prompt: save tables/charts to the work dir; do not paste large tables into the chat.
- Eval later in 9.x.

### 7.4 Markdown report export

- Optional “download report.md” from the last summary + artifact list.
- Manual check.

### 7.5 Dashboards

- Out of v1 unless 7.1–7.4 are solid. Do not start unless explicitly requested.

---

## Phase 8 — Safety hardening

### 8.1 Centralize limits

- All thresholds only in `src/config.py` / env; grep for leftover magic numbers.
- Tests read from settings.

### 8.2 Prompt length limit

- Reject/trim over `MAX_PROMPT_CHARS` before the LLM call.
- Test.

### 8.3 Stop / cancel in UI

- User can stop a run; graph should not start a new tool after stop.
- Manual check; test if cooperative cancel is easy.

### 8.4 Audit log completeness

- Every source touch, tool, generated code, HITL decision, sandbox exit.
- Test a full fake run writes the expected keys.

### 8.5 Docker revisit note

- Short section in README/spec: when Docker would be considered; Ollama stays on the host; no compose in this phase.
- Spec key-decision only if we actually adopt Docker.

---

## Phase 9 — Eval suite formalization

Unit/integration tests already exist from earlier phases.

### 9.1 Pytest markers

- `@pytest.mark.ollama` for tests that need a live local model; default `pytest` skips them.

### 9.2 Eval prompt corpus

- `tests/eval/prompts.json`: tool-selection (list vs sample vs profile); fenced-JSON trap; one-word speed; short pandas code (coding model).
- No sensitive data.

### 9.3 Wrap `scripts/compare_ollama_models.py`

- Share corpus with the eval runner; keep the script runnable by hand (command comment at top).

### 9.4 Parse-success and tool-selection checks

- Assert JSON schema success; assert chosen tool name on the mocked or live set.
- Document how to run: `pytest -m ollama`.

### 9.5 README: test and eval commands

- Default `pytest`; optional Ollama eval; never call cloud APIs.

---

## Out of scope until asked

- Standalone MCP server
- Multi-user auth
- Extra databases beyond Postgres
- LangSmith / cloud eval
- Container sandbox implementation
- Dashboard code generation (7.5)

---

## Current focus

**1.1 done** (`pytest` from repo root: 1 passed). Next: **1.2 Settings module**.
Do not start 1.2 until you say to proceed.
