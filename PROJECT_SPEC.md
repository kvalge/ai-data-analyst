<!-- PROJECT_SPEC.md -->

# Private AI Data Analysis Application — Project Description

## Purpose

A self-hosted, private alternative to prompt-to-analysis-code tools,
built specifically for use with **sensitive data that cannot be sent to third-party
LLM APIs**. The user writes a prompt describing what they want to know; the
application's AI agent decides how to get the relevant data, profiles it, and
either performs automated analysis or generates and executes analysis code
(pandas/SQL) to answer the user's question — all without any data ever leaving
the local machine.

## Core principle: data safety

Every architectural decision is subordinate to this constraint:

- All LLM inference runs locally via **Ollama** — no hosted/cloud model APIs.
- Ollama's own **cloud model** variants (tagged `:cloud` in the model library)
  are explicitly excluded, since those route inference through Ollama's
  remote servers.
- No cloud embeddings, tracing, or eval services (no OpenAI, no LangSmith).
  Observability is local logs and local files only.
- Database access uses **read-only credentials enforced at the database
  level**, not just prompt-level instructions.
- Large datasets are never dumped wholesale into the LLM's context — the
  agent works from schema + samples, and executes code against the full
  data separately.
- Analysis **data files** and **business-context documents** are separate
  corpora. Tabular/data-source files are never ingested into RAG; they are
  only accessed through data tools (sample, profile, query, sandboxed code).

## What the application does (user-facing flow)

The v1 user-facing surface is a **Streamlit** app. The agent, tools, and
sandbox sit behind it; Streamlit does not contain system prompts or tool
logic.

1. **User provides inputs in the UI**
   - **Analysis data:** upload CSV / Excel / JSON, or configure a Postgres
     connection (read-only). Files land under `UPLOAD_DIR`.
   - **Business context (optional):** upload domain documents (markdown,
     text, PDF) — glossary, KPI definitions, process notes. These land in a
     separate context directory and are retrieved later via local RAG; they
     are not executed and are not treated as datasets.
2. **Agent reads and profiles the data** — detects schema, runs automated
   data-quality checks (nulls, duplicates, type mismatches, inconsistent
   formatting, outliers), and performs baseline exploratory data analysis
   (summary stats, distributions, correlations). Results are cached so this
   isn't repeated on every follow-up.
3. **User asks analysis questions in a chat panel** — the prompt is written
   by the user. The UI may later suggest follow-ups; it does not generate
   the system prompt or tool-selection prompt. Those live in the agent
   layer. The agent decides, per request, which tool to use and either runs
   a predefined analysis or generates and executes new code.
4. **Agent returns results in the same UI** — tables, summary text, and/or
   charts (as files/artifacts, not inlined blob data in the LLM context) —
   and supports multi-turn follow-up questions that reference prior results
   ("now break that down by region").
5. **Human-in-the-loop** — see the dedicated section below. Interaction
   (approve / edit / reject / continue) happens in Streamlit; LangGraph
   owns the pause/resume of the graph.

## Architecture

Layered, with strict separation of concerns:

```
Streamlit UI
    → LangGraph agent (decides; owns state, HITL interrupts, retries)
        → Tools with MCP-compatible contracts (does not decide)
            → file / DB / profile / sandbox / (later) RAG retrieve
        → Local RAG retriever (domain context only)
    → Execution sandbox (subprocess; generated code only)
    → Local disk (uploads, context, cache, checkpoints, artifacts, logs)
```

**Orchestration: LangGraph** — stateful graph, not a simple prompt loop.

- The agent makes *controllable, observable* decisions about which
  data-access tool to call — critical when the tool might touch sensitive
  data.
- Checkpointing and `interrupt()` support human-in-the-loop pause/resume.
- Tool-based design lets the agent choose between file access, database
  access, source discovery, and (later) domain-context retrieval per
  request, rather than following one hardcoded path.
- Strict separation of concerns and roles, controlled feedback loops,
  error recovery, structured goal-driven workflows, explicit state, and
  execution visibility (logging, audit records, summary files).

**MCP (tool protocol, not the orchestrator).** MCP does **not** decide
what action to take. LangGraph decides; tools are invoked afterward.

- Every tool is defined with an MCP-compatible contract: name, description,
  JSON Schema arguments, structured result.
- v1 implements tools in-process and binds them to the graph. A standalone
  MCP server (so other local clients can reuse the same tools) is optional
  later and is not required to ship.
- Do not add MCP servers as a substitute for the agent graph.

**RAG (domain context, not a data lake).** Local retrieval over
business-context documents so the agent can use domain vocabulary and KPI
definitions when deciding how to analyse data.

- Retrieval is local-only (local embedding model via Ollama, or a simpler
  local retriever if embeddings are not yet chosen).
- Retrieved snippets are injected into the agent context; raw datasets are
  not.
- Small context corpora may be summarized or included directly until a
  vector index is justified.
- Embedding model and vector store are plan-time choices.

**UI: Streamlit (v1).** Python-native; file upload, chat, tables, charts,
and approval buttons without a separate frontend. Introduced as a thin
shell with source upload (not postponed until reporting). A CLI is out of
scope for v1.

**Data access tools (v1 target set; exact schemas in the plan):**

- `list_available_sources()` — what's currently uploaded/connected
- `read_file_sample(path, n_rows)` — peek at schema/head before full load
- `profile_source(source_id)` — run DQ checks + EDA, cache results
- `query_database(connection_id, sql)` — parameterized, read-only
- `load_full_file(path)` — full load when sampling isn't sufficient
- `run_analysis_code(code)` — execute generated pandas/SQL in a sandbox
- `retrieve_domain_context(query)` — local TF-IDF over uploaded context files; never data files

**Later (not v1 data-access tools):** generated reports and dashboards as
output artifacts, after core Q&A and charts work.

**Execution sandbox:** isolated **subprocess** with time/resource limits
and write access only to a designated working/output area. Container
isolation is a later hardening option, not the v1 path.

**Tech stack (decided):** Python 3.11+, Ollama, LangGraph, Streamlit,
pandas, python-dotenv, Postgres driver (when the DB tool is built).
Vector store / embedding model / chart library: plan-time.

**v1 product scope:** single user, local machine, no authentication,
Postgres as the only database target.

## Human-in-the-loop

HITL is a first-class control, not a late add-on. The spec fixes **when**
the human may intervene; LangGraph node wiring is plan-time.

**Session modes** (chosen in the UI, stored in graph config):

| Mode | Profiling / EDA | Generated SQL or Python | Large full-file load |
|---|---|---|---|
| Guided | Pause after each major step | Always pause | Always pause |
| Standard (default) | Run through automatically | Always pause | Pause above size/row threshold |
| Auto | Run through automatically | Run without asking (still logged) | Pause above threshold |

Even in Auto, the user can stop the run; nothing is silent.

**Interrupt points (where the human intervenes):**

1. **Mode and source setup** — UI, before the graph runs: choose mode,
   upload/connect, optionally add domain-context files.
2. **After source registration** — confirm the intended source(s) before
   profiling if more than one is present or the schema looks empty/broken.
3. **Profiling steps (Guided only)** — after schema detection, after DQ
   checks, after EDA. User inspects the step result and continues, skips
   remaining profiling, or aborts.
4. **Before executing generated SQL or Python (Standard + Guided)** —
   show the code, intended source, and a short rationale. User **approves**,
   **edits then runs**, or **rejects** (agent must not execute). This is
   the primary safety gate.
5. **Before a full-file load** that exceeds configured size/row limits.
6. **On repeated validation failure** — after one strict retry (see
   Validation), surface the error in the UI rather than guessing.

**Where it is implemented:** Streamlit renders the interrupt payload
(code, diffs, profile summaries). LangGraph `interrupt()` / resume carries
the user's decision back into state. The UI must not execute model-generated
code itself.

## Validation, metrics, and observability

Never assume LLM output is well-formed. Fail visibly after a bounded retry.

**Input validation** (before tools run):

- File type allowlist, size and row limits, non-empty readable files.
- Prompt length limit.
- Domain-context files: type/size allowlist; never executed.
- DB connections: read-only role; connection test before use.

**LLM / tool-output validation** (after each model call or tool result):

- Structured output: parse JSON defensively (strip markdown fences),
  validate against a schema; retry **once** with a stricter instruction;
  then fail or escalate model — never guess.
- Tool name must be on the allowlist; arguments must match the tool schema.
- Generated SQL: parseable; reject writes / DDL / multiple statements.
- Generated Python: parseable AST; deny network, subprocess, and writes
  outside the sandbox working area.
- Sandbox results: timeout, non-zero exit, and missing artifacts are
  errors, not empty "success".

**DQ / EDA checks** are a product feature (nulls, duplicates, types,
outliers) and are separate from the engineering validators above.

**Metrics (local only):**

| Metric | Where |
|---|---|
| Latency (TTFT, tok/s, load time, per-graph-step wall time) | Existing model benchmark script; later eval suite; runtime logs |
| Structured-output parse success; retry count | Agent/validation layer logs |
| Tool-selection correctness | Fixed eval prompt set (slow/optional against real Ollama) |
| Sandbox success / timeout / reject rate | Sandbox + audit log |
| HITL approve / edit / reject rates | Audit log (quality signal, not a cloud dashboard) |

No metrics UI in v1. Logs are structured (logger, not print). Audit log
records which sources were touched and which generated code/SQL ran.

## Memory

Several stores; they are not one "chat memory dump".

| Kind | Purpose | Lifetime |
|---|---|---|
| LangGraph checkpoint | Graph state + HITL resume (thread) | Per conversation thread, on local disk |
| Working conversation | Last `MAX_PROMPT_TURNS` (default 8) + **summaries** of prior results, not full dataframes | Thread; the LLM prompt is windowed; checkpoint/UI keep the full thread |
| Profile cache | Schema, DQ, EDA keyed by source identity (e.g. path + hash) | Until the source file/connection changes |
| Artifact refs | Paths to tables/charts/reports the UI can render; the LLM prompt lists those paths for follow-ups | Thread + designated output directory |
| Domain RAG index | Business-context chunks | Until context files change; cross-session |
| Audit log | What ran, for humans | Cross-session; not fed wholesale back to the model |

v1 does **not** include a long-term memory of every past analysis, and
does not put entire datasets into agent state. Checkpoints use local
SQLite. After a tool runs, state keeps a short summary, identities, and
artifact paths — not row lists. A value that cannot `json.dumps` without
a default hook is rejected.

## Testing

Tests are written **with each phase**, not postponed to a final phase.

- **Unit tests (from the first feature onward):** tools, validators,
  parsers, sandbox limits, profile cache. `pytest`, small fixtures under
  `tests/fixtures/`.
- **Graph / integration tests (once LangGraph exists):** mock the LLM;
  assert routing, interrupt points, and resume behavior.
- **Eval suite (formalize the existing model comparison script):** fixed
  prompts for latency, JSON/tool-selection correctness, and output-format
  compliance. Re-run on model, prompt, or tool-contract changes. Optional
  in the default unit-test run (needs Ollama).
- **UI:** manual checks for Streamlit in v1; no browser-test suite yet.

The late roadmap item is **hardening the eval harness**, not the start of
testing.

## Docker

**Do not start with Docker.** Dev and v1 run as: host venv + host Ollama
(needed for local GPU/VRAM on this machine) + optional local Postgres.

- v1 sandbox = subprocess, not a container.
- Revisit Docker when packaging for another machine, offering Compose for
  Postgres, or hardening the sandbox into a container.
- Ollama stays on the host even if the app is containerized later; do not
  put `:cloud` models in any compose file.

## Roadmap (phased)

0. Environment setup, model selection — done
1. Data source connection layer + **thin Streamlit shell** (upload data
   files, list sources, separate slot for business-context files)
2. Profiling & automated EDA (schema, DQ, cached results) + Guided-mode
   step gates
3. Agent orchestration core (LangGraph state, MCP-shaped tools, system
   prompt in `src/agent/`)
4. Code generation & subprocess sandbox + Standard-mode code/SQL approval
5. Multi-turn conversation memory (checkpoint + result summaries)
6. Domain context / local RAG (`retrieve_domain_context`)
7. Richer output (charts, then reports; dashboard generation later)
8. Safety hardening (limits, complete audit log; optional container sandbox)
9. Eval suite formalization (unit/integration tests already exist per phase)

## Testing approach (eval)

Model behavior and output correctness are validated with a repeatable
benchmark script that measures latency (time-to-first-token, tokens/sec,
load time) and checks output correctness/format across a fixed set of
prompts for every model in the lineup. This is the seed of the ongoing
eval suite (Phase 9), so regressions are caught when models, prompts, or
tool contracts change.

## Model lineup

All models run locally via Ollama, chosen after benchmarking latency,
throughput, and structured-output (JSON) correctness on the target
hardware (AMD Ryzen 7 8845HS, RTX 4050 6GB VRAM, 32GB RAM).

Model names and roles are configured via environment variables, **not
hardcoded**, so the lineup can change without touching code:

| Role (env var) | Model | Why |
|---|---|---|
| `OLLAMA_MODEL_PRIMARY` | `qwen3.5:4b` | Main model for agent decisions and general analysis prompts. Fastest of the viable candidates (34–36 tok/s), matched every other candidate on correctness across plain-text and JSON/tool-selection tests. |
| `OLLAMA_MODEL_FALLBACK_FAST` | `qwen3.5:2b` | Used when the primary can't be loaded (e.g. memory pressure from another resident model) or when a faster/lighter response is preferable. Smallest footprint (2.7GB) of the tested candidates; confirmed correct and fastest (57 tok/s) on the tool-selection benchmark. |
| `OLLAMA_MODEL_AGENTIC` | `qwen3.6:35b` | Escalation option for harder or longer-horizon agentic tasks. MoE architecture (35B total, 3B active parameters) — large download (~23GB) and slow to load from disk (~30s, since it doesn't fit in 6GB VRAM and runs on CPU/RAM), but competitive generation speed once running. Used deliberately, not as a default, given the load-time cost. |
| `OLLAMA_MODEL_CODING` | `qwen2.5-coder:7b` | Raw code generation only — **not used for structured/JSON output**, since it failed JSON-only formatting instructions (wrapped output in markdown fences) in two separate tests. |

`granite4.2:8b` was benchmarked as a candidate primary (chosen initially for
its tool-calling/structured-output design focus) but was dropped after
testing showed no correctness advantage over `qwen3.5:4b` and notably
slower throughput (15–16 tok/s vs. 34–36 tok/s).

Actual model names live only in `.env` (gitignored); `.env.example`
documents the required variables with placeholder values only.

## Deferred to the implementation plan

These are intentionally not fixed in this spec:

- LangGraph topology (node names, edges, which node calls `interrupt()`)
- Per-tool JSON schemas and which subset ships in which phase
- Checkpointer backend for tests (in-memory `MemorySaver`)
- Embedding model and local vector store
- Exact SQL/Python static deny-lists and size/row thresholds
- Streamlit layout (pages vs single chat; widget structure)
- Chart library
- Whether/when to wrap in-process tools as a standalone MCP server
- Container sandbox design, if/when Docker is revisited
- Contents of the eval prompt corpus

## Key decisions

- 2026-09: Model lineup set from local Ollama benchmarks; `qwen3.5:4b` primary, `granite4.2:8b` dropped.
- 2026-09-04: Spec expanded before the first implementation plan: Streamlit as v1 UI; HITL modes and interrupt points; LangGraph orchestrates, MCP-shaped tools do not decide; local RAG for domain documents only; tests-with-each-phase; Docker deferred; subprocess sandbox in v1; single-user local app.
- 2026-09-13: Checkpoints persist with SqliteSaver. Tool results in graph state are summaries + artifact paths, not tables.
- 2026-09-13: LLM prompt keeps the last `MAX_PROMPT_TURNS` (default 8). Checkpointed chat history is not trimmed.
- 2026-09-13: Follow-up turns see the last tool summary and artifact paths (paths only, not file contents).
- 2026-09-13: `retrieve_domain_context` is an in-process TF-IDF tool over `CONTEXT_DIR`. Data files are never indexed.
- 2026-09-13: Domain RAG index persists under `CACHE_DIR` and rebuilds when context files change.
