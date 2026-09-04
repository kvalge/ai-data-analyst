---
alwaysApply: true
---

## Core Rules
- With code writing, move on step by step, only one task/functionality at a time.
- After implementing a task, stop and ask for review before continuing.
- If changes are requested, apply them and ask for review again.
- If approved, ask whether to commit. If yes, write a commit message and commit, then move to the next task. If no, move to the next task without committing.

## Code Quality
- Write clean, modular, and maintainable Python code.
- Prefer simplicity over complexity.
- Prefer small functions with a single responsibility.
- Avoid duplication and hardcoded values where possible.
- Use type hints where reasonable.
- Write a short docstring for every public function.
- Add short, clear comments where they improve code understanding.
- Use logging, not print statements, for anything beyond quick local debugging.
- Organize code into purpose-based subfolders (e.g. `tools/`, `agent/`, `execution/` etc) rather than placing most files in one flat folder — mirror the project's module boundaries in the folder structure.

## Project Consistency
- Update `PROJECT_SPEC.md` whenever a real architectural or scope decision changes, it should always describe the project as it actually is right now.
- In case of `PROJECT_SPEC.md` changes, add a short "Key decisions" section at the bottom of the spec, not a full diff log — just one-line entries for major pivots (model changes, architecture changes etc), each pointing to roughly when it happened.
- Write separate commit message for each meaningful spec change separately, with a clear commit message (git commit -m "spec: message text").
- Update `.cursor/PLAN.md` (what is done or changed) *before* asking permission to proceed — the plan file must always reflect true current state, even if the session pauses mid-step.
- If `PROJECT_SPEC.md` changes, review `.cursor/PLAN.md` for consistency and update it if the change affects remaining steps.
- Keep `requirements.txt` updated whenever dependencies change.
- Keep `.env.example` updated with all required environment variables, using placeholder values (never real secrets or real model names).
- Keep `.gitignore` updated to exclude local files, caches, and sensitive data.
- Keep `README.md` updated when functionality or workflow changes.
- Write or update a test for new functionality where practical.

## Security
- Store all sensitive environment variables (API keys, tokens, webhook URLs, DB credentials, etc.) in `.env`.
- Never commit `.env` to version control.
- Ensure `.env` is listed in `.gitignore`.

## Data Safety
- Never send data (file contents, query results, samples) to any non-local API or service — all LLM inference stays on the local Ollama instance.
- Never use Ollama's `:cloud` model variants — only models that run fully locally.
- Enforce database access as read-only at the connection/role level, not just in application logic.
- Never load an entire large dataset into the LLM's prompt context — work from schema + samples; execute generated code against full data separately, outside the prompt.
- All LLM-generated code runs only inside the sandboxed execution environment — never `exec`/`eval` generated code directly in the main application process.
- Log what data sources were accessed and what generated code/queries ran, for auditability.

## LLM Output Handling
- Never assume LLM output is well-formed — validate/parse structured output (JSON, code) defensively; strip common wrapping artifacts (e.g. markdown fences) before parsing.
- On malformed output, retry once with a stricter instruction before falling back to a different model or failing visibly.
- Never silently swallow a parsing failure — surface it, don't guess at intent.

## Execution Safety (Sandboxing)
- LLM-generated code must run in an isolated environment (subprocess or container) with time and resource limits.
- The sandbox must never have write access outside its designated working area.

## Idempotency
- Ensure scripts can be run manually without side effects on previous successful runs.
- At the top of every manually runnable file, add a comment with the command used to run it.

## Architecture
- Separate concerns.
- Follow best practices of Python project structure.
- Never hardcode model names in code — always read them from environment variables.
