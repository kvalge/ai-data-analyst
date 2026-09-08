# compare_ollama_models.py

"""Benchmark latency and compare outputs of the project's Ollama models.

Reads OLLAMA_HOST and the four OLLAMA_MODEL_* names from .env, sends the
same prompts to every model, and prints timing plus the generated text.

Usage:
    python scripts/compare_ollama_models.py
    python scripts/compare_ollama_models.py --quick
    python scripts/compare_ollama_models.py --prompt "Summarize this CSV schema: ..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL_ROLES = (
    ("PRIMARY", "OLLAMA_MODEL_PRIMARY"),
    ("FALLBACK_FAST", "OLLAMA_MODEL_FALLBACK_FAST"),
    ("AGENTIC", "OLLAMA_MODEL_AGENTIC"),
    ("CODING", "OLLAMA_MODEL_CODING"),
)

PROMPTS = (
    {
        "name": "speed",
        "text": "Reply with exactly one word: OK",
    },
    {
        "name": "data_analysis",
        "text": (
            "A shop sold 120, 95, 130, 80, and 150 units over five days. "
            "Give the mean, the day with the largest drop vs the previous day, "
            "and one sentence on what a manager should look at next. "
            "Be concise."
        ),
    },
    {
        "name": "pandas_code",
        "text": (
            "Write a short pandas snippet that loads sales.csv with columns "
            "date, region, revenue; parses date; and prints revenue by region "
            "and by month. Return only the code."
        ),
    },
    {
        "name": "agent_plan",
        "text": (
            "A user uploaded a messy Excel file of customer orders and asked "
            "'why did revenue fall last quarter?'. List 5 numbered analysis "
            "steps a data-analyst agent should take before answering. "
            "One line per step."
        ),
    },
)

QUICK_PROMPTS = PROMPTS[:1]


@dataclass
class ModelRun:
    role: str
    model: str
    prompt_name: str
    output: str = ""
    error: str = ""
    time_to_first_token_s: float | None = None
    total_s: float = 0.0
    load_s: float | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_s: float | None = None
    tokens_per_s: float | None = None
    done_reason: str = ""
    stats: dict = field(default_factory=dict)


def _ns_to_s(value: object) -> float | None:
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return value / 1_000_000_000


def load_models() -> list[tuple[str, str]]:
    missing = [env_key for _, env_key in MODEL_ROLES if not os.getenv(env_key, "").strip()]
    if missing:
        raise SystemExit(
            "Missing model names in .env: " + ", ".join(missing) + "\n"
            "Copy .env.example to .env and set the four OLLAMA_MODEL_* values."
        )
    return [(role, os.getenv(env_key, "").strip()) for role, env_key in MODEL_ROLES]


def generate(
    host: str,
    model: str,
    prompt: str,
    timeout_s: int,
) -> tuple[str, dict, float | None]:
    """Stream /api/generate; return (text, final stats, time-to-first-token)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "options": {
            "temperature": 0,
        },
    }
    request = urllib.request.Request(
        url=f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    chunks: list[str] = []
    stats: dict = {}
    t0 = time.perf_counter()
    ttft: float | None = None

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            event = json.loads(line)
            piece = event.get("response") or ""
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            chunks.append(piece)
            if event.get("done"):
                stats = event
                break

    return "".join(chunks).strip(), stats, ttft


def run_one(
    host: str,
    role: str,
    model: str,
    prompt_name: str,
    prompt: str,
    timeout_s: int,
) -> ModelRun:
    result = ModelRun(role=role, model=model, prompt_name=prompt_name)
    started = time.perf_counter()
    try:
        output, stats, ttft = generate(host, model, prompt, timeout_s)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        result.error = f"HTTP {exc.code}: {body[:400]}"
        result.total_s = time.perf_counter() - started
        return result
    except Exception as exc:  # noqa: BLE001 — report any transport/model failure
        result.error = f"{type(exc).__name__}: {exc}"
        result.total_s = time.perf_counter() - started
        return result

    result.output = output
    result.stats = stats
    result.time_to_first_token_s = ttft
    result.total_s = time.perf_counter() - started
    result.load_s = _ns_to_s(stats.get("load_duration"))
    result.prompt_eval_count = stats.get("prompt_eval_count")
    result.eval_count = stats.get("eval_count")
    result.eval_s = _ns_to_s(stats.get("eval_duration"))
    result.done_reason = str(stats.get("done_reason") or "")
    if result.eval_count and result.eval_s:
        result.tokens_per_s = result.eval_count / result.eval_s
    return result


def fmt_s(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}s"


def fmt_n(value: int | float | None, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def print_table(runs: list[ModelRun], prompt_name: str) -> None:
    rows = [r for r in runs if r.prompt_name == prompt_name]
    headers = (
        "Role",
        "Model",
        "TTFT",
        "Total",
        "Load",
        "Tokens",
        "tok/s",
        "Status",
    )
    table: list[tuple[str, ...]] = [headers]
    for run in rows:
        status = run.error.split(":")[0] if run.error else (run.done_reason or "ok")
        table.append(
            (
                run.role,
                run.model,
                fmt_s(run.time_to_first_token_s),
                fmt_s(run.total_s),
                fmt_s(run.load_s),
                fmt_n(run.eval_count),
                fmt_n(run.tokens_per_s),
                status,
            )
        )

    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    print()
    print(f"Prompt: {prompt_name}")
    print("-" * (sum(widths) + 3 * (len(headers) - 1)))
    for i, row in enumerate(table):
        line = " | ".join(cell.ljust(widths[j]) for j, cell in enumerate(row))
        print(line)
        if i == 0:
            print("-+-".join("-" * w for w in widths))


def print_outputs(runs: list[ModelRun], prompt_name: str) -> None:
    print()
    print("Outputs")
    print("-------")
    for run in runs:
        if run.prompt_name != prompt_name:
            continue
        print()
        print(f"[{run.role}] {run.model}")
        if run.error:
            print(run.error)
        elif run.output:
            print(run.output)
        else:
            print("(empty response)")


def print_speed_summary(runs: list[ModelRun]) -> None:
    successful = [r for r in runs if not r.error]
    if not successful:
        return

    print()
    print("Speed ranking (successful runs, by total wall time)")
    print("---------------------------------------------------")
    ranked = sorted(successful, key=lambda r: r.total_s)
    for i, run in enumerate(ranked, start=1):
        tps = f", {run.tokens_per_s:.1f} tok/s" if run.tokens_per_s else ""
        print(
            f"{i}. {run.role:13} {run.model:22} "
            f"{run.prompt_name:14} {run.total_s:.2f}s{tps}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the one-word speed prompt.",
    )
    parser.add_argument(
        "--prompt",
        help="Use a single custom prompt instead of the built-in set.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Send a tiny prompt to each model first so load time is excluded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    models = load_models()

    if args.prompt:
        prompts = [{"name": "custom", "text": args.prompt}]
    elif args.quick:
        prompts = list(QUICK_PROMPTS)
    else:
        prompts = list(PROMPTS)

    print("Ollama model comparison")
    print(f"Host: {host}")
    print("Models:")
    for role, model in models:
        print(f"  {role:13} {model}")
    print(f"Prompts: {', '.join(p['name'] for p in prompts)}")
    print("Options: temperature=0, think=false, stream=true")

    if args.warmup:
        print("\nWarming up models...")
        for role, model in models:
            print(f"  loading {model} ({role})...", end=" ", flush=True)
            warm = run_one(host, role, model, "warmup", "Say hi.", args.timeout)
            if warm.error:
                print(warm.error)
            else:
                print(f"ok ({fmt_s(warm.total_s)})")

    runs: list[ModelRun] = []
    total = len(models) * len(prompts)
    n = 0
    for prompt in prompts:
        for role, model in models:
            n += 1
            print(
                f"\n[{n}/{total}] {role} / {model} <- {prompt['name']}...",
                flush=True,
            )
            run = run_one(host, role, model, prompt["name"], prompt["text"], args.timeout)
            runs.append(run)
            if run.error:
                print(f"  failed in {fmt_s(run.total_s)}: {run.error}")
            else:
                tps = f", {run.tokens_per_s:.1f} tok/s" if run.tokens_per_s else ""
                print(
                    f"  {fmt_s(run.total_s)} total, "
                    f"TTFT {fmt_s(run.time_to_first_token_s)}{tps}"
                )

    for prompt in prompts:
        print()
        print("=" * 72)
        print(f"PROMPT ({prompt['name']})")
        print(prompt["text"])
        print_table(runs, prompt["name"])
        print_outputs(runs, prompt["name"])

    print()
    print("=" * 72)
    print_speed_summary(runs)

    if any(run.error for run in runs):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
