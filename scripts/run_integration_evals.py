#!/usr/bin/env python3
"""Run Rex scenario evals against a live OpenAI-compatible LLM backend.

Use this to compare direct model access versus Forge proxy mode:

    # Direct backend (default settings.toml api_base)
    uv run python scripts/run_integration_evals.py

    # Forge proxy in front of the same backend
    REX_LLM_API_BASE=http://localhost:8081/v1 uv run python scripts/run_integration_evals.py

Requires a running LLM server at the configured api_base and optionally Forge
proxy at http://localhost:8081/v1.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.evals.integration import run_integration_evals  # noqa: E402
from rex.settings import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-base",
        default=os.environ.get("REX_LLM_API_BASE"),
        help="Override llm.api_base (or set REX_LLM_API_BASE)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("REX_LLM_MODEL"),
        help="Override llm.model (or set REX_LLM_MODEL)",
    )
    args = parser.parse_args()

    settings = load_settings(ROOT / "settings.toml")
    api_base = args.api_base or settings.llm.api_base
    model = args.model or settings.llm.model

    print(f"Running integration evals against {api_base} (model={model})")
    results = run_integration_evals(api_base=api_base, model=model)
    passed = sum(1 for result in results if result.passed)
    total = len(results)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        detail = f" — {result.error}" if result.error else ""
        print(f"[{status}] {result.scenario_id}{detail}")

    print(f"\n{passed}/{total} scenarios passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
