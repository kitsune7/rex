# Forge Proxy Experiment

This guide implements [Plan 2](forge-reliability-plan.md#plan-2-forge-proxy-experiment) from the Forge reliability plan: run [Forge](https://github.com/antoinezambelli/forge) as an OpenAI-compatible proxy in front of Rex's configured LLM backend, then compare reliability using the same scenario evals.

## Why

Rex already reads `llm.api_base` from `settings.toml`. Forge sits between Rex and the local model server and adds guardrails (rescue parsing, retry nudges, response validation) without changing Rex's LangGraph agent or voice state machine.

## Prerequisites

1. A running OpenAI-compatible LLM server (the same one Rex uses today).
2. Forge installed in a separate environment:

```bash
pip install forge-guardrails
# or: pip install git+https://github.com/antoinezambelli/forge.git
```

3. Rex dependencies installed (`uv sync`).

## Experiment steps

### 1. Start your LLM backend

Use whatever you normally run locally. The default Rex config expects:

```toml
[llm]
api_base = "http://localhost:1234/v1"
model = "your-model-name"
```

### 2. Start Forge proxy in front of it

Point Forge at your real backend and expose a proxy port:

```bash
python -m forge.proxy \
  --backend-url http://localhost:1234 \
  --port 8081
```

Forge speaks OpenAI chat-completions at `http://localhost:8081/v1`.

### 3. Run deterministic evals (fake model, no LLM required)

These are fast regression tests that simulate transcripts with a scripted model:

```bash
uv run pytest src/tests/test_scenario_evals.py -q
```

### 4. Baseline: run integration evals against the direct backend

With your LLM server running:

```bash
uv run python scripts/run_integration_evals.py
```

Or override explicitly:

```bash
REX_LLM_API_BASE=http://localhost:1234/v1 \
REX_LLM_MODEL=your-model-name \
uv run python scripts/run_integration_evals.py
```

### 5. Compare: run the same evals through Forge proxy

Leave Forge proxy running and point Rex at the proxy URL:

```bash
REX_LLM_API_BASE=http://localhost:8081/v1 \
REX_LLM_MODEL=your-model-name \
uv run python scripts/run_integration_evals.py
```

Alternatively, update `settings.toml` temporarily:

```toml
[llm]
api_base = "http://localhost:8081/v1"
model = "your-model-name"
```

Then run Rex server or integration evals as usual.

## Success criteria

Compare direct vs proxy runs on the same model:

| Signal | What to look for |
| --- | --- |
| Malformed tool calls | Fewer failed or empty tool turns through proxy |
| Voice errors | Fewer generic "encountered an error" responses in manual voice testing |
| Latency | No meaningful slowdown on simple timer/reminder requests |
| Confirmation flow | `create_reminder` still pauses for human confirmation |

## Optional: pytest integration marker

Live-model evals can also be invoked manually with pytest (they are skipped by default in CI):

```bash
REX_LLM_API_BASE=http://localhost:8081/v1 \
uv run pytest src/tests/test_scenario_evals.py -m integration -q
```

## Next steps

- If proxy mode helps, document it as an optional deployment path and keep `settings.example.toml` pointed at the proxy URL as a commented example.
- If proxy mode does not help enough, continue with Rex-native guardrails (Plan 3 in the reliability plan) while keeping the scenario eval suite as the baseline.
