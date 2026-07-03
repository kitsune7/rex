# rex

A wake word detection system with speech recognition and voice assistant capabilities.

## Setup

1. **Install dependencies** (uses uv for package management):
   ```bash
   uv sync
   ```

2. **System dependencies** (macOS):
   ```bash
   brew install espeak-ng portaudio
   ```

3. **Configuration** — copy the example settings and edit as needed:
   ```bash
   cp settings.example.toml settings.toml
   ```

4. **First-time model setup** — openwakeword models (~15MB) download automatically on first use. To fetch them manually:
   ```bash
   uv run python -c "from openwakeword.utils import download_models; download_models()"
   ```

## Usage

### Voice assistant (local)

Run the full Rex voice assistant on your machine:

```bash
uv run rex
```

Press `Ctrl+C` to exit.

### Laptop server

Run the FastAPI/WebSocket server (used by the laptop client):

```bash
uv run rex-server
```

Options:
- `--host` — bind address (default: `0.0.0.0`)
- `--port` — port (default: `8765`)
- `--reload` — enable auto-reload for development

### Wake word tester

Test wake word detection with your microphone:

```bash
uv run wake_word
```

Options:
- `--model` — model name under `models/wake_word_models/` (default: `hey_rex`)
- `--threshold` — detection sensitivity, 0.0–1.0 (default: `0.5`)

Press `Ctrl+C` to stop listening.

## Development

### Run tests

```bash
uv run pytest
```

Integration evals (require a running LLM at the configured `api_base`; skipped in normal CI runs):

```bash
uv run pytest src/tests/test_scenario_evals.py -m integration
```

Or use the standalone script (supports `REX_LLM_API_BASE` / `REX_LLM_MODEL` overrides):

```bash
uv run python scripts/run_integration_evals.py
```

See [docs/forge-proxy-experiment.md](docs/forge-proxy-experiment.md) for Forge proxy setup.

### Lint and format

```bash
uv run ruff check
uv run ruff format
```

## Project Structure

- `src/rex/` — main voice assistant CLI and app wiring
- `src/agent/` — LLM agent, tools, and evals
- `src/server/` — FastAPI laptop server
- `src/wake_word/` — wake word detection
- `src/stt/` — speech-to-text
- `src/tts/` — text-to-speech
- `src/audio/` — audio capture and playback
- `scripts/` — standalone utility scripts
