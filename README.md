# rex

A wake word detection system with speech recognition capabilities.

Rex is being ported from Python to Go — see `GO_MIGRATION_PLAN.md` for status.
The Python sources under `src/` remain authoritative until each migration
stage is marked complete.

## Setup (Go)

1. **System dependencies** (macOS):
   ```bash
   brew install portaudio pkg-config espeak-ng onnxruntime whisper-cpp
   ```
   `onnxruntime` is required for wake-word detection. On Linux the
   equivalent is a distribution package or the release tarball from
   https://github.com/microsoft/onnxruntime/releases; point
   `ONNXRUNTIME_SHARED_LIBRARY_PATH` at the resulting `libonnxruntime.so`.

   `whisper-cpp` provides the `whisper-server` binary the Go STT package
   shells out to. On Linux either build from source
   (https://github.com/ggml-org/whisper.cpp) and put `whisper-server`
   on `$PATH`, or set `stt.binary` in `settings.toml`.
2. **Wake-word models**: the three ONNX files needed for wake-word
   detection live under `models/`:
   - `models/shared/melspectrogram.onnx`
   - `models/shared/embedding_model.onnx`
   - `models/shared/silero_vad.onnx`
   - `models/wake_word_models/hey_rex/hey_rex.onnx`

   The `models/wake_word_models/hey_rex/` directory is already checked
   into the repo. The three files under `models/shared/` are not, because
   they are redistributable third-party assets; you can copy them from a
   Python environment that has `openwakeword` and `silero_vad` installed:

   ```bash
   uv sync                     # populates .venv
   mkdir -p models/shared
   cp .venv/lib/python*/site-packages/openwakeword/resources/models/melspectrogram.onnx models/shared/
   cp .venv/lib/python*/site-packages/openwakeword/resources/models/embedding_model.onnx models/shared/
   cp .venv/lib/python*/site-packages/silero_vad/data/silero_vad.onnx models/shared/
   ```

   If these files are missing the binary still boots, but logs
   `wake-word detection disabled` and skips the wake-word pipeline.
3. **Whisper model** (optional, for speech-to-text). Download a GGML
   model from
   https://huggingface.co/ggerganov/whisper.cpp/tree/main — `ggml-small.en.bin`
   (~465 MB) is the size the Python port used. Point `settings.toml`
   at it:
   ```toml
   [stt]
   model_path = "/absolute/path/to/ggml-small.en.bin"
   # language = "en"      # optional, defaults to en
   # binary  = "whisper-server"  # override if not on $PATH
   ```
   When `stt.model_path` is empty Rex boots with STT disabled and logs
   `stt disabled`; the binary otherwise launches `whisper-server` as a
   sidecar so the model stays resident between utterances.
4. **Build and run** the Go binary:
   ```bash
   make run          # runs cmd/rex
   make build        # produces bin/rex
   make test         # runs go test ./...
   ```
   Or without make:
   ```bash
   go run ./cmd/rex
   go build -o bin/rex ./cmd/rex
   go test ./...
   ```

The Go binary currently implements Stages 1–4 of the migration plan: it
loads `settings.toml`, opens the PortAudio output stream, initialises
the wake-word pipeline (openWakeWord + Silero VAD), spawns
`whisper-server` for STT when configured, and waits for Ctrl-C before
cleaning up.

## Setup (Python, legacy)

1. **Install dependencies** (uses uv for package management):
   ```bash
   uv sync
   ```

2. **First-time model setup**:
   The required openwakeword models (~15MB) will be automatically downloaded on first use. Alternatively, you can manually download them:
   ```bash
   uv run python -c "from openwakeword.utils import download_models; download_models()"
   ```

3. Run `brew install espeak-ng portaudio` to get everything working.

## Usage

### Test a pre-trained wake word model:
```bash
uv run wake_word test-pretrained wake_word_training_files/hey_recks.onnx
```

The command will start listening for the wake word. Speak clearly into your microphone to test detection.

Press `Ctrl+C` to stop listening.

### Options:
- `--threshold`: Detection sensitivity (0.0-1.0, default: 0.5)
- `--chunk-size`: Audio chunk size in samples (default: 1280)

## Project Structure

- `src/wake_word/` - Wake word detection implementation
  - `cli.py` - Command-line interface
  - `wake_word_listener.py` - Real-time wake word detection
  - `model_utils.py` - Model management and auto-download
- `wake_word_training_files/` - Custom trained wake word models

