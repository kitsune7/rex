# Rex Voice Assistant — Entry Point

## Build

```bash
go build ./cmd/rex
```

## Run

```bash
./rex
# or
go run ./cmd/rex
```

## Prerequisites

- **LLM server** running at `localhost:1234` (e.g. LM Studio or llama.cpp server)
- **PortAudio** system library installed (`brew install portaudio` on macOS)
- **Whisper model** downloaded to `models/whisper/ggml-small.bin`
- **Piper TTS model** at `models/piper/en_US-lessac-medium.onnx`
- **ONNX Runtime** shared library available for the wake word detector
- **Wake word ONNX models** in `models/wake_word_models/hey_rex/`
- **SQLite** (bundled via go-sqlite3, requires cgo)

## Shutdown

Press Ctrl+C (SIGINT) or send SIGTERM to shut down gracefully.
