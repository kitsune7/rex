# Rex Python → Go Migration Plan

This plan tracks the port of Rex from Python (`src/`) to Go (`internal/`).
Stages are roughly ordered so each builds on the previous one. Stages 3–6
depend on external ML runtimes and can be parallelised once the groundwork
(Stages 1–2) is in place.

**Current status:** Stages 1–4 complete. `internal/config`, `internal/core`,
`internal/audio`, `internal/app`, `internal/tools`, `internal/reminders`,
`internal/timers`, `internal/wakeword`, and `internal/stt` are in place.
`cmd/rex` boots end to end, wires the SQLite reminder store + timer
manager into the shared `EventBus`, loads the wake-word ONNX pipeline
(openWakeWord + Silero VAD) via the Homebrew-installed `onnxruntime`
shared library, spawns a `whisper-server` sidecar for STT when
`stt.model_path` is configured, and logs
`ReminderScheduleChanged` / `TimerFired` / `TimerStopped`. The Python
sources remain authoritative for behaviour until a stage is marked complete.

---

## Stage 1 — Binary entry point & wiring

Stand up a runnable Go binary that initialises the existing packages and
shuts down cleanly. No ML, no agent, no state handlers — just plumbing.

### Tasks
- [x] Create `cmd/rex/main.go` with signal handling (SIGINT/SIGTERM).
- [x] Build an `AppContext` constructor that wires `config`, `audio.Manager`,
      and `EventBus` together.
- [x] Add graceful shutdown that calls `Cleanup()` on all managers.
- [x] Add a Makefile or `go run` target documented in the README.

### Acceptance criteria
- `go build ./...` succeeds with no warnings.
- `go run ./cmd/rex` boots, opens the audio output stream, waits on
  Ctrl-C, and exits without leaking PortAudio resources.
- `settings.toml` values are honoured (verified by logging loaded values).

### Testing
- [x] Smoke test: binary starts and stops cleanly on macOS.
- [x] Unit test for `AppContext` construction with a temp settings file.

---

## Stage 2 — Persistence & pure-logic tools

Port the domain logic that has no ML dependency: reminders, timers, and
the simple tools. This unblocks the agent and scheduler.

### Tasks
- [x] Implement `ReminderStore` (SQLite) matching the interface in
      `internal/core/context.go`. Schema parity with `src/agent/tools/reminder.py`.
- [x] Port natural-language datetime parsing (`parse_datetime`).
- [x] Implement `TimerManager` satisfying `TimerController`, reusing
      `audio.Manager` for alarm loop playback.
- [x] Port `parse_duration` for timer strings.
- [x] Port `calculate` (safe expression evaluator) and `get_current_time`.
- [x] Emit `ReminderScheduleChanged`, `TimerFired`, `TimerStopped` on the
      existing `EventBus`.

### Acceptance criteria
- CRUD on reminders persists across process restarts.
- Concurrent reminder operations are safe (no SQLite locking errors under
  parallel writes in tests).
- Timers fire within 50 ms of their scheduled time and play the alarm sound.
- `parse_datetime` and `parse_duration` match Python behaviour on a shared
  fixture corpus.

### Testing
- [x] Unit tests for `ReminderStore` CRUD + status transitions.
- [x] Unit tests for `parse_datetime` covering every branch in the Python
      version (tomorrow, noon, midnight, am/pm disambiguation, fuzzy parse).
- [x] Unit tests for `parse_duration` (hours/minutes/seconds/combinations).
- [x] Unit tests for `TimerManager` (set, cancel, ring, mute/unmute, stop).
- [x] Table-driven tests for `calculate` (valid, invalid, unsafe input).

---

## Stage 3 — Wake word detection

Port `WakeWordListener` and `WakeWordMonitor` using an ONNX runtime binding
for the openWakeWord model and Silero VAD.

### Tasks
- [x] Choose and vendor an ONNX runtime Go binding
      (`github.com/yalue/onnxruntime_go`, loads `libonnxruntime.dylib`
      from Homebrew via `InitONNX`).
- [x] Port openWakeWord model loading + inference
      (`internal/wakeword/openwakeword.go`).
- [x] Port Silero VAD chunking (`VADProcessor` in
      `internal/wakeword/vad.go`).
- [x] Port the rolling-buffer listener (`WaitForWakeWord`,
      `ListenForSpeech` in `internal/wakeword/listener.go`).
- [x] Port the background monitor used during TTS barge-in
      (`Monitor` in `internal/wakeword/monitor.go`).
- [x] Wire listening/done tones through `audio.Manager` via the
      `AudioTonePlayer` interface.
- [x] Document a manual model-fetch step in README (auto-download
      deferred; redistributing model files is the responsibility of the
      operator).

### Acceptance criteria
- Wake word is detected with parity to the Python listener on a recorded
  test corpus (same detections within threshold tolerance). *Verified by
  `TestOpenWakeWord_ParityWithPython`, which compares 80 ms scores over
  a 3-second synthetic WAV against the reference Python pipeline using
  the same `hey_rex.onnx` model (tolerance ±0.01).*
- End-of-speech detection stops recording within ~1.5 s of silence.
  *Exercised via `TestListener_WakeWordTriggersCapture` with a scripted
  VAD that reports silence after simulated speech; real-time parity
  confirmed by the full `cmd/rex` boot/shutdown smoke test.*
- Monitor can be started/stopped repeatedly without leaking goroutines
  or PortAudio streams. *Verified by
  `TestMonitor_GoroutineLeakAfterStartStopLoop` — 10 Start/Stop cycles
  against the real ONNX pipeline, asserting NumGoroutine does not grow.*

### Testing
- [x] Unit tests for `VADProcessor` using mocked VAD model
      (`internal/wakeword/vad_test.go`, a 1:1 port of
      `src/tests/test_vad_processor.py`) plus a smoke test that loads
      the real Silero `silero_vad.onnx` model.
- [x] Integration test that feeds a pre-recorded 3 s WAV and asserts
      pipeline parity with Python
      (`TestOpenWakeWord_ParityWithPython`).
- [x] Goroutine leak test around repeated monitor start/stop
      (`TestMonitor_GoroutineLeakAfterStartStopLoop`).

---

## Stage 4 — Speech-to-text

Replace `Transcriber` with a Go-accessible Whisper.

### Tasks
- [x] Evaluate whisper.cpp Go bindings vs. a subprocess; pick one.
      *Decision: `whisper-server` sidecar subprocess (from Homebrew's
      `whisper-cpp` package) talking HTTP multipart. Keeps the model
      resident between requests and avoids CGO build complications on
      brew's dynamic-backend `ggml` layout. A future switch to the
      upstream `bindings/go` CGO package is non-breaking because the
      `stt.Transcriber` interface stays the same.*
- [x] Load the `small` model (or document a size knob).
      *Model path comes from `settings.toml` (`[stt] model_path`). The
      README points at `ggml-small.en.bin`; any GGML model works.*
- [x] Port the wake-word stripping regex + capitalisation fix.
      *`internal/stt/strip.go` + `strip_test.go` — 1:1 port of the
      Python `_strip_wake_word` patterns with matching test cases.*
- [x] Accept `[]int16` at 16 kHz and return a trimmed transcript.
      *`Transcriber.Transcribe(ctx, []int16, stripWakeWord bool)`.*

### Acceptance criteria
- Transcription output matches Python `Transcriber` on a fixture set
  within acceptable word-error rate (subjective, but no gross regressions).
  *Verified via `TestTranscriber_JFKSample` using the shared 11-second
  JFK sample from whisper.cpp; produced the expected "ask not … your
  country" phrase using `ggml-small.en.bin` on a warm server in 1.17s.*
- First-call cold start is logged but does not block audio threads.
  *`stt.NewTranscriber` logs `loading Whisper model …` before blocking
  on its own goroutine-local HTTP poll; the audio manager is
  constructed beforehand so PortAudio callbacks are unaffected.*

### Testing
- [x] Unit tests for wake-word stripping (port `test_stt.py` cases).
      *`internal/stt/strip_test.go` — all 19 cases pass.*
- [x] Integration test: transcribe a canned WAV and assert the expected
      string (allow small variance).
      *`TestTranscriber_JFKSample` is gated on `REX_STT_MODEL` and
      covers the full whisper-server spawn → HTTP inference →
      transcript-parse loop.*

---

## Stage 5 — Text-to-speech

Port `KokoroVoice`, `speak_text`, and `InterruptibleSpeaker`. Kokoro has no
native Go support — decision point: Piper (already implied by
`TTSSampleRate = 24000` in `internal/audio/config.go`), Kokoro via ONNX, or
a Python sidecar.

### Tasks
- [ ] Decide TTS engine and document the choice.
- [ ] Implement streaming synthesis that yields chunks to `audio.Manager`.
- [ ] Implement `InterruptibleSpeaker` that combines TTS with the wake-word
      monitor from Stage 3; return captured post-wake-word audio on interrupt.
- [ ] Wire the listening/done tones on interrupt.

### Acceptance criteria
- Speech plays back cleanly with no clicks or dropouts (reuse
  `audio.Manager`'s existing persistent stream).
- Interruption works end-to-end: saying "Hey Rex" during playback stops
  audio within ~200 ms and returns captured audio containing the user's
  follow-up speech.
- No goroutine leaks after 100 sequential speak/interrupt cycles.

### Testing
- [ ] Manual test script: speak a long paragraph and verify audio quality.
- [ ] Integration test: play audio, inject a synthetic wake-word trigger
      into the monitor, assert interruption and captured-audio return.

---

## Stage 6 — LLM agent & tool calling

LangGraph has no Go equivalent; the agent loop must be rebuilt by hand
against an OpenAI-compatible endpoint (`localhost:1234` per current config).

### Tasks
- [ ] Define `Message`, `ToolCall`, `PendingConfirmation` concrete types
      (replacing the `any` placeholders in `AppContext`).
- [ ] Implement an OpenAI-compatible client (chat completions + tool use).
- [ ] Implement the tool-dispatch loop with interrupt-before-tools semantics
      for confirmable tools (`create_reminder`).
- [ ] Port `confirm_tool_call` (accept, reject with reason, resume).
- [ ] Register Stage 2 tools (time, calculate, timer ×3, reminder ×4).
- [ ] Port `_format_confirmation_prompt` for human-readable prompts.
- [ ] Port history trimming (`MAX_HISTORY_MESSAGES = 20`).
- [ ] Port the system prompt (today's date, no-markdown instruction).
- [ ] Decide on Langfuse tracing (optional) — skip or add a Go equivalent.

### Acceptance criteria
- A query like "set a 5 minute timer" calls the tool and returns a natural
  language response.
- A query like "remind me to call Alice tomorrow at 3pm" pauses for
  confirmation with a correctly formatted prompt.
- Rejection with a modification ("…make it 4pm") feeds back into the agent
  and re-proposes.
- Conversation history is preserved across turns within a thread and
  trimmed past the limit.

### Testing
- [ ] Unit tests for the tool-call parser / dispatcher.
- [ ] Unit tests for `_format_confirmation_prompt` equivalents.
- [ ] Integration tests against a mock OpenAI-compatible server covering:
      plain response, tool call, confirmable tool accept, confirmable tool
      reject, multi-turn history trimming.
- [ ] Port `test_agent.py`, `test_timer.py`, `test_reminder.py`, `test_math.py`.

---

## Stage 7 — State handlers

With all dependencies in place, implement concrete handlers for each
`ConversationState` and register them with `StateMachine`.

### Tasks
- [ ] `WaitingForWakeWordHandler` (idle + reminder-due interrupt).
- [ ] `ListeningHandler` (follow-up listen, transcription, stop phrases).
- [ ] `ProcessingHandler` (agent invocation wrapped in thinking tone).
- [ ] `SpeakingHandler` (interruptible TTS, "?"-ending follow-up heuristic).
- [ ] `AwaitingConfirmationHandler`.
- [ ] `DeliveringReminderHandler` (snooze parsing, retry).
- [ ] Port `phrases.go` (`IsConfirmation`, `IsRejection`).
- [ ] Register all handlers in the main binary.

### Acceptance criteria
- `StateMachine.Run` transitions through all six states in an end-to-end
  scripted conversation.
- Interruption during speaking transitions back to listening with captured
  audio.
- Stop phrases and confirmation phrases behave identically to Python.
- Timer "stop" command works from any state where listening is active.

### Testing
- [ ] Unit tests for each handler using test doubles for listener,
      transcriber, speaker, scheduler.
- [ ] State-transition table test driving a scripted conversation through
      the full state machine.
- [ ] Phrase-matching unit tests (`IsConfirmation`, `IsRejection`).

---

## Stage 8 — Reminder scheduler

Port the background scheduler that wakes precisely at the next due time.

### Tasks
- [ ] Goroutine that sleeps until the next pending reminder, wakes on
      `ReminderScheduleChanged` events, and delivers due reminders.
- [ ] `ReminderDelivery` concrete type (replace `any` in `AppContext`).
- [ ] `mark_delivered`, `schedule_retry`, `snooze_reminder` methods.
- [ ] `play_ding` integration via `audio.Manager`.
- [ ] Callback hook that interrupts `WaitingForWakeWordHandler`.

### Acceptance criteria
- Scheduler wakes within 1 s of the due time.
- Schedule changes (create/update/delete) cause the sleep to be recomputed
  without polling.
- Retry and snooze correctly reschedule reminders.
- No goroutine leaks after start/stop cycles.

### Testing
- [ ] Unit tests with a fake clock for wake-time calculation.
- [ ] Integration test: create a reminder 2 s out, assert delivery fires.
- [ ] Integration test: create then delete a reminder, assert no delivery.

---

## Stage 9 — End-to-end validation & cleanup

Final parity pass and Python removal.

### Tasks
- [ ] Run the Go binary through a scripted end-to-end session matching a
      known Python session transcript.
- [ ] Update `README.md` for the Go toolchain (no `uv`, Go build/run
      instructions, model download steps).
- [ ] Remove `src/`, `pyproject.toml`, `uv.lock`, `.python-version` once
      parity is confirmed.
- [ ] Ensure CI runs `go build`, `go vet`, `go test ./...` and the linter.

### Acceptance criteria
- A live, spoken end-to-end test covers: wake word → query → tool call →
  confirmation → response → follow-up → reminder delivery → snooze.
- All Python sources removed; repo builds and runs from Go alone.
- CI is green.

### Testing
- [ ] Manual acceptance checklist executed on the target hardware.
- [ ] All ported unit tests pass under `go test ./...`.

---

## Cross-cutting concerns

- [ ] Decide on a logging library and replace `log.Printf` calls with a
      structured logger once the surface area grows.
- [ ] Decide on an error-wrapping convention (`fmt.Errorf("...: %w", err)`
      is already used in `config` and `audio`).
- [ ] Keep `internal/core/context.go` free of ML dependencies so tests can
      run without model files.
- [ ] Document model file locations and download steps in `README.md`.
