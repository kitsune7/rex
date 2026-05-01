// Package stt provides speech-to-text transcription for Rex.
//
// It shells out to whisper-server (part of the Homebrew whisper-cpp
// package) as a long-running sidecar, then sends captured microphone
// audio over HTTP multipart requests. The sidecar keeps the Whisper
// model loaded between utterances so per-transcription latency is the
// inference cost alone, not a fresh model load.
package stt

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ErrServerUnavailable is returned by Transcribe when the backing
// whisper-server process has died or never started.
var ErrServerUnavailable = errors.New("stt: whisper-server unavailable")

// sampleRateHz is the sample rate we feed whisper-server. The rest of
// the system already records at 16 kHz to match Whisper's expectation.
const sampleRateHz = 16000

// Options configures a Transcriber.
type Options struct {
	// ModelPath is the GGML model (.bin) whisper-server should load.
	// Required.
	ModelPath string

	// Binary is the whisper-server executable to launch. Defaults to
	// "whisper-server" (resolved via $PATH).
	Binary string

	// Language is the ISO language code passed with each request. An
	// empty value defaults to "en".
	Language string

	// StartTimeout caps how long we wait for whisper-server to begin
	// accepting connections. Defaults to 60s, which is generous for a
	// cold Metal backend init.
	StartTimeout time.Duration

	// RequestTimeout caps a single /inference call. Defaults to 60s.
	RequestTimeout time.Duration

	// Logger receives whisper-server stdout/stderr. If nil, lines are
	// forwarded to the default logger prefixed with "whisper-server:".
	Logger *log.Logger
}

// Transcriber converts int16 16 kHz PCM samples to trimmed transcripts
// by delegating to an owned whisper-server subprocess.
type Transcriber struct {
	opts    Options
	cmd     *exec.Cmd
	sink    *lineSink
	baseURL string
	client  *http.Client

	// exited is closed once the whisper-server process has exited. We
	// set up a single Wait goroutine in NewTranscriber so Close never
	// races against Wait's one-shot contract.
	exited chan struct{}
	waitMu sync.Mutex
	waitEr error

	// mu protects Close's idempotency.
	mu     sync.Mutex
	closed bool
}

// NewTranscriber starts whisper-server, waits for it to accept
// connections, and returns a ready-to-use Transcriber. The caller owns
// the returned value and must call Close to reclaim the subprocess.
//
// NewTranscriber logs a one-time cold-start message before blocking on
// the server's startup so callers can surface the latency to the user.
func NewTranscriber(opts Options) (*Transcriber, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("stt: ModelPath is required")
	}
	if _, err := os.Stat(opts.ModelPath); err != nil {
		return nil, fmt.Errorf("stt: model file: %w", err)
	}
	binary := opts.Binary
	if binary == "" {
		binary = "whisper-server"
	}
	if _, err := exec.LookPath(binary); err != nil {
		return nil, fmt.Errorf("stt: %s not found on PATH — install whisper-cpp (brew install whisper-cpp): %w", binary, err)
	}

	if opts.StartTimeout == 0 {
		opts.StartTimeout = 60 * time.Second
	}
	if opts.RequestTimeout == 0 {
		opts.RequestTimeout = 60 * time.Second
	}
	if opts.Language == "" {
		opts.Language = "en"
	}
	if opts.Logger == nil {
		opts.Logger = log.Default()
	}

	port, err := pickFreePort()
	if err != nil {
		return nil, fmt.Errorf("stt: finding free port: %w", err)
	}

	log.Printf("stt: loading Whisper model %s (may take a moment on first run)", opts.ModelPath)

	cmd := exec.Command(
		binary,
		"--model", opts.ModelPath,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--language", opts.Language,
		"--no-timestamps",
		"--inference-path", "/inference",
	)
	// whisper-server is chatty; capture its output so callers can see
	// load progress but keep it from polluting stderr if the app wants
	// structured logging later.
	sink := newLineSink(opts.Logger)
	cmd.Stdout = sink
	cmd.Stderr = sink

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("stt: starting %s: %w", binary, err)
	}

	t := &Transcriber{
		opts:    opts,
		cmd:     cmd,
		sink:    sink,
		baseURL: fmt.Sprintf("http://127.0.0.1:%d", port),
		client:  &http.Client{Timeout: opts.RequestTimeout},
		exited:  make(chan struct{}),
	}
	go func() {
		err := cmd.Wait()
		t.waitMu.Lock()
		t.waitEr = err
		t.waitMu.Unlock()
		close(t.exited)
	}()

	if err := t.waitForReady(opts.StartTimeout); err != nil {
		_ = t.terminate()
		return nil, err
	}
	return t, nil
}

// Transcribe converts a buffer of int16 16 kHz PCM audio into the text
// Whisper produced. When stripWakeWord is true, leading "hey rex"
// variants are removed and the first letter is capitalised, matching
// src/stt/stt.py.
func (t *Transcriber) Transcribe(ctx context.Context, samples []int16, stripWakeWord bool) (string, error) {
	if t == nil {
		return "", ErrServerUnavailable
	}
	t.mu.Lock()
	closed := t.closed
	t.mu.Unlock()
	if closed {
		return "", ErrServerUnavailable
	}
	if len(samples) == 0 {
		return "", nil
	}

	wav, err := encodeWAV16(samples, sampleRateHz)
	if err != nil {
		return "", fmt.Errorf("stt: encoding wav: %w", err)
	}

	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	fw, err := mw.CreateFormFile("file", "audio.wav")
	if err != nil {
		return "", fmt.Errorf("stt: creating form file: %w", err)
	}
	if _, err := fw.Write(wav); err != nil {
		return "", fmt.Errorf("stt: writing form file: %w", err)
	}
	for k, v := range map[string]string{
		"temperature":     "0.0",
		"temperature_inc": "0.2",
		"response_format": "json",
		"language":        t.opts.Language,
	} {
		if err := mw.WriteField(k, v); err != nil {
			return "", fmt.Errorf("stt: writing form field %s: %w", k, err)
		}
	}
	if err := mw.Close(); err != nil {
		return "", fmt.Errorf("stt: closing multipart: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.baseURL+"/inference", body)
	if err != nil {
		return "", fmt.Errorf("stt: building request: %w", err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := t.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("stt: posting to whisper-server: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("stt: reading response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("stt: whisper-server returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var parsed struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", fmt.Errorf("stt: decoding json: %w", err)
	}
	text := strings.TrimSpace(parsed.Text)
	if stripWakeWord {
		text = StripWakeWord(text)
	}
	return text, nil
}

// Close terminates the whisper-server subprocess. Safe to call more
// than once.
func (t *Transcriber) Close() error {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return nil
	}
	t.closed = true
	t.mu.Unlock()
	return t.terminate()
}

func (t *Transcriber) terminate() error {
	if t.cmd == nil || t.cmd.Process == nil {
		return nil
	}
	// Ask politely first, then force-kill if it overstays its welcome.
	_ = t.cmd.Process.Signal(os.Interrupt)
	select {
	case <-t.exited:
		t.waitMu.Lock()
		err := t.waitEr
		t.waitMu.Unlock()
		if err != nil && !isExpectedExit(err) {
			return err
		}
		return nil
	case <-time.After(5 * time.Second):
		_ = t.cmd.Process.Kill()
		<-t.exited
		return nil
	}
}

// waitForReady polls the server's root path until it responds or we
// exceed the startup budget. whisper-server does not print a structured
// "ready" line, so polling is the simplest signal.
func (t *Transcriber) waitForReady(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 500 * time.Millisecond}
	for {
		select {
		case <-t.exited:
			return fmt.Errorf("stt: whisper-server exited before becoming ready")
		default:
		}
		resp, err := client.Get(t.baseURL + "/")
		if err == nil {
			resp.Body.Close()
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("stt: whisper-server did not start within %s: %w", timeout, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// pickFreePort asks the kernel for an unused TCP port by binding to
// :0 and immediately releasing it.
func pickFreePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// lineSink forwards subprocess output to a logger one line at a time so
// log timestamps line up with events inside Rex.
type lineSink struct {
	mu     sync.Mutex
	logger *log.Logger
	pr     *io.PipeReader
	pw     *io.PipeWriter
}

func newLineSink(logger *log.Logger) *lineSink {
	pr, pw := io.Pipe()
	s := &lineSink{logger: logger, pr: pr, pw: pw}
	go s.pump()
	return s
}

func (s *lineSink) pump() {
	scanner := bufio.NewScanner(s.pr)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		s.logger.Printf("whisper-server: %s", scanner.Text())
	}
}

func (s *lineSink) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pw.Write(p)
}

func isExpectedExit(err error) bool {
	_, ok := errors.AsType[*exec.ExitError](err)
	return ok
}
