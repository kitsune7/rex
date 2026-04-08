package wakeword

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"
)

// Monitor runs wake word detection in a background goroutine. It is designed
// for detecting interruptions during TTS playback: when the wake word is
// heard the monitor captures the following speech and stores it so the caller
// can retrieve it without asking the user to repeat themselves.
type Monitor struct {
	detector *Detector
	vad      *VADProcessor
	input    AudioInput

	sampleRate      int
	chunkSize       int
	bufferDuration  time.Duration
	silenceDuration time.Duration

	detected      atomic.Bool
	capturedAudio []int16
	audioMu       sync.Mutex

	cancel context.CancelFunc
	ready  chan struct{}
	done   chan struct{}
}

// MonitorConfig holds configuration for a Monitor.
type MonitorConfig struct {
	Detector        *Detector
	VAD             *VADProcessor
	AudioInput      AudioInput
	SampleRate      int
	ChunkSize       int
	BufferDuration  time.Duration
	SilenceDuration time.Duration
}

// NewMonitor creates a Monitor. Call Start to begin background detection.
func NewMonitor(cfg MonitorConfig) *Monitor {
	if cfg.SampleRate == 0 {
		cfg.SampleRate = 16000
	}
	if cfg.ChunkSize == 0 {
		cfg.ChunkSize = 1280
	}
	if cfg.BufferDuration == 0 {
		cfg.BufferDuration = 3 * time.Second
	}
	if cfg.SilenceDuration == 0 {
		cfg.SilenceDuration = 1500 * time.Millisecond
	}

	return &Monitor{
		detector:        cfg.Detector,
		vad:             cfg.VAD,
		input:           cfg.AudioInput,
		sampleRate:      cfg.SampleRate,
		chunkSize:       cfg.ChunkSize,
		bufferDuration:  cfg.BufferDuration,
		silenceDuration: cfg.SilenceDuration,
	}
}

// Start begins listening for the wake word in a background goroutine.
func (m *Monitor) Start(ctx context.Context) error {
	// Stop any previous run.
	m.Stop()
	m.Reset()

	ctx, m.cancel = context.WithCancel(ctx)
	m.ready = make(chan struct{})
	m.done = make(chan struct{})

	go m.loop(ctx)
	return nil
}

// Stop signals the background goroutine to exit and waits for it.
func (m *Monitor) Stop() {
	if m.cancel != nil {
		m.cancel()
		m.cancel = nil
	}
	if m.done != nil {
		<-m.done
		m.done = nil
	}
}

// WasDetected reports whether the wake word has been detected since the last
// Reset (non-blocking, safe for concurrent use).
func (m *Monitor) WasDetected() bool {
	return m.detected.Load()
}

// GetCapturedAudio returns the audio captured after wake word detection, or
// nil if nothing was captured.
func (m *Monitor) GetCapturedAudio() []int16 {
	m.audioMu.Lock()
	defer m.audioMu.Unlock()
	return m.capturedAudio
}

// Reset clears the detected flag and any captured audio.
func (m *Monitor) Reset() {
	m.detected.Store(false)
	m.audioMu.Lock()
	m.capturedAudio = nil
	m.audioMu.Unlock()
}

// WaitUntilReady blocks until the monitor goroutine is actively listening.
func (m *Monitor) WaitUntilReady() {
	if m.ready != nil {
		<-m.ready
	}
}

func (m *Monitor) loop(ctx context.Context) {
	defer func() {
		if m.done != nil {
			close(m.done)
		}
	}()

	if err := m.input.Start(); err != nil {
		slog.Error("monitor: start audio input", "err", err)
		close(m.ready)
		return
	}
	defer m.input.Stop()

	m.detector.Reset()
	close(m.ready)

	// Ring buffer for pre-wake-word audio.
	bufSamples := int(m.bufferDuration.Seconds()) * m.sampleRate
	ring := make([]int16, bufSamples)
	ringHead := 0
	ringLen := 0

	writeRing := func(samples []int16) {
		for _, s := range samples {
			ring[ringHead] = s
			ringHead = (ringHead + 1) % len(ring)
			if ringLen < len(ring) {
				ringLen++
			}
		}
	}

	readRing := func() []int16 {
		if ringLen == 0 {
			return nil
		}
		out := make([]int16, ringLen)
		start := (ringHead - ringLen + len(ring)) % len(ring)
		if start+ringLen <= len(ring) {
			copy(out, ring[start:start+ringLen])
		} else {
			n := len(ring) - start
			copy(out[:n], ring[start:])
			copy(out[n:], ring[:ringLen-n])
		}
		return out
	}

	chunk := make([]int16, m.chunkSize)
	for {
		n, err := m.input.Read(ctx, chunk)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return
			}
			slog.Error("monitor: read audio", "err", err)
			return
		}
		samples := chunk[:n]
		writeRing(samples)

		score, err := m.detector.Predict(samples)
		if err != nil {
			slog.Error("monitor: predict", "err", err)
			return
		}
		if score >= m.detector.Threshold {
			slog.Info("monitor: wake word detected")
			m.detected.Store(true)
			m.captureUntilSilence(ctx, readRing())
			return
		}
	}
}

func (m *Monitor) captureUntilSilence(ctx context.Context, prefix []int16) {
	audio := make([]int16, len(prefix))
	copy(audio, prefix)

	m.vad.Reset()
	lastSpeech := time.Now()
	speechDetected := false

	chunk := make([]int16, m.chunkSize)
	for {
		n, err := m.input.Read(ctx, chunk)
		if err != nil {
			break
		}
		samples := chunk[:n]
		audio = append(audio, samples...)

		probs, perr := m.vad.Process(samples)
		if perr != nil {
			slog.Error("monitor: VAD", "err", perr)
			break
		}

		for _, p := range probs {
			if p > 0.5 {
				speechDetected = true
				lastSpeech = time.Now()
			}
			if speechDetected && time.Since(lastSpeech) > m.silenceDuration {
				m.audioMu.Lock()
				m.capturedAudio = audio
				m.audioMu.Unlock()
				return
			}
		}
	}

	// Store whatever we have.
	if len(audio) > 0 {
		m.audioMu.Lock()
		m.capturedAudio = audio
		m.audioMu.Unlock()
	}
}
