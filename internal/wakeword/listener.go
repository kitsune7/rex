package wakeword

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"
)

// AudioInput abstracts a source of 16 kHz mono int16 audio so the listener
// can be tested without real hardware.
type AudioInput interface {
	// Read fills buf with audio samples and returns the number written.
	// It blocks until samples are available or ctx is cancelled.
	Read(ctx context.Context, buf []int16) (int, error)
	Start() error
	Stop() error
}

// AudioPlayer abstracts audio output (tone playback) so the listener does not
// depend on a concrete AudioManager.
type AudioPlayer interface {
	PlayListeningTone()
	PlayDoneTone()
}

// Listener implements wake word detection followed by speech capture.
//
// It maintains a ring buffer of recent audio so that speech that begins before
// the wake word is fully recognised is still captured. After wake word
// detection, it uses VAD to find the end of the utterance, then returns the
// full audio.
type Listener struct {
	detector    *Detector
	vad         *VADProcessor
	audioInput  AudioInput
	audioPlayer AudioPlayer

	// Ring buffer: circular buffer of the last bufferDuration seconds of audio.
	ringBuf  []int16
	ringHead int // next write position
	ringLen  int // number of valid samples

	sampleRate      int
	chunkSize       int
	silenceDuration time.Duration
	interrupted     chan struct{} // closed to signal interruption
}

// ListenerConfig holds configuration for a Listener.
type ListenerConfig struct {
	Detector        *Detector
	VAD             *VADProcessor
	AudioInput      AudioInput
	AudioPlayer     AudioPlayer
	SampleRate      int
	ChunkSize       int
	BufferDuration  time.Duration
	SilenceDuration time.Duration
}

func defaultListenerConfig() ListenerConfig {
	return ListenerConfig{
		SampleRate:      16000,
		ChunkSize:       1280,
		BufferDuration:  3 * time.Second,
		SilenceDuration: 1500 * time.Millisecond,
	}
}

// NewListener creates a Listener with the given configuration.
func NewListener(cfg ListenerConfig) *Listener {
	defaults := defaultListenerConfig()
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaults.SampleRate
	}
	if cfg.ChunkSize == 0 {
		cfg.ChunkSize = defaults.ChunkSize
	}
	if cfg.BufferDuration == 0 {
		cfg.BufferDuration = defaults.BufferDuration
	}
	if cfg.SilenceDuration == 0 {
		cfg.SilenceDuration = defaults.SilenceDuration
	}

	bufSamples := int(cfg.BufferDuration.Seconds()) * cfg.SampleRate
	return &Listener{
		detector:        cfg.Detector,
		vad:             cfg.VAD,
		audioInput:      cfg.AudioInput,
		audioPlayer:     cfg.AudioPlayer,
		ringBuf:         make([]int16, bufSamples),
		sampleRate:      cfg.SampleRate,
		chunkSize:       cfg.ChunkSize,
		silenceDuration: cfg.SilenceDuration,
		interrupted:     make(chan struct{}),
	}
}

// WaitForWakeWordAndSpeech listens for the wake word, then captures speech
// until silence. It calls onWakeWord (if non-nil) as soon as the wake word is
// detected. The returned audio includes ring-buffered audio from before
// detection so the full utterance is preserved.
//
// Returns nil, nil if the context is cancelled before speech completes.
func (l *Listener) WaitForWakeWordAndSpeech(ctx context.Context, onWakeWord func()) ([]int16, error) {
	l.resetInterrupt()
	l.resetRingBuf()
	l.detector.Reset()

	if err := l.audioInput.Start(); err != nil {
		return nil, fmt.Errorf("start audio: %w", err)
	}
	defer l.audioInput.Stop()

	// Phase 1: detect wake word.
	if err := l.waitForWakeWord(ctx); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, errInterrupted) {
			return nil, nil
		}
		return nil, err
	}

	slog.Info("wake word detected")

	if l.audioPlayer != nil {
		l.audioPlayer.PlayListeningTone()
	}
	if onWakeWord != nil {
		onWakeWord()
	}

	// Phase 2: record until silence.
	audio, err := l.recordUntilSilence(ctx, true)
	if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, errInterrupted) {
		return nil, err
	}

	if l.audioPlayer != nil {
		l.audioPlayer.PlayDoneTone()
	}

	return audio, nil
}

// ListenForSpeech records audio without wake word detection. It waits up to
// timeout for speech to begin (determined by VAD), then records until silence.
// Returns nil if no speech is detected within the timeout.
func (l *Listener) ListenForSpeech(ctx context.Context, timeout time.Duration, playTones bool) ([]int16, error) {
	l.resetInterrupt()
	l.resetRingBuf()
	l.vad.Reset()

	if err := l.audioInput.Start(); err != nil {
		return nil, fmt.Errorf("start audio: %w", err)
	}
	defer l.audioInput.Stop()

	deadline := time.Now().Add(timeout)
	chunk := make([]int16, l.chunkSize)

	for {
		if l.isInterrupted() {
			return nil, nil
		}
		if time.Now().After(deadline) {
			return nil, nil
		}

		readCtx, cancel := context.WithDeadline(ctx, deadline)
		n, err := l.audioInput.Read(readCtx, chunk)
		cancel()
		if err != nil {
			if errors.Is(err, context.DeadlineExceeded) {
				return nil, nil
			}
			if errors.Is(err, context.Canceled) {
				return nil, nil
			}
			return nil, err
		}
		samples := chunk[:n]
		l.writeRing(samples)

		probs, err := l.vad.Process(samples)
		if err != nil {
			return nil, err
		}

		for _, p := range probs {
			if p > 0.5 {
				// Speech started. Trim ring buffer to ~0.5s of recent audio.
				recentSamples := l.sampleRate / 2
				prefixAudio := l.readRingRecent(recentSamples)
				l.resetRingBuf()
				l.writeRing(prefixAudio)

				audio, err := l.recordUntilSilence(ctx, true)
				if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, errInterrupted) {
					return nil, err
				}

				if playTones && l.audioPlayer != nil {
					l.audioPlayer.PlayDoneTone()
				}
				return audio, nil
			}
		}
	}
}

// Stop interrupts any in-progress listening operation.
func (l *Listener) Stop() {
	select {
	case <-l.interrupted:
		// already interrupted
	default:
		close(l.interrupted)
	}
}

// IsInterrupted reports whether Stop has been called since the last listen.
func (l *Listener) IsInterrupted() bool {
	return l.isInterrupted()
}

// --- internal helpers ---

var errInterrupted = errors.New("interrupted")

func (l *Listener) isInterrupted() bool {
	select {
	case <-l.interrupted:
		return true
	default:
		return false
	}
}

func (l *Listener) resetInterrupt() {
	// Replace channel so it's open again.
	l.interrupted = make(chan struct{})
}

func (l *Listener) resetRingBuf() {
	l.ringHead = 0
	l.ringLen = 0
}

func (l *Listener) writeRing(samples []int16) {
	for _, s := range samples {
		l.ringBuf[l.ringHead] = s
		l.ringHead = (l.ringHead + 1) % len(l.ringBuf)
		if l.ringLen < len(l.ringBuf) {
			l.ringLen++
		}
	}
}

// readRing returns all valid samples in the ring buffer in order.
func (l *Listener) readRing() []int16 {
	if l.ringLen == 0 {
		return nil
	}
	out := make([]int16, l.ringLen)
	start := (l.ringHead - l.ringLen + len(l.ringBuf)) % len(l.ringBuf)
	if start+l.ringLen <= len(l.ringBuf) {
		copy(out, l.ringBuf[start:start+l.ringLen])
	} else {
		n := len(l.ringBuf) - start
		copy(out[:n], l.ringBuf[start:])
		copy(out[n:], l.ringBuf[:l.ringLen-n])
	}
	return out
}

// readRingRecent returns the most recent n samples from the ring buffer.
func (l *Listener) readRingRecent(n int) []int16 {
	if n > l.ringLen {
		n = l.ringLen
	}
	if n == 0 {
		return nil
	}
	out := make([]int16, n)
	start := (l.ringHead - n + len(l.ringBuf)) % len(l.ringBuf)
	if start+n <= len(l.ringBuf) {
		copy(out, l.ringBuf[start:start+n])
	} else {
		first := len(l.ringBuf) - start
		copy(out[:first], l.ringBuf[start:])
		copy(out[first:], l.ringBuf[:n-first])
	}
	return out
}

func (l *Listener) waitForWakeWord(ctx context.Context) error {
	chunk := make([]int16, l.chunkSize)
	for {
		if l.isInterrupted() {
			return errInterrupted
		}

		n, err := l.audioInput.Read(ctx, chunk)
		if err != nil {
			return err
		}
		samples := chunk[:n]
		l.writeRing(samples)

		score, err := l.detector.Predict(samples)
		if err != nil {
			return fmt.Errorf("detector predict: %w", err)
		}
		if score >= l.detector.Threshold {
			return nil
		}
	}
}

func (l *Listener) recordUntilSilence(ctx context.Context, includeBuffer bool) ([]int16, error) {
	var audio []int16
	if includeBuffer {
		audio = l.readRing()
	}

	lastSpeech := time.Now()
	speechDetected := false
	l.vad.Reset()

	chunk := make([]int16, l.chunkSize)
	for {
		if l.isInterrupted() {
			return audio, errInterrupted
		}

		n, err := l.audioInput.Read(ctx, chunk)
		if err != nil {
			return audio, err
		}
		samples := chunk[:n]
		audio = append(audio, samples...)

		probs, err := l.vad.Process(samples)
		if err != nil {
			return audio, err
		}

		for _, p := range probs {
			if p > 0.5 {
				speechDetected = true
				lastSpeech = time.Now()
			}
			if speechDetected && time.Since(lastSpeech) > l.silenceDuration {
				return audio, nil
			}
		}
	}
}
