package wakeword

import (
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"rex/internal/audio"
)

// AudioTonePlayer is the narrow subset of audio.Manager that the listener
// uses for its listening/done tones. Tests inject a fake to avoid PortAudio.
type AudioTonePlayer interface {
	PlayListeningTone()
	PlayDoneTone()
}

// micStreamFactory creates a microphone input stream. The factory seam
// exists so tests can inject a fake source instead of PortAudio.
type micStreamFactory func(callback func([]int16), blockSize int) (micStream, error)

// micStream is the subset of audio.InputStream the listener interacts with.
type micStream interface {
	Start() error
	Read() error
	Stop() error
	Close() error
}

// ListenerOptions configures a Listener.
type ListenerOptions struct {
	// Models describes where to load the three ONNX models from.
	Models ModelPaths

	// Tones plays the listening/done feedback sounds. Required.
	Tones AudioTonePlayer

	// Threshold is the score (0-1) above which a wake word is considered
	// detected. Defaults to 0.5 when zero.
	Threshold float32

	// BufferDuration is how much pre-wake-word audio to keep. Defaults to
	// 3 seconds.
	BufferDuration time.Duration

	// SilenceDuration is how long of silence ends recording. Defaults to
	// 1.5 seconds.
	SilenceDuration time.Duration

	// MicFactory overrides the microphone stream factory. Tests inject a
	// fake; production leaves this nil and audio.CreateInputStream is used.
	MicFactory micStreamFactory
}

// WakeWordDetector is the seam between the listener and the
// openWakeWord pipeline. Production code uses *OpenWakeWord; tests can
// supply a deterministic fake that returns scripted scores.
type WakeWordDetector interface {
	Predict(chunk []int16) (float32, error)
	Reset()
	Close()
}

// Listener captures audio from the microphone, detects the wake word, and
// returns the speech that follows. It is the Go port of the Python
// WakeWordListener in src/wake_word/wake_word_listener.py.
//
// Lifecycle:
//  1. Call New to construct (loads ONNX models).
//  2. Call WaitForWakeWord to block until a wake word is detected and the
//     user finishes speaking. Returns the full captured audio (pre-wake
//     buffer + post-wake speech).
//  3. Call ListenForSpeech for follow-up queries that skip the wake word.
//  4. Call Interrupt from another goroutine to unblock an in-progress call.
//  5. Call Close when shutting down.
type Listener struct {
	tones      AudioTonePlayer
	micFactory micStreamFactory
	detector   WakeWordDetector
	vadModel   VADFramePredictor
	threshold  float32
	bufferDur  time.Duration
	silenceDur time.Duration

	ownsModels  bool
	mu          sync.Mutex // guards active stream
	interrupted atomic.Bool
	closed      atomic.Bool
}

// NewListener creates a Listener using the given options.
func NewListener(opts ListenerOptions) (*Listener, error) {
	if opts.Tones == nil {
		return nil, fmt.Errorf("wakeword: ListenerOptions.Tones is required")
	}
	if err := opts.Models.Verify(); err != nil {
		return nil, err
	}
	if err := InitONNX(""); err != nil {
		return nil, err
	}
	detector, err := NewOpenWakeWord(opts.Models)
	if err != nil {
		return nil, fmt.Errorf("loading wake-word pipeline: %w", err)
	}
	vadModel, err := NewVADModel(opts.Models.SileroVAD)
	if err != nil {
		detector.Close()
		return nil, fmt.Errorf("loading vad model: %w", err)
	}

	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}
	bufferDur := opts.BufferDuration
	if bufferDur <= 0 {
		bufferDur = 3 * time.Second
	}
	silenceDur := opts.SilenceDuration
	if silenceDur <= 0 {
		silenceDur = 1500 * time.Millisecond
	}
	factory := opts.MicFactory
	if factory == nil {
		factory = defaultMicFactory
	}

	return &Listener{
		tones:      opts.Tones,
		micFactory: factory,
		detector:   detector,
		vadModel:   vadModel,
		threshold:  threshold,
		bufferDur:  bufferDur,
		silenceDur: silenceDur,
		ownsModels: true,
	}, nil
}

// newListenerWithDetector is a test helper that wires a Listener to the
// given detector + VAD predictor without loading ONNX models. The caller
// owns the detector and VAD lifetimes.
func newListenerWithDetector(opts ListenerOptions, detector WakeWordDetector, vad VADFramePredictor) *Listener {
	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}
	bufferDur := opts.BufferDuration
	if bufferDur <= 0 {
		bufferDur = 3 * time.Second
	}
	silenceDur := opts.SilenceDuration
	if silenceDur <= 0 {
		silenceDur = 1500 * time.Millisecond
	}
	factory := opts.MicFactory
	if factory == nil {
		factory = defaultMicFactory
	}
	return &Listener{
		tones:      opts.Tones,
		micFactory: factory,
		detector:   detector,
		vadModel:   vad,
		threshold:  threshold,
		bufferDur:  bufferDur,
		silenceDur: silenceDur,
		ownsModels: false,
	}
}

// Close releases model and audio resources. Safe to call once.
func (l *Listener) Close() {
	if l == nil {
		return
	}
	if !l.closed.CompareAndSwap(false, true) {
		return
	}
	if !l.ownsModels {
		return
	}
	if l.detector != nil {
		l.detector.Close()
	}
	if closer, ok := l.vadModel.(interface{ Close() }); ok && closer != nil {
		closer.Close()
	}
}

// Interrupt causes any in-flight WaitForWakeWord or ListenForSpeech call to
// return early with ErrInterrupted. Thread-safe.
func (l *Listener) Interrupt() {
	l.interrupted.Store(true)
}

// IsInterrupted reports whether Interrupt has been called since the last
// operation began.
func (l *Listener) IsInterrupted() bool {
	return l.interrupted.Load()
}

// ErrInterrupted is returned when a listener operation is cancelled via
// Interrupt() before it finishes.
var ErrInterrupted = fmt.Errorf("wakeword: listener interrupted")

// bufferChunkSamples is the input read size (80 ms @ 16 kHz).
const bufferChunkSamples = 1280

// WaitForWakeWord blocks until the wake word is detected, then records
// audio until the user stops speaking. Returns the combined audio (the
// rolling pre-wake buffer plus the new speech) as int16 samples at 16 kHz.
//
// onWakeWord, if non-nil, is invoked immediately after the wake word fires.
// Use it to mute other audio sources before the listening tone plays.
func (l *Listener) WaitForWakeWord(onWakeWord func()) ([]int16, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.interrupted.Store(false)
	l.detector.Reset()

	stream, samples, readErr, stopRead, err := l.openMic()
	if err != nil {
		return nil, err
	}
	defer stopRead()
	defer stream.Close()

	ring := newRingBuffer(int(l.bufferDur.Seconds() * float64(audio.InputSampleRate)))

	// Phase 1: wait for wake word, maintaining the rolling buffer.
	detected, err := l.waitForDetection(samples, readErr, ring)
	if err != nil {
		return nil, err
	}
	if !detected {
		return nil, ErrInterrupted
	}

	log.Print("wakeword: wake word detected, capturing speech")
	l.tones.PlayListeningTone()
	if onWakeWord != nil {
		onWakeWord()
	}

	// Phase 2: record until silence.
	captured, err := l.captureUntilSilence(samples, readErr, ring.snapshot())
	l.tones.PlayDoneTone()
	return captured, err
}

// ListenForSpeech records speech without requiring a wake word. It waits
// up to timeout for speech to begin, then records until silence. Used for
// follow-up questions.
//
// playTones controls whether the done tone plays; the caller is responsible
// for any ready tone that preceded this call.
func (l *Listener) ListenForSpeech(timeout time.Duration, playTones bool) ([]int16, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.interrupted.Store(false)

	stream, samples, readErr, stopRead, err := l.openMic()
	if err != nil {
		return nil, err
	}
	defer stopRead()
	defer stream.Close()

	ring := newRingBuffer(int(l.bufferDur.Seconds() * float64(audio.InputSampleRate)))

	vad := NewVADProcessor(l.vadModel)
	deadline := time.Now().Add(timeout)

	for {
		if l.interrupted.Load() {
			return nil, ErrInterrupted
		}
		if time.Now().After(deadline) {
			return nil, nil
		}

		chunk, err := l.readChunkInterruptible(samples, readErr)
		if err != nil {
			return nil, err
		}
		if chunk == nil {
			return nil, ErrInterrupted
		}
		ring.add(chunk)
		vad.AddInt16(chunk)

		probs, err := vad.Process()
		if err != nil {
			return nil, err
		}
		for _, prob := range probs {
			if prob > 0.5 {
				// Trim the rolling buffer to 0.5 s of context so we don't
				// include arbitrary pre-speech silence.
				ring.trimTo(audio.InputSampleRate / 2)
				result, err := l.captureUntilSilence(samples, readErr, ring.snapshot())
				if err == nil && playTones {
					l.tones.PlayDoneTone()
				}
				return result, err
			}
		}
	}
}

// waitForDetection runs the wake-word pipeline on incoming audio until it
// fires or the listener is interrupted.
func (l *Listener) waitForDetection(samples <-chan []int16, readErr <-chan error, ring *ringBuffer) (bool, error) {
	for {
		if l.interrupted.Load() {
			return false, nil
		}
		chunk, err := l.readChunkInterruptible(samples, readErr)
		if err != nil {
			return false, err
		}
		if chunk == nil {
			return false, nil
		}
		ring.add(chunk)

		score, err := l.detector.Predict(chunk)
		if err != nil {
			return false, err
		}
		if score >= l.threshold {
			return true, nil
		}
	}
}

// captureUntilSilence keeps recording audio after the wake word until the
// VAD reports sustained silence. `prefix` is the rolling buffer content
// captured before the wake word fired, included at the start of the result.
func (l *Listener) captureUntilSilence(samples <-chan []int16, readErr <-chan error, prefix []int16) ([]int16, error) {
	captured := append([]int16(nil), prefix...)
	vad := NewVADProcessor(l.vadModel)

	lastSpeech := time.Now()
	speechDetected := false

	for {
		if l.interrupted.Load() {
			return captured, nil
		}
		chunk, err := l.readChunkInterruptible(samples, readErr)
		if err != nil {
			return captured, err
		}
		if chunk == nil {
			return captured, nil
		}
		captured = append(captured, chunk...)
		vad.AddInt16(chunk)

		probs, err := vad.Process()
		if err != nil {
			return captured, err
		}
		for _, prob := range probs {
			if prob > 0.5 {
				speechDetected = true
				lastSpeech = time.Now()
			}
			if speechDetected && time.Since(lastSpeech) > l.silenceDur {
				return captured, nil
			}
		}
	}
}

// openMic starts a microphone stream and returns channels that deliver
// samples (and any read error) to the listener. The returned stopRead
// function halts the reader goroutine; the caller must also close the
// stream afterwards.
func (l *Listener) openMic() (micStream, <-chan []int16, <-chan error, func(), error) {
	samples := make(chan []int16, 16)
	readErr := make(chan error, 1)
	stopCh := make(chan struct{})

	stream, err := l.micFactory(func(buf []int16) {
		select {
		case samples <- buf:
		case <-stopCh:
		}
	}, bufferChunkSamples)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("opening microphone: %w", err)
	}
	if err := stream.Start(); err != nil {
		_ = stream.Close()
		return nil, nil, nil, nil, fmt.Errorf("starting microphone: %w", err)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stopCh:
				return
			default:
			}
			if err := stream.Read(); err != nil {
				select {
				case readErr <- err:
				case <-stopCh:
				}
				return
			}
		}
	}()

	stop := func() {
		close(stopCh)
		_ = stream.Stop()
		wg.Wait()
	}
	return stream, samples, readErr, stop, nil
}

// readChunk blocks until the next audio chunk or an error is available.
func readChunk(samples <-chan []int16, readErr <-chan error) ([]int16, error) {
	select {
	case c, ok := <-samples:
		if !ok {
			return nil, fmt.Errorf("wakeword: sample channel closed")
		}
		return c, nil
	case err := <-readErr:
		return nil, err
	}
}

// readChunkInterruptible is readChunk plus a 50 ms poll on l.interrupted so
// a listener can abort even when no audio is arriving from the mic. Returns
// (nil, nil) when interrupted.
func (l *Listener) readChunkInterruptible(samples <-chan []int16, readErr <-chan error) ([]int16, error) {
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case c, ok := <-samples:
			if !ok {
				return nil, fmt.Errorf("wakeword: sample channel closed")
			}
			return c, nil
		case err := <-readErr:
			return nil, err
		case <-ticker.C:
			if l.interrupted.Load() {
				return nil, nil
			}
		}
	}
}

// defaultMicFactory wraps audio.CreateInputStream in the micStream interface.
func defaultMicFactory(callback func([]int16), blockSize int) (micStream, error) {
	return audio.CreateInputStream(callback, blockSize)
}
