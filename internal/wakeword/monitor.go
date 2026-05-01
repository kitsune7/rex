package wakeword

import (
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"rex/internal/audio"
)

// MonitorOptions configures a Monitor.
type MonitorOptions struct {
	Models          ModelPaths
	Threshold       float32
	BufferDuration  time.Duration
	SilenceDuration time.Duration
	MicFactory      micStreamFactory
}

// Monitor is a background wake-word detector intended for barge-in during
// TTS playback. It runs in its own goroutine and exposes atomic Detected()
// and CapturedAudio() accessors so the main loop can poll without blocking.
//
// Lifecycle:
//  1. NewMonitor — loads the models.
//  2. Start — spawns a goroutine that opens the microphone and watches for
//     the wake word. After a detection, it continues to record until VAD
//     reports silence, then stops.
//  3. Detected / CapturedAudio — non-blocking readers for the foreground
//     goroutine.
//  4. Stop — halts the goroutine and closes the microphone. Safe to call
//     repeatedly.
//  5. Close — releases the ONNX resources. Must be called once, after Stop.
//
// Monitor owns its own OpenWakeWord and VAD model instances so it can run
// concurrently with the foreground Listener without contention.
type Monitor struct {
	micFactory micStreamFactory
	detector   WakeWordDetector
	vadModel   VADFramePredictor
	threshold  float32
	bufferDur  time.Duration
	silenceDur time.Duration

	ownsModels bool
	mu         sync.Mutex
	running    bool
	stopCh     chan struct{}
	doneCh     chan struct{}

	detected atomic.Bool
	ready    atomic.Bool
	captured atomic.Pointer[[]int16]
}

// NewMonitor loads the ONNX models and returns a stopped Monitor.
func NewMonitor(opts MonitorOptions) (*Monitor, error) {
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

	return &Monitor{
		micFactory: factory,
		detector:   detector,
		vadModel:   vadModel,
		threshold:  threshold,
		bufferDur:  bufferDur,
		silenceDur: silenceDur,
		ownsModels: true,
	}, nil
}

// Start spawns the monitor goroutine. If the monitor is already running
// Start is a no-op. Safe to call repeatedly.
func (m *Monitor) Start() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.running {
		return nil
	}
	m.detected.Store(false)
	m.ready.Store(false)
	m.captured.Store(nil)
	m.detector.Reset()
	m.vadModel.ResetState()

	m.stopCh = make(chan struct{})
	m.doneCh = make(chan struct{})
	m.running = true

	go m.loop(m.stopCh, m.doneCh)
	return nil
}

// Stop halts the monitor goroutine and closes its microphone. Blocks until
// the goroutine exits. Safe to call repeatedly.
func (m *Monitor) Stop() {
	m.mu.Lock()
	if !m.running {
		m.mu.Unlock()
		return
	}
	stopCh := m.stopCh
	doneCh := m.doneCh
	m.running = false
	m.mu.Unlock()

	close(stopCh)
	<-doneCh
}

// Close releases ONNX resources. Stops the monitor first if needed.
func (m *Monitor) Close() {
	m.Stop()
	if !m.ownsModels {
		return
	}
	if m.detector != nil {
		m.detector.Close()
	}
	if closer, ok := m.vadModel.(interface{ Close() }); ok && closer != nil {
		closer.Close()
	}
}

// Detected reports whether the wake word has fired since Start was called.
func (m *Monitor) Detected() bool { return m.detected.Load() }

// WaitUntilReady blocks until the monitor is actively listening, or the
// timeout expires. Returns true if the monitor is ready.
func (m *Monitor) WaitUntilReady(timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if m.ready.Load() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return m.ready.Load()
}

// CapturedAudio returns the audio that was captured after the wake word
// fired, or nil if no detection has occurred (or the monitor hasn't
// finished recording the follow-up speech). The returned slice must not be
// modified by the caller.
func (m *Monitor) CapturedAudio() []int16 {
	p := m.captured.Load()
	if p == nil {
		return nil
	}
	return *p
}

// loop is the monitor goroutine. It opens the microphone, feeds audio into
// the wake-word pipeline, captures follow-up speech on detection, and exits
// when stopCh is closed.
func (m *Monitor) loop(stopCh, doneCh chan struct{}) {
	defer close(doneCh)

	samples := make(chan []int16, 16)
	readErr := make(chan error, 1)
	readStop := make(chan struct{})

	stream, err := m.micFactory(func(buf []int16) {
		select {
		case samples <- buf:
		case <-readStop:
		}
	}, bufferChunkSamples)
	if err != nil {
		log.Printf("wakeword monitor: open mic: %v", err)
		return
	}
	defer stream.Close()

	if err := stream.Start(); err != nil {
		log.Printf("wakeword monitor: start mic: %v", err)
		return
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-readStop:
				return
			default:
			}
			if err := stream.Read(); err != nil {
				select {
				case readErr <- err:
				case <-readStop:
				}
				return
			}
		}
	}()

	defer func() {
		close(readStop)
		_ = stream.Stop()
		wg.Wait()
	}()

	m.ready.Store(true)
	ring := newRingBuffer(int(m.bufferDur.Seconds() * float64(audio.InputSampleRate)))

	// Phase 1: wait for the wake word.
	for {
		select {
		case <-stopCh:
			return
		case err := <-readErr:
			log.Printf("wakeword monitor: read error: %v", err)
			return
		case chunk := <-samples:
			ring.add(chunk)
			score, err := m.detector.Predict(chunk)
			if err != nil {
				log.Printf("wakeword monitor: predict: %v", err)
				return
			}
			if score >= m.threshold {
				m.detected.Store(true)
				m.captureFollowUp(stopCh, samples, readErr, ring.snapshot())
				return
			}
		}
	}
}

// captureFollowUp records audio after wake word detection until the VAD
// reports sustained silence or the monitor is stopped. Saves the audio to
// m.captured so the foreground can retrieve it.
func (m *Monitor) captureFollowUp(stopCh chan struct{}, samples <-chan []int16, readErr <-chan error, prefix []int16) {
	captured := append([]int16(nil), prefix...)
	vad := NewVADProcessor(m.vadModel)
	lastSpeech := time.Now()
	speechDetected := false

	finish := func() {
		copy := append([]int16(nil), captured...)
		m.captured.Store(&copy)
	}

	for {
		select {
		case <-stopCh:
			finish()
			return
		case err := <-readErr:
			log.Printf("wakeword monitor: follow-up read error: %v", err)
			finish()
			return
		case chunk := <-samples:
			captured = append(captured, chunk...)
			vad.AddInt16(chunk)
			probs, err := vad.Process()
			if err != nil {
				log.Printf("wakeword monitor: vad: %v", err)
				finish()
				return
			}
			for _, prob := range probs {
				if prob > 0.5 {
					speechDetected = true
					lastSpeech = time.Now()
				}
				if speechDetected && time.Since(lastSpeech) > m.silenceDur {
					finish()
					return
				}
			}
		}
	}
}
