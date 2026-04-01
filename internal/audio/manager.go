package audio

import (
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gordonklaus/portaudio"
)

// audioChunk represents a piece of audio to be played, optionally followed
// by a completion signal.
type audioChunk struct {
	samples []float32
	done    chan struct{} // if non-nil, closed when this chunk finishes playing
}

// Manager coordinates all audio I/O for the Rex voice assistant.
//
// All output goes through a single persistent PortAudio stream to avoid race
// conditions. Audio is queued via channels and played by an output goroutine
// that fills PortAudio's callback buffer from the queue, then from loop audio,
// then silence.
type Manager struct {
	// Output stream (always open while manager is alive).
	outputStream *portaudio.Stream
	outputBuf    []float32 // PortAudio callback buffer

	// Channel for queued audio chunks.
	queue chan audioChunk

	// Loop audio: played continuously when the queue is empty.
	loopSamples atomic.Pointer[[]float32]

	// Sound file cache: path -> resampled float32 samples at OutputSampleRate.
	soundCache sync.Map

	// Mute state.
	muted atomic.Bool

	// When set, the output goroutine clears its current audio and drains the queue.
	stopRequested atomic.Bool

	// Signals the output goroutine to exit.
	stopCh chan struct{}
	wg     sync.WaitGroup
}

// NewManager initializes PortAudio and opens a persistent output stream.
// Call Cleanup when done to release resources.
func NewManager() (*Manager, error) {
	if err := portaudio.Initialize(); err != nil {
		return nil, fmt.Errorf("initializing portaudio: %w", err)
	}

	m := &Manager{
		outputBuf: make([]float32, ChunkSize),
		queue:     make(chan audioChunk, 256),
		stopCh:    make(chan struct{}),
	}

	stream, err := portaudio.OpenDefaultStream(
		0,              // no input
		OutputChannels, // mono output
		float64(OutputSampleRate),
		ChunkSize,
		m.outputBuf,
	)
	if err != nil {
		portaudio.Terminate()
		return nil, fmt.Errorf("opening output stream: %w", err)
	}
	m.outputStream = stream

	if err := stream.Start(); err != nil {
		stream.Close()
		portaudio.Terminate()
		return nil, fmt.Errorf("starting output stream: %w", err)
	}

	m.wg.Add(1)
	go m.outputLoop()

	return m, nil
}

// outputLoop is the goroutine that continuously writes audio to the PortAudio
// output stream. It pulls from the queue first, falls back to loop audio, then
// writes silence.
func (m *Manager) outputLoop() {
	defer m.wg.Done()

	var current []float32
	var pos int
	var doneCh chan struct{} // completion channel for current chunk

	for {
		select {
		case <-m.stopCh:
			// Signal any pending completion.
			if doneCh != nil {
				close(doneCh)
				doneCh = nil
			}
			return
		default:
		}

		// Check if stop was requested (clear current audio immediately).
		if m.stopRequested.CompareAndSwap(true, false) {
			current = nil
			pos = 0
			if doneCh != nil {
				close(doneCh)
				doneCh = nil
			}
			// Drain the queue, signaling any pending completions.
			for {
				select {
				case chunk := <-m.queue:
					if chunk.done != nil {
						close(chunk.done)
					}
				default:
					goto drained
				}
			}
		drained:
			for i := range m.outputBuf {
				m.outputBuf[i] = 0
			}
			_ = m.outputStream.Write()
			continue
		}

		if m.muted.Load() {
			for i := range m.outputBuf {
				m.outputBuf[i] = 0
			}
			if err := m.outputStream.Write(); err != nil {
				select {
				case <-m.stopCh:
					return
				default:
				}
			}
			continue
		}

		filled := 0
		for filled < ChunkSize {
			// Use current audio if available.
			if current != nil && pos < len(current) {
				n := copy(m.outputBuf[filled:], current[pos:])
				pos += n
				filled += n

				if pos >= len(current) {
					current = nil
					pos = 0
					if doneCh != nil {
						close(doneCh)
						doneCh = nil
					}
				}
				continue
			}

			// Try to dequeue the next chunk (non-blocking).
			select {
			case chunk := <-m.queue:
				current = chunk.samples
				pos = 0
				doneCh = chunk.done
				continue
			default:
			}

			// Fall back to loop audio.
			if lp := m.loopSamples.Load(); lp != nil {
				loopData := *lp
				cpy := make([]float32, len(loopData))
				copy(cpy, loopData)
				current = cpy
				pos = 0
				continue
			}

			// Fill remaining with silence.
			for i := filled; i < ChunkSize; i++ {
				m.outputBuf[i] = 0
			}
			filled = ChunkSize
		}

		if err := m.outputStream.Write(); err != nil {
			select {
			case <-m.stopCh:
				return
			default:
			}
		}
	}
}

// Resample converts audio from one sample rate to another using linear interpolation.
func Resample(samples []float32, fromRate, toRate int) []float32 {
	if fromRate == toRate {
		return samples
	}

	ratio := float64(toRate) / float64(fromRate)
	newLen := int(float64(len(samples)) * ratio)
	if newLen == 0 {
		return nil
	}

	result := make([]float32, newLen)
	lastIdx := float64(len(samples) - 1)

	for i := 0; i < newLen; i++ {
		srcIdx := float64(i) / ratio
		if srcIdx >= lastIdx {
			result[i] = samples[len(samples)-1]
			continue
		}
		lo := int(srcIdx)
		frac := float32(srcIdx - float64(lo))
		result[i] = samples[lo]*(1-frac) + samples[lo+1]*frac
	}

	return result
}

// normalizeFloat32 ensures samples are in [-1, 1] range. If any sample exceeds
// that range, the data is assumed to be int16-scale and is divided by 32768.
func normalizeFloat32(samples []float32) []float32 {
	needsNorm := false
	for _, s := range samples {
		if s > 1.0 || s < -1.0 {
			needsNorm = true
			break
		}
	}
	if !needsNorm {
		return samples
	}

	out := make([]float32, len(samples))
	for i, s := range samples {
		out[i] = s / 32768.0
	}
	return out
}

// QueueAudio sends audio samples for playback. If the sample rate differs from
// OutputSampleRate, the audio is resampled first. Returns immediately.
func (m *Manager) QueueAudio(samples []float32, sampleRate int) {
	if m.muted.Load() {
		return
	}

	if sampleRate != OutputSampleRate {
		samples = Resample(samples, sampleRate, OutputSampleRate)
	}
	samples = normalizeFloat32(samples)

	m.queue <- audioChunk{samples: samples}
}

// QueueAudioBlocking queues audio and blocks until playback completes.
// If interruptCheck is non-nil, it is polled every 50ms; returning true
// stops playback early. Returns true if interrupted, false if completed.
func (m *Manager) QueueAudioBlocking(samples []float32, sampleRate int, interruptCheck func() bool) bool {
	if m.muted.Load() {
		return false
	}

	if sampleRate != OutputSampleRate {
		samples = Resample(samples, sampleRate, OutputSampleRate)
	}
	samples = normalizeFloat32(samples)

	done := make(chan struct{})
	m.queue <- audioChunk{samples: samples, done: done}

	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			return false
		case <-ticker.C:
			if interruptCheck != nil && interruptCheck() {
				m.StopCurrent()
				return true
			}
		}
	}
}

// StartLoop sets audio samples to loop continuously when the queue is empty.
func (m *Manager) StartLoop(samples []float32, sampleRate int) {
	if sampleRate != OutputSampleRate {
		samples = Resample(samples, sampleRate, OutputSampleRate)
	}
	samples = normalizeFloat32(samples)

	cpy := make([]float32, len(samples))
	copy(cpy, samples)
	m.loopSamples.Store(&cpy)
}

// StopLoop clears the loop audio.
func (m *Manager) StopLoop() {
	m.loopSamples.Store(nil)
}

// StopCurrent stops current playback and clears the queue. The actual cleanup
// happens in the output goroutine on its next iteration.
func (m *Manager) StopCurrent() {
	m.stopRequested.Store(true)
}

// PlaySoundFile loads an audio file, caches the resampled result, and plays it.
// If blocking is true, it waits for playback to complete.
func (m *Manager) PlaySoundFile(path string, blocking bool) error {
	samples, err := m.loadCachedSound(path)
	if err != nil {
		return err
	}

	if blocking {
		m.QueueAudioBlocking(samples, OutputSampleRate, nil)
	} else {
		m.QueueAudio(samples, OutputSampleRate)
	}
	return nil
}

// GetSoundDuration returns the duration of an audio file. The file is loaded
// and cached on the first call.
func (m *Manager) GetSoundDuration(path string) (time.Duration, error) {
	samples, err := m.loadCachedSound(path)
	if err != nil {
		return 0, err
	}
	secs := float64(len(samples)) / float64(OutputSampleRate)
	return time.Duration(secs * float64(time.Second)), nil
}

// loadCachedSound loads a sound file, resamples to OutputSampleRate, and caches it.
func (m *Manager) loadCachedSound(path string) ([]float32, error) {
	if cached, ok := m.soundCache.Load(path); ok {
		return cached.([]float32), nil
	}

	raw, sampleRate, err := LoadSoundFile(path)
	if err != nil {
		return nil, fmt.Errorf("loading sound file %s: %w", path, err)
	}

	// Resample to output rate if needed.
	if sampleRate != OutputSampleRate {
		raw = Resample(raw, sampleRate, OutputSampleRate)
	}

	m.soundCache.Store(path, raw)
	return raw, nil
}

// PlayListeningTone plays an ascending C->G tone. Non-blocking.
func (m *Manager) PlayListeningTone() {
	m.QueueAudio(GenerateListeningTone(OutputSampleRate), OutputSampleRate)
}

// PlayDoneTone plays a descending G->C tone. Non-blocking.
func (m *Manager) PlayDoneTone() {
	m.QueueAudio(GenerateDoneTone(OutputSampleRate), OutputSampleRate)
}

// StartThinkingTone starts looping the thinking tone.
func (m *Manager) StartThinkingTone() {
	m.StartLoop(GenerateThinkingSequence(OutputSampleRate), OutputSampleRate)
}

// StopThinkingTone stops the thinking tone loop.
func (m *Manager) StopThinkingTone() {
	m.StopLoop()
}

// Mute silences output and clears any queued or looping audio.
func (m *Manager) Mute() {
	m.muted.Store(true)
	m.StopCurrent()
	m.StopLoop()
}

// Unmute re-enables audio output.
func (m *Manager) Unmute() {
	m.muted.Store(false)
}

// CreateInputStream creates a new microphone input stream with standard settings.
// The callback receives blocks of int16 samples at InputSampleRate.
func (m *Manager) CreateInputStream(callback func([]int16), blockSize int) (*InputStream, error) {
	return CreateInputStream(callback, blockSize)
}

// Cleanup releases all audio resources. The Manager should not be used after this.
func (m *Manager) Cleanup() {
	close(m.stopCh)
	m.wg.Wait()

	if m.outputStream != nil {
		if err := m.outputStream.Stop(); err != nil {
			log.Printf("stopping output stream: %v", err)
		}
		if err := m.outputStream.Close(); err != nil {
			log.Printf("closing output stream: %v", err)
		}
	}

	m.soundCache = sync.Map{}

	if err := portaudio.Terminate(); err != nil {
		log.Printf("terminating portaudio: %v", err)
	}
}
