package wakeword

import (
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeMic implements micStream by dequeuing 1280-sample chunks from a slice.
// It returns an error when the script is exhausted so reads unblock cleanly
// on test shutdown.
type fakeMic struct {
	mu       sync.Mutex
	chunks   [][]int16
	pos      int
	started  atomic.Bool
	closed   atomic.Bool
	callback func([]int16)
	sleep    time.Duration
}

func (f *fakeMic) Start() error { f.started.Store(true); return nil }

func (f *fakeMic) Read() error {
	f.mu.Lock()
	if f.closed.Load() {
		f.mu.Unlock()
		return errFakeClosed
	}
	if f.pos >= len(f.chunks) {
		// Script exhausted: block until closed so the listener can exit
		// cleanly via Interrupt() rather than returning an error here.
		f.mu.Unlock()
		for !f.closed.Load() {
			time.Sleep(10 * time.Millisecond)
		}
		return errFakeClosed
	}
	chunk := f.chunks[f.pos]
	f.pos++
	cb := f.callback
	sleep := f.sleep
	f.mu.Unlock()

	if sleep > 0 {
		time.Sleep(sleep)
	}
	if cb != nil {
		cp := append([]int16(nil), chunk...)
		cb(cp)
	}
	return nil
}

func (f *fakeMic) Stop() error  { return nil }
func (f *fakeMic) Close() error { f.closed.Store(true); return nil }

var (
	errFakeClosed    = &fakeError{"fake mic closed"}
	errFakeExhausted = &fakeError{"fake mic script exhausted"}
)

type fakeError struct{ msg string }

func (e *fakeError) Error() string { return e.msg }

// fakeTones records how many times each tone was played.
type fakeTones struct {
	listening atomic.Int32
	done      atomic.Int32
}

func (t *fakeTones) PlayListeningTone() { t.listening.Add(1) }
func (t *fakeTones) PlayDoneTone()      { t.done.Add(1) }

// scriptedDetector is a WakeWordDetector test double that returns high
// scores for a scripted number of leading chunks, then low scores.
type scriptedDetector struct {
	mu       sync.Mutex
	scores   []float32
	idx      int
	resets   int
}

func (d *scriptedDetector) Predict(chunk []int16) (float32, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	var s float32
	if d.idx < len(d.scores) {
		s = d.scores[d.idx]
	}
	d.idx++
	return s, nil
}

func (d *scriptedDetector) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.idx = 0
	d.resets++
}

func (d *scriptedDetector) Close() {}

// scriptedVAD is a VADFramePredictor test double that reports high speech
// probability for the first N frames and silence afterwards.
type scriptedVAD struct {
	mu          sync.Mutex
	speechFrames int
	idx         int
}

func (v *scriptedVAD) PredictFrame(frame []float32) (float32, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.idx++
	if v.idx <= v.speechFrames {
		return 0.9, nil
	}
	return 0.05, nil
}

func (v *scriptedVAD) VADFrameSize() int { return 512 }
func (v *scriptedVAD) ResetState()       {}

// TestListener_WakeWordTriggersCapture exercises the full WaitForWakeWord
// path with a scripted detector and VAD so the test is deterministic.
func TestListener_WakeWordTriggersCapture(t *testing.T) {
	// Scripted detector: 3 silent chunks, then one at threshold.
	det := &scriptedDetector{scores: []float32{0, 0, 0, 0.9}}
	// VAD: 10 frames of speech (~0.32 s), then silence so captureUntilSilence exits.
	vad := &scriptedVAD{speechFrames: 10}

	// Simulate real-time audio delivery so silenceDur's wall-clock check can
	// trip. Each 1280-sample chunk at 16 kHz is 80 ms of audio.
	mic := &fakeMic{chunks: silenceScript(100), sleep: 20 * time.Millisecond}
	tones := &fakeTones{}

	l := newListenerWithDetector(ListenerOptions{
		Tones:           tones,
		Threshold:       0.5,
		BufferDuration:  500 * time.Millisecond,
		SilenceDuration: 100 * time.Millisecond,
		MicFactory: func(cb func([]int16), blockSize int) (micStream, error) {
			mic.callback = cb
			return mic, nil
		},
	}, det, vad)
	defer l.Close()

	done := make(chan struct{})
	var result []int16
	var resultErr error
	go func() {
		result, resultErr = l.WaitForWakeWord(nil)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		l.Interrupt()
		<-done
		t.Fatalf("WaitForWakeWord did not return within 5s")
	}

	if resultErr != nil {
		t.Fatalf("WaitForWakeWord: %v", resultErr)
	}
	if len(result) == 0 {
		t.Fatalf("WaitForWakeWord returned empty capture")
	}
	if tones.listening.Load() != 1 {
		t.Errorf("listening tone played %d times, want 1", tones.listening.Load())
	}
	if tones.done.Load() != 1 {
		t.Errorf("done tone played %d times, want 1", tones.done.Load())
	}
	if det.resets != 1 {
		t.Errorf("detector Reset called %d times, want 1", det.resets)
	}
}

// TestListener_InterruptReturnsEarly verifies that Interrupt unblocks a
// pending WaitForWakeWord call even when no wake word is present.
func TestListener_InterruptReturnsEarly(t *testing.T) {
	det := &scriptedDetector{} // always returns 0 — never triggers.
	vad := &scriptedVAD{}

	mic := &fakeMic{chunks: silenceScript(1000)}
	tones := &fakeTones{}

	l := newListenerWithDetector(ListenerOptions{
		Tones:           tones,
		BufferDuration:  500 * time.Millisecond,
		SilenceDuration: 500 * time.Millisecond,
		MicFactory: func(cb func([]int16), blockSize int) (micStream, error) {
			mic.callback = cb
			return mic, nil
		},
	}, det, vad)
	defer l.Close()

	done := make(chan error, 1)
	go func() {
		_, err := l.WaitForWakeWord(nil)
		done <- err
	}()

	time.Sleep(50 * time.Millisecond)
	l.Interrupt()

	select {
	case err := <-done:
		if err != ErrInterrupted {
			t.Errorf("WaitForWakeWord returned %v, want ErrInterrupted", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("WaitForWakeWord did not return after Interrupt")
	}
}

// silenceScript returns n chunks of 1280 silent int16 samples.
func silenceScript(n int) [][]int16 {
	out := make([][]int16, n)
	for i := range out {
		out[i] = make([]int16, bufferChunkSamples)
	}
	return out
}

// mustHaveModels skips the test when the ONNX model files aren't present.
func mustHaveModels(t *testing.T) {
	t.Helper()
	paths := DefaultModelPaths(repoRoot(t), "hey_rex")
	if err := paths.Verify(); err != nil {
		t.Skipf("models not present: %v", err)
	}
	if err := InitONNX(""); err != nil {
		t.Skipf("onnx init failed: %v", err)
	}
}

// buildWakeWordScript assembles a mic script: a couple of silent chunks,
// then the synthetic WAV (which won't trigger the wake word but exercises
// the pipeline), then silence. Because the synthetic WAV isn't a real wake
// word, the test is intentionally skip-tolerant.
func buildWakeWordScript(t *testing.T) [][]int16 {
	t.Helper()
	samples := loadTestWAV(t, filepath.Join("testdata", "synthetic.wav"))
	var chunks [][]int16
	for i := 0; i+bufferChunkSamples <= len(samples); i += bufferChunkSamples {
		chunk := make([]int16, bufferChunkSamples)
		copy(chunk, samples[i:i+bufferChunkSamples])
		chunks = append(chunks, chunk)
	}
	// Pad with silence so captureUntilSilence has something to trim against.
	for i := 0; i < 30; i++ {
		chunks = append(chunks, make([]int16, bufferChunkSamples))
	}
	return chunks
}

// TestRingBuffer_AddAndSnapshot verifies the ring buffer evicts oldest
// samples beyond capacity.
func TestRingBuffer_AddAndSnapshot(t *testing.T) {
	r := newRingBuffer(5)
	r.add([]int16{1, 2, 3})
	if got := r.snapshot(); !int16SliceEqual(got, []int16{1, 2, 3}) {
		t.Errorf("snapshot=%v, want [1 2 3]", got)
	}
	r.add([]int16{4, 5, 6, 7})
	if got := r.snapshot(); !int16SliceEqual(got, []int16{3, 4, 5, 6, 7}) {
		t.Errorf("snapshot=%v, want [3 4 5 6 7]", got)
	}
}

// TestRingBuffer_TrimTo keeps only the most recent N samples.
func TestRingBuffer_TrimTo(t *testing.T) {
	r := newRingBuffer(10)
	r.add([]int16{1, 2, 3, 4, 5, 6, 7})
	r.trimTo(3)
	if got := r.snapshot(); !int16SliceEqual(got, []int16{5, 6, 7}) {
		t.Errorf("after trimTo(3), snapshot=%v, want [5 6 7]", got)
	}
}

func int16SliceEqual(a, b []int16) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}
