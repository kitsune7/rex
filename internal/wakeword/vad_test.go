package wakeword

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// fakePredictor is a VADFramePredictor test double that returns scripted
// probabilities and records the frames it received. Port of the mock_vad_model
// fixture in src/tests/test_vad_processor.py.
type fakePredictor struct {
	chunkSize  int
	scores     []float32
	nextScore  int
	callCount  int
	lastFrame  []float32
	resetCount int
}

func (f *fakePredictor) PredictFrame(frame []float32) (float32, error) {
	f.callCount++
	f.lastFrame = append([]float32(nil), frame...)
	if f.scores == nil {
		return 0.5, nil
	}
	s := f.scores[f.nextScore%len(f.scores)]
	f.nextScore++
	return s, nil
}

func (f *fakePredictor) VADFrameSize() int { return f.chunkSize }

func (f *fakePredictor) ResetState() { f.resetCount++ }

func newFakePredictor(scores ...float32) *fakePredictor {
	return &fakePredictor{chunkSize: 512, scores: scores}
}

func TestVADProcessor_AddAudioStoresInBuffer(t *testing.T) {
	p := NewVADProcessor(newFakePredictor())
	chunk := make([]float32, 256)

	p.AddAudio(chunk)

	if got, want := p.BufferedSamples(), 256; got != want {
		t.Errorf("BufferedSamples = %d, want %d", got, want)
	}
}

func TestVADProcessor_AddMultipleChunks(t *testing.T) {
	p := NewVADProcessor(newFakePredictor())
	chunk1 := make([]float32, 256)
	chunk2 := make([]float32, 256)
	for i := range chunk2 {
		chunk2[i] = 1
	}

	p.AddAudio(chunk1)
	p.AddAudio(chunk2)

	if got, want := p.BufferedSamples(), 512; got != want {
		t.Errorf("BufferedSamples = %d, want %d", got, want)
	}
}

func TestVADProcessor_ProcessReturnsEmptyWhenInsufficientSamples(t *testing.T) {
	p := NewVADProcessor(newFakePredictor())
	p.AddAudio(make([]float32, 256))

	got, err := p.Process()
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected no probabilities, got %v", got)
	}
}

func TestVADProcessor_ProcessReturnsProbabilityWhenEnoughSamples(t *testing.T) {
	fake := newFakePredictor(0.8)
	p := NewVADProcessor(fake)
	p.AddAudio(make([]float32, 512))

	got, err := p.Process()
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if len(got) != 1 || got[0] != 0.8 {
		t.Errorf("Process = %v, want [0.8]", got)
	}
	if fake.callCount != 1 {
		t.Errorf("PredictFrame called %d times, want 1", fake.callCount)
	}
}

func TestVADProcessor_HandlesMultipleChunksWorthOfAudio(t *testing.T) {
	fake := newFakePredictor(0.3, 0.7, 0.9)
	p := NewVADProcessor(fake)
	p.AddAudio(make([]float32, 1536)) // exactly 3 frames

	got, err := p.Process()
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	want := []float32{0.3, 0.7, 0.9}
	if len(got) != len(want) {
		t.Fatalf("Process returned %d probs, want %d", len(got), len(want))
	}
	for i, v := range got {
		if v != want[i] {
			t.Errorf("prob[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestVADProcessor_KeepsRemainderInBuffer(t *testing.T) {
	p := NewVADProcessor(newFakePredictor(0.5))
	p.AddAudio(make([]float32, 700)) // 512 + 188 remainder

	_, err := p.Process()
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if got, want := p.BufferedSamples(), 188; got != want {
		t.Errorf("BufferedSamples after process = %d, want %d", got, want)
	}
}

func TestVADProcessor_ResetClearsBuffer(t *testing.T) {
	p := NewVADProcessor(newFakePredictor())
	p.AddAudio(make([]float32, 256))
	p.AddAudio(make([]float32, 256))

	p.Reset()

	if got := p.BufferedSamples(); got != 0 {
		t.Errorf("BufferedSamples after Reset = %d, want 0", got)
	}
}

func TestVADProcessor_ProcessAfterReset(t *testing.T) {
	fake := newFakePredictor(0.6)
	p := NewVADProcessor(fake)
	p.AddAudio(make([]float32, 512))
	if _, err := p.Process(); err != nil {
		t.Fatalf("first Process: %v", err)
	}

	p.Reset()

	p.AddAudio(make([]float32, 512))
	got, err := p.Process()
	if err != nil {
		t.Fatalf("second Process: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("Process returned %d probs, want 1", len(got))
	}
}

func TestVADProcessor_IncrementalProcessing(t *testing.T) {
	fake := newFakePredictor(0.5)
	p := NewVADProcessor(fake)

	for i := 0; i < 4; i++ {
		p.AddAudio(make([]float32, 128))
	}

	got, err := p.Process()
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("Process returned %d probs after 4×128 samples, want 1", len(got))
	}
}

func TestVADProcessor_ModelReceivesCorrectFrame(t *testing.T) {
	fake := newFakePredictor(0.5)
	p := NewVADProcessor(fake)

	want := make([]float32, 512)
	for i := range want {
		want[i] = float32(i)
	}
	p.AddAudio(want)
	if _, err := p.Process(); err != nil {
		t.Fatalf("Process: %v", err)
	}

	if len(fake.lastFrame) != 512 {
		t.Fatalf("fake received %d samples, want 512", len(fake.lastFrame))
	}
	for i, v := range want {
		if fake.lastFrame[i] != v {
			t.Fatalf("frame[%d] = %v, want %v", i, fake.lastFrame[i], v)
		}
	}
}

func TestVADProcessor_AddInt16Unnormalised(t *testing.T) {
	fake := newFakePredictor(0.5)
	p := NewVADProcessor(fake)

	input := make([]int16, 512)
	for i := range input {
		input[i] = int16(i)
	}
	p.AddInt16(input)

	if _, err := p.Process(); err != nil {
		t.Fatalf("Process: %v", err)
	}
	if fake.lastFrame[10] != float32(10) {
		t.Errorf("int16 conversion altered sample value: frame[10] = %v, want 10", fake.lastFrame[10])
	}
}

// TestVADProcessor_SmokeAgainstRealSileroOnnx exercises the real Silero model
// end-to-end. Skipped if model files are absent or ONNX Runtime is not on
// the system.
func TestVADProcessor_SmokeAgainstRealSileroOnnx(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("smoke test assumes Homebrew onnxruntime install on darwin")
	}
	paths := DefaultModelPaths(repoRoot(t), "hey_rex")
	if err := paths.Verify(); err != nil {
		t.Skipf("models not present: %v", err)
	}
	if err := InitONNX(""); err != nil {
		t.Skipf("onnx init failed: %v", err)
	}

	model, err := NewVADModel(paths.SileroVAD)
	if err != nil {
		t.Fatalf("NewVADModel: %v", err)
	}
	defer model.Close()

	p := NewVADProcessor(model)
	// Feed 512 zero samples — should produce a low speech probability.
	p.AddAudio(make([]float32, 512))
	got, err := p.Process()
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("Process returned %d probs, want 1", len(got))
	}
	if got[0] < 0 || got[0] > 1 {
		t.Errorf("probability %v outside [0, 1]", got[0])
	}
}

// repoRoot walks up from the current test file until it finds go.mod.
func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(file)
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatalf("could not find repo root from %s", file)
	return ""
}
