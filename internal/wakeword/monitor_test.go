package wakeword

import (
	"runtime"
	"sync/atomic"
	"testing"
	"time"
)

// TestMonitor_GoroutineLeakAfterStartStopLoop verifies that repeatedly
// starting and stopping the monitor does not leak goroutines.
func TestMonitor_GoroutineLeakAfterStartStopLoop(t *testing.T) {
	mustHaveModels(t)

	// Stabilise goroutine count before measurement.
	runtime.GC()
	runtime.Gosched()
	start := runtime.NumGoroutine()

	m := monitorWithFakeMic(t)
	defer m.Close()

	for i := 0; i < 10; i++ {
		if err := m.Start(); err != nil {
			t.Fatalf("Start #%d: %v", i, err)
		}
		if !m.WaitUntilReady(time.Second) {
			t.Fatalf("monitor not ready on iter %d", i)
		}
		m.Stop()
	}

	// Give any straggler goroutines a chance to exit.
	time.Sleep(100 * time.Millisecond)
	runtime.GC()
	runtime.Gosched()

	end := runtime.NumGoroutine()
	// Allow a small margin for runtime-internal goroutines.
	if end > start+2 {
		t.Errorf("goroutine leak: started with %d, ended with %d", start, end)
	}
}

// TestMonitor_DetectAndCaptureWithScriptedDetector verifies the monitor
// wires the pipeline correctly — a scripted detector fires on the 2nd
// chunk and the monitor captures the following audio.
func TestMonitor_DetectAndCaptureWithScriptedDetector(t *testing.T) {
	// Build a monitor from fakes — no ONNX models required.
	det := &scriptedDetector{scores: []float32{0, 0.9}}
	vad := &scriptedVAD{speechFrames: 8}

	mic := &fakeMic{chunks: silenceScript(50), sleep: 20 * time.Millisecond}
	m := &Monitor{
		micFactory: func(cb func([]int16), blockSize int) (micStream, error) {
			mic.callback = cb
			return mic, nil
		},
		detector:   det,
		vadModel:   vad,
		threshold:  0.5,
		bufferDur:  500 * time.Millisecond,
		silenceDur: 100 * time.Millisecond,
	}

	if err := m.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer m.Stop()

	if !m.WaitUntilReady(2 * time.Second) {
		t.Fatal("monitor not ready")
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && !m.Detected() {
		time.Sleep(20 * time.Millisecond)
	}
	if !m.Detected() {
		t.Fatal("monitor did not detect wake word within 3s")
	}

	// Wait for the monitor to finish capturing audio (silence closes it).
	deadline = time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if m.CapturedAudio() != nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if m.CapturedAudio() == nil {
		t.Fatal("monitor did not capture audio after detection")
	}
}

// TestMonitor_StopWhileWaiting ensures Stop unblocks the monitor when it's
// still waiting for a wake word.
func TestMonitor_StopWhileWaiting(t *testing.T) {
	det := &scriptedDetector{}
	vad := &scriptedVAD{}
	mic := &fakeMic{chunks: silenceScript(1000)}

	m := &Monitor{
		micFactory: func(cb func([]int16), blockSize int) (micStream, error) {
			mic.callback = cb
			return mic, nil
		},
		detector:   det,
		vadModel:   vad,
		threshold:  0.5,
		bufferDur:  500 * time.Millisecond,
		silenceDur: 500 * time.Millisecond,
	}

	if err := m.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}

	done := make(chan struct{})
	go func() {
		m.Stop()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("Stop did not return within 3s")
	}

	if m.Detected() {
		t.Errorf("monitor should not have detected anything")
	}
}

// monitorWithFakeMic constructs a monitor with the real ONNX models but a
// fake microphone that always produces silence. Intended for the
// goroutine-leak test.
func monitorWithFakeMic(t *testing.T) *Monitor {
	t.Helper()
	paths := DefaultModelPaths(repoRoot(t), "hey_rex")
	if err := InitONNX(""); err != nil {
		t.Fatalf("InitONNX: %v", err)
	}
	detector, err := NewOpenWakeWord(paths)
	if err != nil {
		t.Fatalf("NewOpenWakeWord: %v", err)
	}
	vadModel, err := NewVADModel(paths.SileroVAD)
	if err != nil {
		detector.Close()
		t.Fatalf("NewVADModel: %v", err)
	}

	chunks := silenceScript(10_000)
	mic := &fakeMic{chunks: chunks, sleep: 5 * time.Millisecond}

	return &Monitor{
		micFactory: func(cb func([]int16), blockSize int) (micStream, error) {
			// Reset the mic so each Start call begins fresh.
			mic.mu.Lock()
			mic.pos = 0
			mic.closed = atomic.Bool{}
			mic.callback = cb
			mic.mu.Unlock()
			return mic, nil
		},
		detector:   detector,
		vadModel:   vadModel,
		threshold:  0.5,
		bufferDur:  500 * time.Millisecond,
		silenceDur: 500 * time.Millisecond,
	}
}
