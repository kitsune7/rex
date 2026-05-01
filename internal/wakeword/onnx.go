// Package wakeword provides wake-word detection, voice activity detection,
// and the rolling-buffer listener used by the state machine when Rex is
// waiting to be addressed.
//
// The implementation ports the Python openWakeWord + Silero VAD pipeline
// (src/wake_word/wake_word_listener.py) on top of ONNX Runtime via the
// github.com/yalue/onnxruntime_go binding.
package wakeword

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// InitONNX loads libonnxruntime and initialises the ONNX Runtime environment.
// It is safe to call repeatedly; only the first call performs initialisation.
//
// Resolution order for the shared library:
//  1. Explicit libraryPath argument, if non-empty.
//  2. ONNXRUNTIME_SHARED_LIBRARY_PATH environment variable.
//  3. Platform default (Homebrew on macOS, standard system paths elsewhere).
func InitONNX(libraryPath string) error {
	return onnxInit.do(libraryPath)
}

// ShutdownONNX releases the ONNX Runtime environment. After Shutdown, callers
// must re-initialise via InitONNX before creating new sessions. Intended for
// use in tests or graceful shutdown; normal programs can simply leak the
// environment until exit.
func ShutdownONNX() error {
	onnxInit.mu.Lock()
	defer onnxInit.mu.Unlock()
	if !onnxInit.initialised {
		return nil
	}
	err := ort.DestroyEnvironment()
	onnxInit.initialised = false
	return err
}

type onnxInitState struct {
	mu          sync.Mutex
	initialised bool
}

var onnxInit = &onnxInitState{}

func (s *onnxInitState) do(libraryPath string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.initialised {
		return nil
	}

	path := libraryPath
	if path == "" {
		path = os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	}
	if path == "" {
		path = defaultLibraryPath()
	}
	if path != "" {
		ort.SetSharedLibraryPath(path)
	}

	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("initialising onnx runtime (library=%q): %w", path, err)
	}
	s.initialised = true
	return nil
}

// defaultLibraryPath returns a best-guess path to libonnxruntime on the
// current platform, or an empty string if we have nothing sensible to offer.
func defaultLibraryPath() string {
	switch runtime.GOOS {
	case "darwin":
		for _, p := range []string{
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
		} {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	case "linux":
		for _, p := range []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
		} {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	return ""
}

// errNotInitialised is returned when a session is created before InitONNX.
var errNotInitialised = errors.New("wakeword: ONNX runtime not initialised; call InitONNX first")

func requireInitialised() error {
	onnxInit.mu.Lock()
	defer onnxInit.mu.Unlock()
	if !onnxInit.initialised {
		return errNotInitialised
	}
	return nil
}
