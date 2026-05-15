package tts

import (
	"context"
	"math"
)

// WakeWordMonitor detects a wake word during audio playback.
// Implementations live in sibling packages; this local interface
// keeps the tts package decoupled.
type WakeWordMonitor interface {
	// Start begins listening for the wake word. The provided context
	// controls the lifetime of the monitor.
	Start(ctx context.Context) error

	// Stop ends wake word monitoring.
	Stop()

	// WasDetected returns true if the wake word has been detected
	// since the last call to Reset.
	WasDetected() bool

	// GetCapturedAudio returns audio captured after wake word detection.
	GetCapturedAudio() []int16

	// Reset clears any prior detection state.
	Reset()

	// WaitUntilReady blocks until the monitor is ready to detect.
	WaitUntilReady()
}

// InterruptibleSpeaker combines TTS playback with wake word monitoring
// so that a user can interrupt speech by saying the wake word.
type InterruptibleSpeaker struct {
	voice   *Voice
	player  AudioPlayer
	monitor WakeWordMonitor
	speaker Speaker
}

// NewInterruptibleSpeaker creates an InterruptibleSpeaker wired to
// the given voice, audio player, and wake word monitor.
func NewInterruptibleSpeaker(voice *Voice, player AudioPlayer, monitor WakeWordMonitor) *InterruptibleSpeaker {
	return &InterruptibleSpeaker{
		voice:   voice,
		player:  player,
		monitor: monitor,
	}
}

// listeningTone is a short 440 Hz sine wave used to signal that Rex is listening.
var listeningTone = generateTone(440, 0.15, DefaultSampleRate)

// doneTone is a short 880 Hz sine wave used to signal capture is complete.
var doneTone = generateTone(880, 0.1, DefaultSampleRate)

// SpeakInterruptibly speaks text while monitoring for the wake word.
// If the user says the wake word during playback, speech is interrupted
// and the audio captured after the wake word is returned.
//
// Returns:
//   - interrupted: true if the wake word was detected
//   - capturedAudio: audio recorded after the wake word (nil if not interrupted)
//   - err: any error during synthesis or playback
func (is *InterruptibleSpeaker) SpeakInterruptibly(ctx context.Context, text string) (interrupted bool, capturedAudio []int16, err error) {
	is.monitor.Reset()

	if err := is.monitor.Start(ctx); err != nil {
		return false, nil, err
	}
	defer is.monitor.Stop()

	is.monitor.WaitUntilReady()

	wasInterrupted, err := is.speaker.Speak(text, is.voice, is.player, is.monitor.WasDetected)
	if err != nil {
		return false, nil, err
	}

	if wasInterrupted {
		is.monitor.Stop()
		is.player.QueueAudio(listeningTone, DefaultSampleRate)
		captured := is.monitor.GetCapturedAudio()
		is.player.QueueAudio(doneTone, DefaultSampleRate)
		return true, captured, nil
	}

	return false, nil, nil
}

// generateTone creates a sine wave tone at the given frequency and duration.
func generateTone(freqHz float64, durationSec float64, sampleRate int) []float32 {
	n := int(durationSec * float64(sampleRate))
	samples := make([]float32, n)
	for i := range samples {
		t := float64(i) / float64(sampleRate)
		samples[i] = float32(0.3 * math.Sin(2*math.Pi*freqHz*t))
	}
	return samples
}
