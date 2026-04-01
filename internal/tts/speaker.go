package tts

import "fmt"

// AudioPlayer is the interface for queuing audio playback.
// Implementations live in sibling packages; this local interface
// keeps the tts package decoupled.
type AudioPlayer interface {
	// QueueAudio enqueues samples for playback and returns immediately.
	QueueAudio(samples []float32, sampleRate int)

	// QueueAudioBlocking enqueues samples and blocks until they finish playing.
	// It periodically calls interruptCheck (if non-nil); if the check returns
	// true, playback stops and the method returns true (interrupted).
	// Returns false when playback completes normally.
	QueueAudioBlocking(samples []float32, sampleRate int, interruptCheck func() bool) bool
}

// Speaker coordinates TTS synthesis with audio playback.
type Speaker struct{}

// Speak synthesises text via the given voice and streams audio chunks to the
// player. If interruptCheck returns true at any point, the piper process is
// killed and Speak returns (true, nil). On normal completion it returns
// (false, nil).
func (s *Speaker) Speak(text string, voice *Voice, player AudioPlayer, interruptCheck func() bool) (bool, error) {
	ch, cancel, err := voice.SynthesizeStream(text)
	if err != nil {
		return false, fmt.Errorf("starting synthesis: %w", err)
	}
	defer cancel()

	for chunk := range ch {
		if chunk.Err != nil {
			return false, fmt.Errorf("synthesis chunk error: %w", chunk.Err)
		}

		if interruptCheck != nil && interruptCheck() {
			return true, nil
		}

		wasInterrupted := player.QueueAudioBlocking(chunk.Samples, chunk.SampleRate, interruptCheck)
		if wasInterrupted {
			return true, nil
		}
	}

	return false, nil
}
