// Package audio provides audio I/O management for the Rex voice assistant.
//
// It handles input/output streams, feedback tones, sound file playback,
// and audio resampling through a single persistent PortAudio output stream.
package audio

const (
	// InputSampleRate is the sample rate expected by Whisper STT.
	InputSampleRate = 16000

	// OutputSampleRate is the unified output sample rate.
	OutputSampleRate = 44100

	// TTSSampleRate is the native sample rate for Piper TTS.
	TTSSampleRate = 24000

	// InputChannels is the number of input audio channels (mono).
	InputChannels = 1

	// OutputChannels is the number of output audio channels (mono).
	OutputChannels = 1

	// ChunkSize is the default block size in samples for audio I/O.
	ChunkSize = 1024
)
