package audio

import (
	"fmt"

	"github.com/gordonklaus/portaudio"
)

// InputStream wraps a PortAudio input stream configured for speech capture.
// It uses 16kHz mono int16 format, which is what Whisper expects.
type InputStream struct {
	stream    *portaudio.Stream
	callback  func([]int16)
	blockSize int
	buffer    []int16
}

// CreateInputStream creates a new audio input stream with standard settings.
// The callback is invoked with each block of int16 samples from the microphone.
func CreateInputStream(callback func([]int16), blockSize int) (*InputStream, error) {
	if blockSize <= 0 {
		blockSize = ChunkSize
	}

	is := &InputStream{
		callback:  callback,
		blockSize: blockSize,
		buffer:    make([]int16, blockSize),
	}

	stream, err := portaudio.OpenDefaultStream(
		InputChannels, // input channels
		0,             // output channels (none)
		float64(InputSampleRate),
		blockSize,
		is.buffer,
	)
	if err != nil {
		return nil, fmt.Errorf("opening input stream: %w", err)
	}
	is.stream = stream

	return is, nil
}

// Start begins capturing audio from the microphone.
func (is *InputStream) Start() error {
	return is.stream.Start()
}

// Read reads a single block of audio from the input stream and delivers it
// to the callback. Callers should call this in a loop from a goroutine.
func (is *InputStream) Read() error {
	if err := is.stream.Read(); err != nil {
		return fmt.Errorf("reading input stream: %w", err)
	}
	if is.callback != nil {
		// Copy the buffer so the callback can safely retain the data.
		buf := make([]int16, len(is.buffer))
		copy(buf, is.buffer)
		is.callback(buf)
	}
	return nil
}

// Stop stops capturing audio without releasing resources.
func (is *InputStream) Stop() error {
	return is.stream.Stop()
}

// Close releases the underlying PortAudio stream.
func (is *InputStream) Close() error {
	return is.stream.Close()
}
