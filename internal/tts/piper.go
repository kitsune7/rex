// Package tts provides text-to-speech synthesis using the Piper TTS engine.
//
// Piper is a C++ TTS engine driven via its CLI. This package wraps the
// piper binary, piping text to stdin and reading raw PCM audio from stdout.
package tts

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"strings"
)

// DefaultSampleRate is the sample rate Piper produces by default.
const DefaultSampleRate = 22050

// chunkSize is the number of int16 samples read from piper stdout per chunk.
const chunkSize = 4096

// AudioChunk holds a slice of audio samples produced by streaming synthesis.
type AudioChunk struct {
	Samples    []float32
	SampleRate int
	Err        error
}

// Voice wraps a Piper TTS model and provides synthesis methods.
type Voice struct {
	modelPath   string
	configPath  string
	piperBinary string
}

// NewVoice creates a Voice that uses the given Piper model file.
// It returns an error if the model file does not exist.
func NewVoice(modelPath string) (*Voice, error) {
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("piper model not found: %w", err)
	}

	configPath := modelPath + ".json"

	return &Voice{
		modelPath:   modelPath,
		configPath:  configPath,
		piperBinary: "piper",
	}, nil
}

// SetBinary overrides the default piper binary path.
func (v *Voice) SetBinary(path string) {
	v.piperBinary = path
}

// SetConfigPath overrides the auto-detected config path.
func (v *Voice) SetConfigPath(path string) {
	v.configPath = path
}

// Synthesize runs Piper synchronously and returns all audio samples at once.
func (v *Voice) Synthesize(text string) ([]float32, int, error) {
	cmd := v.buildCommand()

	cmd.Stdin = strings.NewReader(text)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, 0, fmt.Errorf("piper failed: %w: %s", err, stderr.String())
	}

	samples, err := pcmToFloat32(stdout.Bytes())
	if err != nil {
		return nil, 0, fmt.Errorf("decoding piper output: %w", err)
	}

	return samples, DefaultSampleRate, nil
}

// SynthesizeStream runs Piper and returns a channel that yields AudioChunks
// as they are read from piper's stdout. The channel is closed when synthesis
// finishes or an error occurs. The returned CancelFunc kills the piper process
// and should be called if the caller abandons the stream early.
func (v *Voice) SynthesizeStream(text string) (<-chan AudioChunk, func(), error) {
	cmd := v.buildCommand()

	cmd.Stdin = strings.NewReader(text)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("creating stdout pipe: %w", err)
	}

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("starting piper: %w", err)
	}

	ch := make(chan AudioChunk, 4)

	cancel := func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	}

	go func() {
		defer close(ch)
		defer func() { _ = cmd.Wait() }()

		buf := make([]byte, chunkSize*2) // 2 bytes per int16 sample
		for {
			n, readErr := io.ReadAtLeast(stdout, buf, 2)
			if n >= 2 {
				// Ensure we have an even number of bytes.
				usable := n - (n % 2)
				samples, convErr := pcmToFloat32(buf[:usable])
				if convErr != nil {
					ch <- AudioChunk{Err: convErr}
					cancel()
					return
				}
				ch <- AudioChunk{
					Samples:    samples,
					SampleRate: DefaultSampleRate,
				}
			}
			if readErr != nil {
				if readErr != io.EOF && readErr != io.ErrUnexpectedEOF {
					ch <- AudioChunk{Err: fmt.Errorf("reading piper output: %w: %s", readErr, stderr.String())}
				}
				return
			}
		}
	}()

	return ch, cancel, nil
}

// buildCommand creates the exec.Cmd for piper with the standard flags.
func (v *Voice) buildCommand() *exec.Cmd {
	args := []string{
		"--model", v.modelPath,
		"--output_raw",
	}
	if v.configPath != "" {
		if _, err := os.Stat(v.configPath); err == nil {
			args = append(args, "--config", v.configPath)
		}
	}
	return exec.Command(v.piperBinary, args...)
}

// pcmToFloat32 converts raw little-endian int16 PCM bytes to float32 samples
// normalized to the range [-1.0, 1.0].
func pcmToFloat32(data []byte) ([]float32, error) {
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("PCM data has odd byte count (%d)", len(data))
	}

	numSamples := len(data) / 2
	samples := make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		s := int16(binary.LittleEndian.Uint16(data[i*2:]))
		samples[i] = float32(s) / float32(math.MaxInt16)
	}

	return samples, nil
}
