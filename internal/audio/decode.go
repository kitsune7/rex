package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/hajimehoshi/go-mp3"
)

// LoadSoundFile reads an audio file and returns float32 samples in [-1, 1] range
// along with its sample rate. Supported formats: WAV and MP3.
func LoadSoundFile(path string) ([]float32, int, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".wav":
		return loadWAV(path)
	case ".mp3":
		return loadMP3(path)
	default:
		return nil, 0, fmt.Errorf("unsupported audio format: %s", ext)
	}
}

// loadWAV decodes a WAV file and returns float32 samples and sample rate.
func loadWAV(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("opening wav file: %w", err)
	}
	defer f.Close()

	decoder := wav.NewDecoder(f)
	if !decoder.IsValidFile() {
		return nil, 0, fmt.Errorf("invalid wav file: %s", path)
	}

	buf, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, 0, fmt.Errorf("decoding wav: %w", err)
	}

	sampleRate := int(decoder.SampleRate)
	numChannels := int(decoder.NumChans)
	bitDepth := int(decoder.BitDepth)

	return intBufferToFloat32(buf, numChannels, bitDepth), sampleRate, nil
}

// intBufferToFloat32 converts a go-audio IntBuffer to mono float32 samples.
func intBufferToFloat32(buf *audio.IntBuffer, numChannels, bitDepth int) []float32 {
	data := buf.Data
	numFrames := len(data) / numChannels

	samples := make([]float32, numFrames)
	maxVal := float32((int(1) << (bitDepth - 1)) - 1)

	if numChannels == 1 {
		for i := 0; i < numFrames; i++ {
			samples[i] = float32(data[i]) / maxVal
		}
	} else {
		// Mix to mono by averaging channels.
		invChans := 1.0 / float32(numChannels)
		for i := 0; i < numFrames; i++ {
			var sum float32
			for ch := 0; ch < numChannels; ch++ {
				sum += float32(data[i*numChannels+ch])
			}
			samples[i] = sum * invChans / maxVal
		}
	}

	return samples
}

// loadMP3 decodes an MP3 file and returns float32 samples and sample rate.
// go-mp3 always decodes to 16-bit stereo interleaved PCM.
func loadMP3(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("opening mp3 file: %w", err)
	}
	defer f.Close()

	decoder, err := mp3.NewDecoder(f)
	if err != nil {
		return nil, 0, fmt.Errorf("creating mp3 decoder: %w", err)
	}

	sampleRate := decoder.SampleRate()

	// Read all decoded PCM data (16-bit stereo interleaved).
	pcmData, err := io.ReadAll(decoder)
	if err != nil {
		return nil, 0, fmt.Errorf("decoding mp3: %w", err)
	}

	// Convert 16-bit stereo interleaved to mono float32.
	numStereoSamples := len(pcmData) / 4 // 2 bytes per sample, 2 channels
	samples := make([]float32, numStereoSamples)

	for i := 0; i < numStereoSamples; i++ {
		offset := i * 4
		left := int16(binary.LittleEndian.Uint16(pcmData[offset:]))
		right := int16(binary.LittleEndian.Uint16(pcmData[offset+2:]))
		// Average channels to mono and normalize.
		samples[i] = (float32(left) + float32(right)) / 2.0 / 32768.0
	}

	return samples, sampleRate, nil
}
