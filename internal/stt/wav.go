package stt

import (
	"bytes"
	"encoding/binary"
)

// encodeWAV16 serialises 16 kHz mono int16 PCM samples into a WAV byte
// slice whisper-server can decode. We produce it in memory because
// whisper-server accepts audio via multipart upload and has no stdin
// pipe.
func encodeWAV16(samples []int16, sampleRate int) ([]byte, error) {
	const (
		numChannels   uint16 = 1
		bitsPerSample uint16 = 16
		audioFormat   uint16 = 1 // PCM
	)
	byteRate := uint32(sampleRate) * uint32(numChannels) * uint32(bitsPerSample) / 8
	blockAlign := numChannels * bitsPerSample / 8
	dataLen := uint32(len(samples) * 2)
	riffLen := 36 + dataLen

	buf := bytes.NewBuffer(make([]byte, 0, 44+int(dataLen)))
	buf.WriteString("RIFF")
	if err := binary.Write(buf, binary.LittleEndian, riffLen); err != nil {
		return nil, err
	}
	buf.WriteString("WAVE")

	buf.WriteString("fmt ")
	if err := binary.Write(buf, binary.LittleEndian, uint32(16)); err != nil {
		return nil, err
	}
	for _, v := range []any{
		audioFormat,
		numChannels,
		uint32(sampleRate),
		byteRate,
		blockAlign,
		bitsPerSample,
	} {
		if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
			return nil, err
		}
	}

	buf.WriteString("data")
	if err := binary.Write(buf, binary.LittleEndian, dataLen); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, samples); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
