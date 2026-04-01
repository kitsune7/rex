package audio

import "math"

// Note frequencies in Hz.
const (
	C4 = 261.63 // Middle C
	G4 = 392.00 // Perfect fifth above C4
	D4 = 293.66 // D above middle C
	A4 = 440.00 // A above middle C
)

// Tone timing parameters.
const (
	noteDuration = 0.1  // seconds per note
	gapDuration  = 0.05 // seconds between notes

	thinkingNoteDuration = 0.4
	thinkingGapDuration  = 0.05
	thinkingVolume       = 0.2

	defaultVolume          = 0.3
	defaultEnvelopeDuration = 0.02 // 20ms fade in/out
	thinkingEnvelopeDuration = 0.05 // 50ms for smoother loop transitions
)

// generateTone creates a sine wave tone with smooth raised-cosine envelope.
func generateTone(frequency, duration float64, sampleRate int, volume, envelopeDuration float64) []float32 {
	numSamples := int(float64(sampleRate) * duration)
	if numSamples == 0 {
		return nil
	}

	samples := make([]float32, numSamples)
	envelopeSamples := int(envelopeDuration * float64(sampleRate))

	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		sine := math.Sin(2 * math.Pi * frequency * t)

		// Raised cosine envelope for click-free transitions.
		var env float64
		switch {
		case i < envelopeSamples:
			// Fade in: 0.5 * (1 - cos(pi * progress))
			progress := float64(i) / float64(envelopeSamples)
			env = 0.5 * (1 - math.Cos(math.Pi*progress))
		case i >= numSamples-envelopeSamples:
			// Fade out: 0.5 * (1 + cos(pi * progress))
			progress := float64(i-(numSamples-envelopeSamples)) / float64(envelopeSamples)
			env = 0.5 * (1 + math.Cos(math.Pi*progress))
		default:
			env = 1.0
		}

		samples[i] = float32(sine * env * volume)
	}

	return samples
}

// generateTwoToneSequence creates a two-note sequence with a silence gap between them.
func generateTwoToneSequence(freq1, freq2 float64, sampleRate int, noteDur, gapDur, volume, envDur float64) []float32 {
	tone1 := generateTone(freq1, noteDur, sampleRate, volume, envDur)
	gapSamples := int(float64(sampleRate) * gapDur)
	gap := make([]float32, gapSamples)
	tone2 := generateTone(freq2, noteDur, sampleRate, volume, envDur)

	result := make([]float32, 0, len(tone1)+len(gap)+len(tone2))
	result = append(result, tone1...)
	result = append(result, gap...)
	result = append(result, tone2...)
	return result
}

// GenerateListeningTone creates an ascending C->G tone indicating Rex is listening.
func GenerateListeningTone(sampleRate int) []float32 {
	return generateTwoToneSequence(C4, G4, sampleRate, noteDuration, gapDuration, defaultVolume, defaultEnvelopeDuration)
}

// GenerateDoneTone creates a descending G->C tone indicating Rex finished listening.
func GenerateDoneTone(sampleRate int) []float32 {
	return generateTwoToneSequence(G4, C4, sampleRate, noteDuration, gapDuration, defaultVolume, defaultEnvelopeDuration)
}

// GenerateThinkingSequence creates a D->A tone sequence for looping during LLM inference.
// Uses slower timing and softer volume than the listening/done tones, with a trailing
// gap for seamless looping.
func GenerateThinkingSequence(sampleRate int) []float32 {
	seq := generateTwoToneSequence(D4, A4, sampleRate, thinkingNoteDuration, thinkingGapDuration, thinkingVolume, thinkingEnvelopeDuration)
	trailingGap := make([]float32, int(float64(sampleRate)*thinkingGapDuration))
	return append(seq, trailingGap...)
}
