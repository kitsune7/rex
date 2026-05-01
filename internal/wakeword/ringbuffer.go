package wakeword

// ringBuffer is a fixed-capacity FIFO of int16 samples used to keep the
// most recent N audio samples before the wake word fires.
//
// It stores entries in a backing slice and drops the oldest samples when
// the capacity is exceeded. The type is internal to the wakeword package.
type ringBuffer struct {
	capacity int
	data     []int16
}

// newRingBuffer returns a ringBuffer that retains at most capacity samples.
func newRingBuffer(capacity int) *ringBuffer {
	if capacity < 0 {
		capacity = 0
	}
	return &ringBuffer{capacity: capacity}
}

// add appends samples and evicts the oldest if the capacity is exceeded.
func (r *ringBuffer) add(chunk []int16) {
	r.data = append(r.data, chunk...)
	if len(r.data) > r.capacity {
		r.data = r.data[len(r.data)-r.capacity:]
	}
}

// snapshot returns a copy of the current buffer contents.
func (r *ringBuffer) snapshot() []int16 {
	out := make([]int16, len(r.data))
	copy(out, r.data)
	return out
}

// trimTo shrinks the buffer so it retains at most keep of its newest
// samples. If keep >= len, the buffer is unchanged.
func (r *ringBuffer) trimTo(keep int) {
	if keep < 0 {
		keep = 0
	}
	if len(r.data) <= keep {
		return
	}
	r.data = append(r.data[:0:0], r.data[len(r.data)-keep:]...)
}

// len reports how many samples are currently buffered.
func (r *ringBuffer) len() int { return len(r.data) }
