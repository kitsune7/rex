"""Tests for VADProcessor class."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from wake_word.wake_word_listener import VADProcessor


class TestVADProcessor:
    """Tests for VADProcessor class."""

    @pytest.fixture
    def mock_vad_model(self):
        """Create a mock VAD model that returns configurable speech probabilities."""
        model = MagicMock()
        # Default to returning 0.5 probability
        model.return_value.item.return_value = 0.5
        return model

    @pytest.fixture
    def vad_processor(self, mock_vad_model):
        """Create a VADProcessor with mock model."""
        return VADProcessor(mock_vad_model, sample_rate=16000, chunk_size=512)

    def test_add_audio_stores_in_buffer(self, vad_processor):
        """Test that add_audio stores chunks in the buffer."""
        chunk = np.zeros(256, dtype=np.float32)

        vad_processor.add_audio(chunk)

        assert len(vad_processor._buffer) == 1
        assert np.array_equal(vad_processor._buffer[0], chunk)

    def test_add_multiple_chunks(self, vad_processor):
        """Test adding multiple audio chunks."""
        chunk1 = np.zeros(256, dtype=np.float32)
        chunk2 = np.ones(256, dtype=np.float32)

        vad_processor.add_audio(chunk1)
        vad_processor.add_audio(chunk2)

        assert len(vad_processor._buffer) == 2

    def test_process_returns_empty_when_insufficient_samples(self, vad_processor):
        """Test process returns empty list when not enough samples for a chunk."""
        # Add less than chunk_size (512) samples
        chunk = np.zeros(256, dtype=np.float32)
        vad_processor.add_audio(chunk)

        result = vad_processor.process()

        assert result == []

    def test_process_returns_probability_when_enough_samples(self, vad_processor, mock_vad_model):
        """Test process returns speech probability when enough samples."""
        mock_vad_model.return_value.item.return_value = 0.8

        # Add exactly chunk_size (512) samples
        chunk = np.zeros(512, dtype=np.float32)
        vad_processor.add_audio(chunk)

        result = vad_processor.process()

        assert len(result) == 1
        assert result[0] == 0.8

    def test_process_handles_multiple_chunks_worth_of_audio(self, vad_processor, mock_vad_model):
        """Test processing multiple chunks worth of audio."""
        # Configure model to return different values on successive calls
        mock_vad_model.return_value.item.side_effect = [0.3, 0.7, 0.9]

        # Add enough for 3 VAD chunks (512 * 3 = 1536 samples)
        chunk = np.zeros(1536, dtype=np.float32)
        vad_processor.add_audio(chunk)

        result = vad_processor.process()

        assert len(result) == 3
        assert result == [0.3, 0.7, 0.9]

    def test_process_keeps_remainder_in_buffer(self, vad_processor, mock_vad_model):
        """Test that remaining samples stay in buffer after processing."""
        mock_vad_model.return_value.item.return_value = 0.5

        # Add 700 samples (512 + 188 remaining)
        chunk = np.zeros(700, dtype=np.float32)
        vad_processor.add_audio(chunk)

        vad_processor.process()

        # Should have 188 samples remaining
        total_remaining = sum(len(c) for c in vad_processor._buffer)
        assert total_remaining == 188

    def test_reset_clears_buffer(self, vad_processor):
        """Test reset clears the audio buffer."""
        chunk = np.zeros(256, dtype=np.float32)
        vad_processor.add_audio(chunk)
        vad_processor.add_audio(chunk)

        vad_processor.reset()

        assert len(vad_processor._buffer) == 0

    def test_process_after_reset(self, vad_processor, mock_vad_model):
        """Test processing works correctly after reset."""
        mock_vad_model.return_value.item.return_value = 0.6

        # Add and process some audio
        chunk1 = np.zeros(512, dtype=np.float32)
        vad_processor.add_audio(chunk1)
        vad_processor.process()

        # Reset
        vad_processor.reset()

        # Add new audio and process
        chunk2 = np.ones(512, dtype=np.float32)
        vad_processor.add_audio(chunk2)
        result = vad_processor.process()

        assert len(result) == 1

    def test_incremental_processing(self, vad_processor, mock_vad_model):
        """Test incremental addition and processing of audio."""
        mock_vad_model.return_value.item.return_value = 0.5

        # Add in small increments
        for _ in range(4):
            chunk = np.zeros(128, dtype=np.float32)  # 128 samples each
            vad_processor.add_audio(chunk)

        # Now should have 512 samples total
        result = vad_processor.process()

        assert len(result) == 1

    def test_model_receives_correct_audio_format(self, mock_vad_model):
        """Test that the VAD model receives audio in the correct format."""
        processor = VADProcessor(mock_vad_model, sample_rate=16000, chunk_size=512)

        # Add audio as float32
        chunk = np.random.randn(512).astype(np.float32)
        processor.add_audio(chunk)

        processor.process()

        # Check that model was called
        mock_vad_model.assert_called_once()

        # Get the tensor that was passed to the model
        call_args = mock_vad_model.call_args
        audio_tensor = call_args[0][0]
        sample_rate = call_args[0][1]

        assert sample_rate == 16000
        assert len(audio_tensor) == 512
