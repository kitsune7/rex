"""Tests for speech-to-text wake word stripping."""

import pytest

from stt.stt import Transcriber


class TestStripWakeWord:
    """Tests for Transcriber._strip_wake_word method."""

    @pytest.fixture
    def strip_wake_word(self):
        """Get the _strip_wake_word method without instantiating the full Transcriber.

        We access the method directly to avoid loading the Whisper model.
        """
        # Create a minimal instance without calling __init__
        transcriber = object.__new__(Transcriber)
        return transcriber._strip_wake_word

    def test_hey_rex_at_start(self, strip_wake_word):
        """Test stripping 'hey rex' from the start."""
        assert strip_wake_word("hey rex what time is it") == "What time is it"
        assert strip_wake_word("Hey Rex what time is it") == "What time is it"
        assert strip_wake_word("HEY REX what time is it") == "What time is it"

    def test_hey_rex_with_punctuation(self, strip_wake_word):
        """Test stripping wake word followed by punctuation."""
        assert strip_wake_word("hey rex, what time is it") == "What time is it"
        assert strip_wake_word("hey rex. what time is it") == "What time is it"

    def test_heyrex_no_space(self, strip_wake_word):
        """Test stripping 'heyrex' without space."""
        assert strip_wake_word("heyrex what time is it") == "What time is it"

    def test_hay_rex_variant(self, strip_wake_word):
        """Test stripping 'hay rex' variant (common misheard)."""
        assert strip_wake_word("hay rex what time is it") == "What time is it"

    def test_hey_racks_variant(self, strip_wake_word):
        """Test stripping 'hey racks' variant (common misheard)."""
        assert strip_wake_word("hey racks what time is it") == "What time is it"

    def test_hey_wrecks_variant(self, strip_wake_word):
        """Test stripping 'hey wrecks' variant (common misheard)."""
        assert strip_wake_word("hey wrecks what time is it") == "What time is it"

    def test_no_wake_word(self, strip_wake_word):
        """Test text without wake word is unchanged (except capitalization)."""
        assert strip_wake_word("what time is it") == "What time is it"
        assert strip_wake_word("hello there") == "Hello there"

    def test_wake_word_not_at_start(self, strip_wake_word):
        """Test that wake word in the middle is not stripped."""
        result = strip_wake_word("I said hey rex earlier")
        # The pattern only matches at the start, so it should be unchanged
        assert result == "I said hey rex earlier"

    def test_empty_string(self, strip_wake_word):
        """Test handling empty string."""
        assert strip_wake_word("") == ""

    def test_only_wake_word(self, strip_wake_word):
        """Test handling string that is only the wake word."""
        assert strip_wake_word("hey rex") == ""
        assert strip_wake_word("hey rex ") == ""

    def test_capitalizes_first_letter(self, strip_wake_word):
        """Test that the first letter is capitalized after stripping."""
        assert strip_wake_word("hey rex hello") == "Hello"
        assert strip_wake_word("hey rex 5 minutes") == "5 minutes"

    def test_preserves_rest_of_text(self, strip_wake_word):
        """Test that the rest of the text is preserved exactly."""
        result = strip_wake_word("hey rex Set a Timer for 5 Minutes Please")
        assert result == "Set a Timer for 5 Minutes Please"
