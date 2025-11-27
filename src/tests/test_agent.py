"""Tests for agent response extraction."""

import pytest

from agent.agent import extract_text_response


class TestExtractTextResponse:
    """Tests for extract_text_response function."""

    def test_string_content(self):
        """Test extracting string content from response."""
        response = {"messages": [type("Message", (), {"content": "Hello, how can I help?"})()]}
        assert extract_text_response(response) == "Hello, how can I help?"

    def test_list_content_with_text_dict(self):
        """Test extracting content from list of dicts with 'text' field."""
        response = {"messages": [type("Message", (), {"content": [{"text": "Response text here"}]})()]}
        assert extract_text_response(response) == "Response text here"

    def test_list_content_without_text_field(self):
        """Test extracting content from list without 'text' field."""
        response = {"messages": [type("Message", (), {"content": ["plain string"]})()]}
        assert extract_text_response(response) == "plain string"

    def test_empty_messages(self):
        """Test handling empty messages list."""
        response = {"messages": []}
        assert extract_text_response(response) == "[No response generated]"

    def test_non_dict_response(self):
        """Test handling non-dict responses."""
        assert extract_text_response("plain string") == "plain string"
        assert extract_text_response(123) == "123"

    def test_missing_messages_key(self):
        """Test handling response without 'messages' key."""
        response = {"other_key": "value"}
        assert extract_text_response(response) == "{'other_key': 'value'}"

    def test_message_without_content_attribute(self):
        """Test handling message without content attribute."""
        response = {"messages": [{"raw": "data"}]}
        result = extract_text_response(response)
        assert "raw" in result or "data" in result

    def test_none_response(self):
        """Test handling None response."""
        assert extract_text_response(None) == "None"

    def test_last_message_is_used(self):
        """Test that the last message in the list is extracted."""
        response = {
            "messages": [
                type("Message", (), {"content": "First message"})(),
                type("Message", (), {"content": "Second message"})(),
                type("Message", (), {"content": "Last message"})(),
            ]
        }
        assert extract_text_response(response) == "Last message"
