"""Tests for settings loading."""

from rex.settings import load_settings


def test_llm_api_base_defaults_to_localhost(tmp_path):
    settings = load_settings(tmp_path / "missing-settings.toml")

    assert settings.llm.api_base == "http://localhost:1234/v1"
    assert settings.llm.model == "gpt-3.5-turbo"


def test_llm_api_base_can_be_overridden(tmp_path):
    settings_path = tmp_path / "settings.toml"
    settings_path.write_text(
        """
[llm]
api_base = "http://example.test:4321/v1"
model = "local-model"
""",
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    assert settings.llm.api_base == "http://example.test:4321/v1"
    assert settings.llm.model == "local-model"
