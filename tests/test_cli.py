"""Tests for the CLI interface."""

from rex.cli import main


def test_cli_main(capsys):
    """Test the main function of the CLI."""
    main()
    captured = capsys.readouterr()
    assert "Hello from rex!" in captured.out

