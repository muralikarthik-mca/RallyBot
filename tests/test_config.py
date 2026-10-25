"""Configuration tests isolated from developer environment settings."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from rallybot.config import Settings, get_settings
from rallybot.main import main


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Use an empty working directory, clean settings variables, and a fresh cache."""
    monkeypatch.chdir(tmp_path)
    field_names = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in field_names:
            monkeypatch.delenv(name)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def test_default_configuration() -> None:
    assert Settings().model_dump() == {
        "app_name": "RallyBot",
        "app_env": "development",
        "log_level": "INFO",
        "database_url": "sqlite:///./data/rallybot.db",
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "llama3.2",
        "ollama_timeout_seconds": 5.0,
        "ollama_generation_timeout_seconds": 60.0,
        "discord_bot_token": None,
        "discord_guild_id": None,
    }


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "APP_NAME": "RallyBot Test",
        "APP_ENV": "testing",
        "LOG_LEVEL": "DEBUG",
        "DATABASE_URL": "sqlite:///./test.db",
        "OLLAMA_BASE_URL": "http://localhost:9999",
        "OLLAMA_MODEL": "test-model",
        "DISCORD_BOT_TOKEN": "sample-token",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings()
    for name, value in values.items():
        actual = getattr(settings, name.lower())
        if isinstance(actual, SecretStr):
            actual = actual.get_secret_value()
        assert actual == value


@pytest.mark.parametrize(
    ("name", "value"), [("APP_ENV", "staging"), ("LOG_LEVEL", "TRACE")]
)
def test_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError) as error:
        Settings()
    assert error.value.errors()[0]["loc"] == (name.lower(),)


def test_missing_optional_discord_token() -> None:
    assert get_settings().discord_bot_token is None


def test_secret_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "sample-token-never-a-real-secret"
    monkeypatch.setenv("DISCORD_BOT_TOKEN", token)
    settings = Settings()

    assert isinstance(settings.discord_bot_token, SecretStr)
    assert str(settings.discord_bot_token) == "**********"
    for rendered in (str(settings), repr(settings), settings.model_dump_json()):
        assert token not in rendered
        assert "**********" in rendered


def test_cached_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    monkeypatch.setenv("APP_NAME", "Updated RallyBot")
    assert get_settings() is first
    assert get_settings().app_name == "RallyBot"

    get_settings.cache_clear()
    refreshed = get_settings()
    assert refreshed is not first
    assert refreshed.app_name == "Updated RallyBot"


def test_dotenv_loading_and_environment_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "APP_NAME=From dotenv\nAPP_ENV=production\nLOG_LEVEL=WARNING\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_NAME", "From environment")

    settings = Settings()
    assert settings.app_name == "From environment"
    assert settings.app_env == "production"
    assert settings.log_level == "WARNING"
    assert settings.ollama_model == "llama3.2"


def test_cli_output_does_not_expose_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "sample-token")
    main([])
    captured = capsys.readouterr()
    assert captured.out == "RallyBot initialized successfully!\n"
    assert captured.err == ""
