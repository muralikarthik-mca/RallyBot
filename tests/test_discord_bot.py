"""Discord connection tests with fake credentials and mocked gateway operations."""

import asyncio
import importlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import discord
import pytest

from rallybot.config import Settings, get_settings
from rallybot.discord_bot.bot import DiscordBotError, RallyBotClient, start_discord_bot
from rallybot.llm.ollama_client import OllamaHealthResult
from rallybot.logging_config import configure_logging


@pytest.fixture(autouse=True)
def isolated_discord(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields:
            monkeypatch.delenv(name)
    monkeypatch.setenv("DISCORD_GUILD_ID", "123456789012345678")
    monkeypatch.setattr(discord.http.HTTPClient, "static_login", AsyncMock(
        side_effect=AssertionError("Real Discord login is forbidden in tests"),
    ))
    logger = logging.getLogger("rallybot")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    get_settings.cache_clear()
    try:
        yield
    finally:
        for handler in list(logger.handlers):
            if handler not in handlers:
                logger.removeHandler(handler)
                handler.close()
        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
        get_settings.cache_clear()


@pytest.mark.parametrize("token", [None, "", "   "])
def test_missing_token_does_not_connect(monkeypatch: pytest.MonkeyPatch, token: str | None) -> None:
    start = AsyncMock()
    monkeypatch.setattr(RallyBotClient, "start", start)
    with pytest.raises(DiscordBotError, match="token is missing"):
        start_discord_bot(Settings(discord_bot_token=token))
    start.assert_not_awaited()


def test_token_masking() -> None:
    settings = Settings(discord_bot_token="fake-discord-secret")
    assert str(settings.discord_bot_token) == "**********"
    for text in (repr(settings), str(settings), settings.model_dump_json()):
        assert "fake-discord-secret" not in text


def test_client_initialization_and_minimal_intents() -> None:
    async def inspect() -> None:
        async with RallyBotClient(123456789012345678) as client:
            assert isinstance(client, discord.Client)
            assert client.intents.guilds
            assert client.intents.value == discord.Intents(guilds=True).value
            assert not client.intents.message_content
            assert not client.intents.members
            assert not client.intents.presences
            assert not client.intents.messages
            assert client.tree.get_commands() == []
        assert client.is_closed()
    asyncio.run(inspect())


def test_ready_logging_contains_only_safe_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    async def ready() -> None:
        async with RallyBotClient(123456789012345678) as client:
            client._connection.user = SimpleNamespace(id=123456, name="private-name-never-print")
            await client.on_ready()
    asyncio.run(ready())
    captured = capsys.readouterr()
    assert captured.out == ""
    entry = json.loads(captured.err)
    assert entry["message"] == "Discord bot ready (bot_id=123456)"
    assert "private-name-never-print" not in captured.err


def test_event_errors_are_sanitized(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    async def event_error() -> None:
        async with RallyBotClient(123456789012345678) as client:
            try:
                raise RuntimeError("fake-discord-secret")
            except RuntimeError:
                await client.on_error("private-event-name", "fake-discord-secret")
    asyncio.run(event_error())
    output = capsys.readouterr().err
    assert json.loads(output)["message"] == "Discord event handler failed"
    assert "fake-discord-secret" not in output
    assert "private-event-name" not in output
    assert "Traceback" not in output


def test_successful_start_passes_token_once_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[tuple[str, bool]] = []
    clients: list[RallyBotClient] = []
    async def start(self: RallyBotClient, token: str, *, reconnect: bool) -> None:
        started.append((token, reconnect))
        clients.append(self)
    monkeypatch.setattr(RallyBotClient, "start", start)
    start_discord_bot(Settings(discord_bot_token="fake-discord-secret"))
    assert started == [("fake-discord-secret", True)]
    assert clients[0].is_closed()


@pytest.mark.parametrize(("error", "message"), [
    (discord.LoginFailure("fake-discord-secret"), "authentication failed"),
    (aiohttp.ClientConnectionError("fake-discord-secret"), "connection failed"),
    (OSError("fake-discord-secret"), "connection failed"),
    (TimeoutError("fake-discord-secret"), "connection failed"),
    (RuntimeError("fake-discord-secret"), "startup failed"),
])
def test_failures_are_sanitized_and_close(
    monkeypatch: pytest.MonkeyPatch, error: Exception, message: str
) -> None:
    clients: list[RallyBotClient] = []
    async def start(self: RallyBotClient, token: str, *, reconnect: bool) -> None:
        clients.append(self)
        raise error
    monkeypatch.setattr(RallyBotClient, "start", start)
    with pytest.raises(DiscordBotError, match=message) as caught:
        start_discord_bot(Settings(discord_bot_token="fake-discord-secret"))
    assert "fake-discord-secret" not in str(caught.value)
    assert clients[0].is_closed()


def test_ctrl_c_closes_gracefully(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO")
    clients: list[RallyBotClient] = []
    async def interrupted(self: RallyBotClient, token: str, *, reconnect: bool) -> None:
        clients.append(self)
        raise KeyboardInterrupt
    monkeypatch.setattr(RallyBotClient, "start", interrupted)
    start_discord_bot(Settings(discord_bot_token="fake-discord-secret"))
    assert clients[0].is_closed()
    assert json.loads(capsys.readouterr().err)["message"] == "Discord bot stopped"


def test_cancellation_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("rallybot.discord_bot.bot")
    clients: list[RallyBotClient] = []
    async def cancelled(self: RallyBotClient, token: str, *, reconnect: bool) -> None:
        clients.append(self)
        raise asyncio.CancelledError
    monkeypatch.setattr(RallyBotClient, "start", cancelled)
    settings = Settings(discord_bot_token="fake-discord-secret")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(module._connect(settings.discord_bot_token, settings.discord_guild_id))
    assert clients[0].is_closed()


def test_cli_registration() -> None:
    executable = Path(sys.executable).with_name("rallybot.exe")
    result = subprocess.run([str(executable), "--help"], capture_output=True, text=True, check=True)
    for command in ("discord-start", "db-init", "db-check", "ollama-check", "health"):
        assert command in result.stdout


def test_cli_missing_token(capsys: pytest.CaptureFixture[str]) -> None:
    module = importlib.import_module("rallybot.main")
    assert module.main(["discord-start"]) == 1
    captured = capsys.readouterr()
    assert "token is missing" in captured.err
    assert "Traceback" not in captured.err


def test_cli_auth_failure_safe_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = importlib.import_module("rallybot.main")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-discord-secret")
    monkeypatch.setattr(RallyBotClient, "start", AsyncMock(
        side_effect=discord.LoginFailure("fake-discord-secret"),
    ))
    assert module.main(["discord-start"]) == 1
    output = capsys.readouterr()
    assert "authentication failed" in output.err
    assert "fake-discord-secret" not in output.out + output.err
    assert "Traceback" not in output.err


def test_cli_graceful_shutdown_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("rallybot.main")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-discord-secret")
    monkeypatch.setattr(RallyBotClient, "start", AsyncMock(side_effect=KeyboardInterrupt))
    assert module.main(["discord-start"]) == 0


def test_existing_cli_commands_still_work(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = importlib.import_module("rallybot.main")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
    monkeypatch.setattr(module, "check_ollama_health", lambda settings: OllamaHealthResult(
        True, True, "llama3.2", ["llama3.2:latest"], None, "Available.",
    ))
    for args in ([], ["db-init"], ["db-check"], ["ollama-check"]):
        assert module.main(args) == 0
    assert "RallyBot initialized successfully!" in capsys.readouterr().out
