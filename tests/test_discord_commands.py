"""Guild-only /ping tests with all Discord API operations mocked."""

import asyncio
import importlib
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest
from discord import app_commands
from pydantic import ValidationError

from rallybot.config import Settings, get_settings
from rallybot.discord_bot.bot import DiscordBotError, RallyBotClient, start_discord_bot
from rallybot.discord_bot.commands import ping, register_commands
from rallybot.logging_config import configure_logging


@pytest.fixture(autouse=True)
def isolated_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields:
            monkeypatch.delenv(name)
    monkeypatch.setattr(discord.http.HTTPClient, "static_login", AsyncMock(
        side_effect=AssertionError("Real Discord login is forbidden in tests"),
    ))
    monkeypatch.setattr(discord.http.HTTPClient, "bulk_upsert_guild_commands", AsyncMock(
        side_effect=AssertionError("Real Discord sync is forbidden in tests"),
    ))
    monkeypatch.setattr(discord.http.HTTPClient, "bulk_upsert_global_commands", AsyncMock(
        side_effect=AssertionError("Global synchronization is forbidden"),
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


def test_ping_registration_is_guild_only_and_idempotent() -> None:
    async def inspect() -> None:
        async with RallyBotClient(123456789012345678) as client:
            for _ in range(3):
                register_commands(client.tree, client.development_guild)
            commands = client.tree.get_commands(guild=client.development_guild)
            assert len(commands) == 7
            assert commands[0].name == "ping"
            assert commands[0].callback is ping
            assert client.tree.get_commands() == []
            assert client.tree.get_commands(guild=discord.Object(id=999)) == []
            assert not client.tree.fallback_to_global
    asyncio.run(inspect())


def test_ping_response_and_safe_invocation_log(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock()),
        token="fake-private-interaction-token", data={"private": "do-not-log"},
    )
    asyncio.run(ping(interaction))
    interaction.response.send_message.assert_awaited_once_with("Pong!")
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["message"] == "Discord ping invoked"
    assert "fake-private" not in output.err
    assert "do-not-log" not in output.err


def test_guild_configuration_environment_and_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert Settings().discord_guild_id is None
    (tmp_path / ".env").write_text("DISCORD_GUILD_ID=123456789012345678\n", encoding="utf-8")
    assert Settings().discord_guild_id == 123456789012345678
    monkeypatch.setenv("DISCORD_GUILD_ID", "234567890123456789")
    assert Settings().discord_guild_id == 234567890123456789


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_invalid_guild_configuration(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DISCORD_GUILD_ID", value)
    with pytest.raises(ValidationError):
        Settings()


def test_missing_guild_fails_before_connecting(monkeypatch: pytest.MonkeyPatch) -> None:
    start = AsyncMock()
    monkeypatch.setattr(RallyBotClient, "start", start)
    with pytest.raises(DiscordBotError, match="DISCORD_GUILD_ID"):
        start_discord_bot(Settings(discord_bot_token="fake-token"))
    start.assert_not_awaited()


def test_sync_once_per_client_not_on_ready(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO")
    async def inspect() -> None:
        for _ in range(2):
            async with RallyBotClient(123456789012345678) as client:
                sync = AsyncMock(return_value=[])
                monkeypatch.setattr(client.tree, "sync", sync)
                await client.setup_hook()
                await client.setup_hook()
                await client.on_ready()
                await client.on_ready()
                sync.assert_awaited_once_with(guild=client.development_guild)
                assert len(client.tree.get_commands(guild=client.development_guild)) == 7
    asyncio.run(inspect())
    messages = [json.loads(line)["message"] for line in capsys.readouterr().err.splitlines()]
    assert messages.count("Discord ping command registered") == 2
    assert messages.count("Discord development guild commands synchronized") == 2


def test_sync_uses_guild_endpoint_and_ping_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async def inspect() -> None:
        async with RallyBotClient(123456789012345678) as client:
            client._connection.application_id = 111111111111111111
            upsert = AsyncMock(return_value=[])
            monkeypatch.setattr(client.http, "bulk_upsert_guild_commands", upsert)
            await client.setup_hook()
            args, kwargs = upsert.await_args
            assert args == (111111111111111111, 123456789012345678)
            assert [command["name"] for command in kwargs["payload"]] == [
                "ping", "ladder", "register", "challenge", "accept", "decline", "cancel-challenge",
            ]
            client.http.bulk_upsert_global_commands.assert_not_awaited()
    asyncio.run(inspect())


def test_sync_failure_is_safe_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[RallyBotClient] = []
    async def start(self: RallyBotClient, token: str, *, reconnect: bool) -> None:
        clients.append(self)
        await self.setup_hook()
    monkeypatch.setattr(RallyBotClient, "start", start)
    monkeypatch.setattr(app_commands.CommandTree, "sync", AsyncMock(
        side_effect=RuntimeError("fake-private-token and response payload"),
    ))
    with pytest.raises(DiscordBotError) as caught:
        start_discord_bot(Settings(discord_bot_token="fake-token", discord_guild_id=123456789012345678))
    assert "fake-private-token" not in str(caught.value)
    assert clients[0].is_closed()
    assert not clients[0]._commands_synced


def test_command_error_handler_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    async def inspect() -> None:
        async with RallyBotClient(123456789012345678) as client:
            await client.tree.on_error(SimpleNamespace(token="fake-private-token"),
                                       app_commands.AppCommandError("fake-private-token"))
    asyncio.run(inspect())
    output = capsys.readouterr().err
    assert json.loads(output)["message"] == "Discord slash command failed"
    assert "fake-private-token" not in output


def test_cli_missing_guild_is_safe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = importlib.import_module("rallybot.main")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-private-token")
    assert module.main(["discord-start"]) == 1
    output = capsys.readouterr()
    assert "DISCORD_GUILD_ID" in output.err
    assert "fake-private-token" not in output.out + output.err
