"""Sample ladder and Discord presentation tests without live services."""

import asyncio
import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from rallybot.discord_bot import commands
from rallybot.discord_bot.bot import RallyBotClient
from rallybot.ladder.models import PlayerRanking
from rallybot.ladder.service import get_sample_ladder
from rallybot.ladder.service import seed_development_ladder
from rallybot.config import Settings, get_settings
from rallybot.database.connection import initialize_database
from rallybot.ladder.repository import PlayerRepository
from rallybot.logging_config import configure_logging


@pytest.fixture(autouse=True)
def isolated_ladder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields:
            monkeypatch.delenv(name)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
    get_settings.cache_clear()
    initialize_database(get_settings().database_url)
    seed_development_ladder(get_settings())
    monkeypatch.setattr(discord.http.HTTPClient, "static_login", AsyncMock(
        side_effect=AssertionError("Real Discord login is forbidden in tests"),
    ))
    logger = logging.getLogger("rallybot")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    try:
        yield
    finally:
        get_settings.cache_clear()
        for handler in list(logger.handlers):
            if handler not in handlers:
                logger.removeHandler(handler)
                handler.close()
        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate


@pytest.fixture
def interaction() -> SimpleNamespace:
    return SimpleNamespace(
        response=SimpleNamespace(
            defer=AsyncMock(), is_done=Mock(return_value=True), send_message=AsyncMock(),
        ),
        edit_original_response=AsyncMock(),
        token="fake-private-token", data={"private": "never-log-this"},
    )


def test_sample_ladder_values_and_rank_order() -> None:
    before = set(Path.cwd().iterdir())
    rankings = get_sample_ladder()
    assert len(rankings) == 5
    assert all(isinstance(player, PlayerRanking) for player in rankings)
    assert [(player.rank, player.player_name, player.points) for player in rankings] == [
        (1, "Alex Johnson", 1200),
        (2, "Sam Wilson", 1100),
        (3, "Chris Lee", 1000),
        (4, "Jordan Davis", 900),
        (5, "Taylor Brown", 800),
    ]
    assert set(Path.cwd().iterdir()) == before


def test_sample_results_do_not_share_mutable_list_state() -> None:
    first = get_sample_ladder()
    first.pop()
    assert len(get_sample_ladder()) == 5


def test_embed_contents_label_and_discord_limits() -> None:
    embed = commands.build_ladder_embed(get_sample_ladder())
    assert "Development Sample Data" not in embed.description
    assert embed.title == "Tennis Ladder"
    assert len(embed.fields) == 5
    for field, player in zip(embed.fields, get_sample_ladder(), strict=True):
        assert field.name == f"{player.rank}. {player.player_name}"
        assert field.value == f"{player.points:,} points"
        assert field.inline is False
        assert len(field.name) <= 256
        assert len(field.value) <= 1024
    assert len(embed.title) <= 256
    assert len(embed.description) <= 4096
    assert len(embed.fields) <= 25
    assert len(embed) <= 6000


def test_ladder_registration_preserves_ping_without_duplicates() -> None:
    async def inspect() -> None:
        async with RallyBotClient(123456789012345678) as client:
            for _ in range(3):
                commands.register_commands(client.tree, client.development_guild)
            registered = client.tree.get_commands(guild=client.development_guild)
            assert [command.name for command in registered] == [
                "ping", "ladder", "register", "challenge", "accept", "decline", "cancel-challenge",
            ]
            assert client.tree.get_command("ladder", guild=client.development_guild).callback is commands.ladder
            assert client.tree.get_command("ping", guild=client.development_guild).callback is commands.ping
            assert client.tree.get_commands() == []
    asyncio.run(inspect())


def test_ladder_defers_before_service_and_sends_embed(
    monkeypatch: pytest.MonkeyPatch, interaction: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    events: list[str] = []
    event_loop_thread = threading.get_ident()
    async def acknowledge(*, thinking: bool) -> None:
        assert thinking
        events.append("deferred")
    def service() -> list[PlayerRanking]:
        assert threading.get_ident() != event_loop_thread
        assert events == ["deferred"]
        events.append("service")
        return get_sample_ladder()
    interaction.response.defer.side_effect = acknowledge
    monkeypatch.setattr(commands, "get_ladder", service)
    asyncio.run(commands.ladder(interaction))
    interaction.response.defer.assert_awaited_once_with(thinking=True)
    interaction.edit_original_response.assert_awaited_once()
    embed = interaction.edit_original_response.await_args.kwargs["embed"]
    assert isinstance(embed, discord.Embed)
    assert len(embed.fields) == 5
    assert events == ["deferred", "service"]
    output = capsys.readouterr()
    assert "Discord ladder invoked" in output.err
    assert "fake-private-token" not in output.err
    assert "never-log-this" not in output.err


def test_discord_ladder_retrieves_database_players(interaction: SimpleNamespace) -> None:
    repository = PlayerRepository(get_settings().database_url)
    repository.create_player("Database Player", 2000)
    asyncio.run(commands.ladder(interaction))
    embed = interaction.edit_original_response.await_args.kwargs["embed"]
    assert embed.fields[0].name == "1. Database Player"
    assert embed.fields[0].value == "2,000 points"
    assert len(embed.fields) == 6
    assert "Development Sample Data" not in embed.description


def test_discord_empty_ladder_message(
    monkeypatch: pytest.MonkeyPatch, interaction: SimpleNamespace
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite:///empty.db")
    get_settings.cache_clear()
    initialize_database(get_settings().database_url)
    asyncio.run(commands.ladder(interaction))
    interaction.edit_original_response.assert_awaited_once_with(
        content="The ladder is empty. No players have been added yet.",
    )


def test_discord_database_failure_is_safe(
    monkeypatch: pytest.MonkeyPatch, interaction: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///missing-private-directory/players.db")
    get_settings.cache_clear()
    asyncio.run(commands.ladder(interaction))
    assert "Unable to display" in interaction.edit_original_response.await_args.kwargs["content"]
    output = capsys.readouterr()
    assert "missing-private-directory" not in output.out + output.err
    assert not Path("missing-private-directory").exists()


def test_large_ladder_embed_remains_within_limits() -> None:
    rankings = [PlayerRanking(rank, "Very long name " * 100, 1000) for rank in range(1, 41)]
    embed = commands.build_ladder_embed(rankings)
    assert 0 < len(embed.fields) <= 25
    assert len(embed) <= 6000
    assert all(len(field.name) <= 256 for field in embed.fields)
    assert "of 40 players" in embed.footer.text


def test_ping_still_returns_pong(interaction: SimpleNamespace) -> None:
    asyncio.run(commands.ping(interaction))
    interaction.response.send_message.assert_awaited_once_with("Pong!")
    interaction.response.defer.assert_not_awaited()


@pytest.mark.parametrize("failure_point", ["service", "embed", "send"])
def test_ladder_failures_are_safe(
    monkeypatch: pytest.MonkeyPatch, interaction: SimpleNamespace,
    capsys: pytest.CaptureFixture[str], failure_point: str,
) -> None:
    configure_logging("INFO")
    error = RuntimeError("credentials=fake-private-token never-log-this")
    if failure_point == "service":
        monkeypatch.setattr(commands, "get_ladder", Mock(side_effect=error))
    elif failure_point == "embed":
        monkeypatch.setattr(commands, "build_ladder_embed", Mock(side_effect=error))
    else:
        interaction.edit_original_response.side_effect = [error, None]
    asyncio.run(commands.ladder(interaction))
    last_call = interaction.edit_original_response.await_args
    assert last_call.kwargs == {
        "content": "Unable to display the ladder right now. Please try again later.",
        "embed": None,
    }
    output = capsys.readouterr()
    assert "Discord ladder command failed" in output.err
    for value in ("fake-private-token", "never-log-this", "Traceback"):
        assert value not in output.out + output.err + repr(last_call)


def test_defer_failure_uses_safe_initial_response(
    interaction: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO")
    interaction.response.defer.side_effect = RuntimeError("fake-private-token")
    interaction.response.is_done.return_value = False
    asyncio.run(commands.ladder(interaction))
    interaction.response.send_message.assert_awaited_once_with(
        "Unable to display the ladder right now. Please try again later.", ephemeral=True,
    )
    interaction.edit_original_response.assert_not_awaited()
    assert "fake-private-token" not in capsys.readouterr().err


def test_error_response_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, interaction: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    monkeypatch.setattr(commands, "get_ladder", Mock(side_effect=RuntimeError("fake-private-token")))
    interaction.edit_original_response.side_effect = RuntimeError("fake-private-token")
    asyncio.run(commands.ladder(interaction))
    output = capsys.readouterr()
    assert "Discord ladder error response failed" in output.err
    assert "fake-private-token" not in output.err
