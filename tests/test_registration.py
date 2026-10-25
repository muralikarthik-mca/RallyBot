"""Private player registration with isolated SQLite and mocked Discord APIs."""

import asyncio
import logging
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from rallybot.config import Settings, get_settings
from rallybot.database.connection import initialize_database
from rallybot.discord_bot import commands
from rallybot.discord_bot.bot import RallyBotClient
from rallybot.ladder.registration import RegistrationResult, RegistrationStatus, register_player
from rallybot.ladder.models import Player
from rallybot.ladder.repository import PlayerRepository
from rallybot.ladder.service import get_ladder, seed_development_ladder
from rallybot.logging_config import configure_logging


@pytest.fixture(autouse=True)
def isolated_registration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields:
            monkeypatch.delenv(name)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///registration.db")
    monkeypatch.setattr(discord.http.HTTPClient, "static_login", AsyncMock(
        side_effect=AssertionError("Real Discord login is forbidden in tests"),
    ))
    logger = logging.getLogger("rallybot")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    get_settings.cache_clear()
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
def repository() -> PlayerRepository:
    initialize_database(get_settings().database_url)
    return PlayerRepository(get_settings().database_url)


@pytest.fixture
def interaction() -> SimpleNamespace:
    return SimpleNamespace(
        user=SimpleNamespace(id=123456789012345678),
        response=SimpleNamespace(defer=AsyncMock(), is_done=Mock(return_value=True), send_message=AsyncMock()),
        edit_original_response=AsyncMock(),
        token="fake-private-token", data={"discord_user_id": "not-authenticated"},
    )


def test_success_trim_starting_points_and_identity(repository: PlayerRepository) -> None:
    result = register_player(123456789012345678, "  New Player  ", repository)
    assert result.status == RegistrationStatus.SUCCESS
    assert result.player.display_name == "New Player"
    assert result.player.points == 1000
    assert result.player.discord_user_id == "123456789012345678"
    assert repository.get_player_by_discord_user_id("123456789012345678") == result.player
    assert repository.get_player_by_discord_user_id("other-identity") is None


def test_duplicate_preserves_existing_name_points_and_timestamps(repository: PlayerRepository) -> None:
    existing = repository.create_player("Original Name", 1700, "123")
    result = register_player(123, "Replacement Name", repository)
    assert result.status == RegistrationStatus.ALREADY_REGISTERED
    assert result.player == existing
    assert repository.list_players() == [existing]


def test_same_name_different_users_and_seeded_players_are_preserved(repository: PlayerRepository) -> None:
    seed_development_ladder(get_settings(), repository)
    before = repository.list_players()
    first = register_player(111, "Alex Johnson", repository)
    second = register_player(222, "Alex Johnson", repository)
    assert first.status == second.status == RegistrationStatus.SUCCESS
    assert first.player.id != second.player.id
    assert repository.list_players()[:5] == before
    assert len(repository.list_players()) == 7


@pytest.mark.parametrize("name", ["", " ", "\t\n", "a" * 51, " " + "a" * 51 + " "])
def test_invalid_name_does_not_create_or_query_database(name: str) -> None:
    result = register_player(123, name)
    assert result.status == RegistrationStatus.INVALID_NAME
    assert result.player is None
    assert not Path("registration.db").exists()


@pytest.mark.parametrize("name", ["A", "a" * 50, "  " + "a" * 50 + "  "])
def test_name_length_boundaries_after_trimming(repository: PlayerRepository, name: str) -> None:
    result = register_player(123, name, repository)
    assert result.status == RegistrationStatus.SUCCESS
    assert result.player.display_name == name.strip()


def test_identity_lookup_is_parameterized(repository: PlayerRepository) -> None:
    register_player(123, "Private Name", repository)
    assert repository.get_player_by_discord_user_id("123' OR 1=1 --") is None


def test_concurrent_registration_handles_the_unique_constraint(repository: PlayerRepository) -> None:
    barrier = threading.Barrier(2)

    class RacingRepository(PlayerRepository):
        def get_player_by_discord_user_id(self, discord_user_id: str) -> Player | None:
            player = super().get_player_by_discord_user_id(discord_user_id)
            if player is None:
                barrier.wait(timeout=5)
            return player

    racing_repository = RacingRepository(repository.database_url)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(register_player, 123, name, racing_repository)
                   for name in ("First Attempt", "Second Attempt")]
        results = [future.result(timeout=10) for future in futures]
    assert {result.status for result in results} == {
        RegistrationStatus.SUCCESS, RegistrationStatus.ALREADY_REGISTERED,
    }
    assert results[0].player == results[1].player
    assert results[0].player.points == 1000
    assert len(repository.list_players()) == 1


def test_register_command_has_only_required_name_and_is_not_duplicated() -> None:
    async def inspect() -> None:
        async with RallyBotClient(123456789012345678) as client:
            for _ in range(2):
                commands.register_commands(client.tree, client.development_guild)
            registered = client.tree.get_commands(guild=client.development_guild)
            assert [command.name for command in registered] == [
                "ping", "ladder", "register", "challenge", "accept", "decline", "cancel-challenge",
            ]
            command = client.tree.get_command("register", guild=client.development_guild)
            assert [parameter.name for parameter in command.parameters] == ["name"]
            assert command.parameters[0].required
            assert command.parameters[0].type == discord.AppCommandOptionType.string
            assert command.callback is commands.register
            assert client.tree.get_commands() == []
    asyncio.run(inspect())


def test_discord_registration_success_is_private_and_uses_authenticated_identity(
    repository: PlayerRepository, interaction: SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    asyncio.run(commands.register(interaction, "  My Name  "))
    interaction.response.defer.assert_awaited_once_with(thinking=True, ephemeral=True)
    kwargs = interaction.edit_original_response.await_args.kwargs
    assert kwargs["content"] == "Registration successful! Welcome to RallyBot, My Name. Starting points: 1,000."
    assert kwargs["allowed_mentions"].to_dict()["parse"] == []
    assert repository.get_player_by_discord_user_id(str(interaction.user.id)).display_name == "My Name"
    assert repository.get_player_by_discord_user_id("not-authenticated") is None
    output = capsys.readouterr()
    for value in ("My Name", str(interaction.user.id), "fake-private-token", "not-authenticated"):
        assert value not in output.out + output.err


def test_discord_duplicate_shows_only_own_existing_name(
    repository: PlayerRepository, interaction: SimpleNamespace,
) -> None:
    register_player(interaction.user.id, "My Existing Name", repository)
    register_player(999, "Other User Private Name", repository)
    asyncio.run(commands.register(interaction, "New Name"))
    interaction.response.defer.assert_awaited_once_with(thinking=True, ephemeral=True)
    message = interaction.edit_original_response.await_args.kwargs["content"]
    assert message == "You are already registered as My Existing Name."
    assert "Other User Private Name" not in message


@pytest.mark.parametrize("name", ["", "   ", "a" * 51])
def test_discord_invalid_names_are_private(interaction: SimpleNamespace, name: str) -> None:
    asyncio.run(commands.register(interaction, name))
    interaction.response.defer.assert_awaited_once_with(thinking=True, ephemeral=True)
    assert interaction.edit_original_response.await_args.kwargs["content"] == (
        "Please provide a valid name (1–50 characters)."
    )
    assert not Path("registration.db").exists()


def test_registration_is_deferred_before_worker_thread(
    monkeypatch: pytest.MonkeyPatch, repository: PlayerRepository, interaction: SimpleNamespace,
) -> None:
    loop_thread = threading.get_ident()
    acknowledged = threading.Event()
    async def defer(**kwargs: object) -> None:
        acknowledged.set()
    def registration(identity: int, name: str) -> RegistrationResult:
        assert acknowledged.is_set()
        assert threading.get_ident() != loop_thread
        return register_player(identity, name, repository)
    interaction.response.defer.side_effect = defer
    monkeypatch.setattr(commands, "register_player", registration)
    asyncio.run(commands.register(interaction, "Worker Player"))
    assert repository.list_players()[0].display_name == "Worker Player"


def test_database_failure_is_safe_and_does_not_create_database(
    interaction: SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    asyncio.run(commands.register(interaction, "My Name"))
    assert interaction.edit_original_response.await_args.kwargs["content"] == (
        "Registration is temporarily unavailable. Please try again later."
    )
    output = capsys.readouterr()
    assert "registration.db" not in output.out + output.err
    assert not Path("registration.db").exists()


def test_unexpected_failure_and_failed_error_response_are_safe(
    monkeypatch: pytest.MonkeyPatch, interaction: SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    monkeypatch.setattr(commands, "register_player", Mock(side_effect=RuntimeError("fake-private-token")))
    interaction.edit_original_response.side_effect = RuntimeError("fake-private-token")
    asyncio.run(commands.register(interaction, "My Name"))
    output = capsys.readouterr()
    assert "Discord registration error response failed" in output.err
    assert "fake-private-token" not in output.err
    assert "Traceback" not in output.err


def test_defer_failure_responds_privately(interaction: SimpleNamespace) -> None:
    interaction.response.defer.side_effect = RuntimeError("fake-private-token")
    interaction.response.is_done.return_value = False
    asyncio.run(commands.register(interaction, "My Name"))
    kwargs = interaction.response.send_message.await_args.kwargs
    assert kwargs["ephemeral"] is True
    assert interaction.response.send_message.await_args.args == (
        "Registration is temporarily unavailable. Please try again later.",
    )


def test_registered_player_is_visible_in_ladder_and_ping_still_works(
    repository: PlayerRepository, interaction: SimpleNamespace,
) -> None:
    asyncio.run(commands.register(interaction, "New Tennis Player"))
    assert [(p.player_name, p.points) for p in get_ladder(repository)] == [("New Tennis Player", 1000)]
    interaction.edit_original_response.reset_mock()
    asyncio.run(commands.ladder(interaction))
    embed = interaction.edit_original_response.await_args.kwargs["embed"]
    assert embed.fields[0].name == "1. New Tennis Player"
    assert embed.fields[0].value == "1,000 points"
    asyncio.run(commands.ping(interaction))
    interaction.response.send_message.assert_awaited_once_with("Pong!")
