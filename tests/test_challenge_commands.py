"""Discord challenge presentation with fake interactions and local test data."""

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

from rallybot.challenges.models import Challenge, ChallengeStatus
from rallybot.challenges.service import ChallengeService
from rallybot.config import Settings, get_settings
from rallybot.database.connection import initialize_database
from rallybot.discord_bot import challenge_commands, commands
from rallybot.discord_bot.bot import RallyBotClient
from rallybot.logging_config import configure_logging


@pytest.fixture(autouse=True)
def isolated_challenge_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields:
            monkeypatch.delenv(name)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///discord-challenges.db")
    monkeypatch.setattr(discord.http.HTTPClient, "static_login", AsyncMock(
        side_effect=AssertionError("Real Discord login is forbidden in tests"),
    ))
    logger = logging.getLogger("rallybot")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    configure_logging("INFO")
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
def service() -> ChallengeService:
    initialize_database(get_settings().database_url)
    service = ChallengeService()
    service.players.create_player("Challenger Private Name", discord_user_id="111")
    service.players.create_player("Opponent Private Name", discord_user_id="222")
    return service


@pytest.fixture
def interaction() -> SimpleNamespace:
    return SimpleNamespace(
        user=SimpleNamespace(id=111), token="never-print-token",
        data={"player_id": 999, "private": "never-print-payload"},
        response=SimpleNamespace(defer=AsyncMock(), is_done=Mock(return_value=True), send_message=AsyncMock()),
        edit_original_response=AsyncMock(), followup=SimpleNamespace(send=AsyncMock()),
    )


def test_command_registration_options_and_no_global_commands() -> None:
    async def inspect() -> None:
        async with RallyBotClient(123456789012345678) as client:
            for _ in range(2):
                commands.register_commands(client.tree, client.development_guild)
            names = [command.name for command in client.tree.get_commands(guild=client.development_guild)]
            assert names == ["ping", "ladder", "register", "challenge", "accept", "decline", "cancel-challenge"]
            assert client.tree.get_commands() == []
            for name in ("challenge", "accept", "decline", "cancel-challenge"):
                command = client.tree.get_command(name, guild=client.development_guild)
                assert len(command.parameters) == 1
                parameter = command.parameters[0]
                assert parameter.required
                assert parameter.name == ("opponent" if name == "challenge" else "challenge_id")
                assert parameter.type == (discord.AppCommandOptionType.user if name == "challenge"
                                          else discord.AppCommandOptionType.integer)
    asyncio.run(inspect())


def test_challenge_creation_public_confirmation_and_safe_logs(
    service: ChallengeService, interaction: SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    asyncio.run(challenge_commands.challenge(interaction, SimpleNamespace(id=222)))
    interaction.response.defer.assert_awaited_once_with(thinking=True, ephemeral=True)
    interaction.edit_original_response.assert_awaited_once()
    kwargs = interaction.followup.send.await_args.kwargs
    assert kwargs["ephemeral"] is False
    assert kwargs["allowed_mentions"].to_dict()["parse"] == []
    assert interaction.followup.send.await_args.args == ("Challenge #1 created. Status: PENDING.",)
    stored = service.challenges.get_challenge(1)
    assert stored.challenger_player_id == service.players.get_player_by_discord_user_id("111").id
    assert stored.opponent_player_id == service.players.get_player_by_discord_user_id("222").id
    output = capsys.readouterr()
    assert "Challenge created" in output.err
    for value in ("never-print-token", "never-print-payload", "Private Name"):
        assert value not in output.out + output.err


@pytest.mark.parametrize(("command", "actor", "status"), [
    ("accept", 222, ChallengeStatus.ACCEPTED),
    ("decline", 222, ChallengeStatus.DECLINED),
    ("cancel_challenge", 111, ChallengeStatus.CANCELLED),
])
def test_action_public_success(
    service: ChallengeService, interaction: SimpleNamespace, command: str, actor: int, status: ChallengeStatus,
) -> None:
    challenge = service.create_challenge(111, 222)
    interaction.user.id = actor
    asyncio.run(getattr(challenge_commands, command)(interaction, challenge.id))
    interaction.response.defer.assert_awaited_once_with(thinking=True, ephemeral=True)
    assert interaction.followup.send.await_args.kwargs["ephemeral"] is False
    assert f"Status: {status.value}" in interaction.followup.send.await_args.args[0]
    assert service.challenges.get_challenge(challenge.id).status == status


@pytest.mark.parametrize(("actor", "opponent", "message"), [
    (444, 222, "register before creating"), (111, 444, "opponent must register"),
    (111, 111, "cannot challenge yourself"),
])
def test_creation_failures_stay_private(
    service: ChallengeService, interaction: SimpleNamespace, actor: int, opponent: int, message: str,
) -> None:
    interaction.user.id = actor
    asyncio.run(challenge_commands.challenge(interaction, SimpleNamespace(id=opponent)))
    interaction.response.defer.assert_awaited_once_with(thinking=True, ephemeral=True)
    assert message in interaction.edit_original_response.await_args.kwargs["content"]
    interaction.followup.send.assert_not_awaited()


def test_unauthorized_and_duplicate_failures_are_private(service: ChallengeService, interaction: SimpleNamespace) -> None:
    challenge = service.create_challenge(111, 222)
    asyncio.run(challenge_commands.accept(interaction, challenge.id))
    assert "not authorized" in interaction.edit_original_response.await_args.kwargs["content"]
    interaction.followup.send.assert_not_awaited()
    asyncio.run(challenge_commands.challenge(interaction, SimpleNamespace(id=222)))
    assert "pending challenge already exists" in interaction.edit_original_response.await_args.kwargs["content"]
    interaction.followup.send.assert_not_awaited()


def test_database_failures_are_sanitized(
    monkeypatch: pytest.MonkeyPatch, interaction: SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    monkeypatch.setattr(challenge_commands, "_execute", Mock(side_effect=RuntimeError("never-print-token")))
    asyncio.run(challenge_commands.challenge(interaction, SimpleNamespace(id=222)))
    assert interaction.edit_original_response.await_args.kwargs["content"] == (
        "Challenges are temporarily unavailable. Please try again later."
    )
    interaction.followup.send.assert_not_awaited()
    output = capsys.readouterr()
    assert "never-print-token" not in output.out + output.err
    assert "Traceback" not in output.err


def test_defer_failure_responds_privately(interaction: SimpleNamespace) -> None:
    interaction.response.defer.side_effect = RuntimeError("never-print-token")
    interaction.response.is_done.return_value = False
    asyncio.run(challenge_commands.accept(interaction, 1))
    assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True
    interaction.followup.send.assert_not_awaited()


def test_publication_failure_does_not_undo_committed_action(
    service: ChallengeService, interaction: SimpleNamespace, capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    interaction.followup.send.side_effect = RuntimeError("never-print-token")
    asyncio.run(challenge_commands.challenge(interaction, SimpleNamespace(id=222)))
    assert service.challenges.get_challenge(1).status == ChallengeStatus.PENDING
    assert "was saved as PENDING" in interaction.edit_original_response.await_args.kwargs["content"]
    assert "Do not repeat" in interaction.edit_original_response.await_args.kwargs["content"]
    assert "never-print-token" not in capsys.readouterr().err


def test_sqlite_work_runs_off_event_loop(
    monkeypatch: pytest.MonkeyPatch, service: ChallengeService, interaction: SimpleNamespace,
) -> None:
    loop_thread = threading.get_ident()
    acknowledged = threading.Event()
    async def defer(**kwargs: object) -> None:
        acknowledged.set()
    original = challenge_commands._execute
    def execute(actor_id: int, argument: int, status: ChallengeStatus | None) -> Challenge:
        assert acknowledged.is_set()
        assert threading.get_ident() != loop_thread
        return original(actor_id, argument, status)
    interaction.response.defer.side_effect = defer
    monkeypatch.setattr(challenge_commands, "_execute", execute)
    asyncio.run(challenge_commands.challenge(interaction, SimpleNamespace(id=222)))
    assert service.challenges.get_challenge(1) is not None
