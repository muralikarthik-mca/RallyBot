"""Deterministic challenge persistence, authorization, and concurrency tests."""

import logging
import os
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest

from rallybot.challenges.models import Challenge, ChallengeStatus
from rallybot.challenges.repository import ChallengeRepository, DuplicatePendingChallengeError
from rallybot.challenges.service import ChallengeService, ChallengeValidationError
from rallybot.config import Settings, get_settings
from rallybot.database.connection import DatabaseError, database_connection, initialize_database
from rallybot.ladder.repository import PlayerRepository
from rallybot.logging_config import configure_logging


@pytest.fixture(autouse=True)
def isolated_challenges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields:
            monkeypatch.delenv(name)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///challenges.db")
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
    for identity, name in ((111, "Challenger"), (222, "Opponent"), (333, "Unrelated")):
        service.players.create_player(name, discord_user_id=str(identity))
    return service


def test_creation_retrieval_listing_and_reopening(service: ChallengeService) -> None:
    players_before = service.players.list_players()
    result = service.create_challenge(111, 222)
    assert result.status == ChallengeStatus.PENDING
    assert result.challenger_player_id == players_before[0].id
    assert result.opponent_player_id == players_before[1].id
    assert result.created_at.utcoffset() == timedelta(0)
    assert result.created_at == result.updated_at
    reopened = ChallengeRepository(get_settings().database_url)
    assert reopened.get_challenge(result.id) == result
    assert reopened.get_challenge(999) is None
    assert reopened.list_challenges(players_before[0].id) == [result]
    assert reopened.list_challenges(players_before[1].id) == [result]
    assert reopened.list_challenges(players_before[2].id) == []
    assert service.players.list_players() == players_before


@pytest.mark.parametrize(("challenger", "opponent", "message"), [
    (444, 222, "register before creating"), (111, 444, "opponent must register"),
    (111, 111, "cannot challenge yourself"),
])
def test_creation_validation(service: ChallengeService, challenger: int, opponent: int, message: str) -> None:
    with pytest.raises(ChallengeValidationError, match=message):
        service.create_challenge(challenger, opponent)
    assert service.challenges.list_challenges(1) == []


@pytest.mark.parametrize(("challenger", "opponent"), [(111, 222), (222, 111)])
def test_duplicate_pending_pair_in_both_directions(
    service: ChallengeService, challenger: int, opponent: int
) -> None:
    original = service.create_challenge(111, 222)
    with pytest.raises(ChallengeValidationError, match="pending challenge already exists"):
        service.create_challenge(challenger, opponent)
    assert service.challenges.list_challenges(original.challenger_player_id) == [original]


@pytest.mark.parametrize(("status", "actor"), [
    (ChallengeStatus.ACCEPTED, 222), (ChallengeStatus.DECLINED, 222), (ChallengeStatus.CANCELLED, 111),
])
def test_valid_transitions_are_persisted_and_do_not_update_points(
    service: ChallengeService, status: ChallengeStatus, actor: int,
) -> None:
    players_before = service.players.list_players()
    original = service.create_challenge(111, 222)
    updated = service.transition_challenge(actor, original.id, status)
    assert updated.status == status
    assert updated.id == original.id
    assert updated.created_at == original.created_at
    assert updated.updated_at >= original.updated_at
    assert updated.updated_at.utcoffset() == timedelta(0)
    assert service.challenges.get_challenge(original.id) == updated
    assert service.players.list_players() == players_before
    # Only pending challenges block a new challenge for this pair.
    assert service.create_challenge(222, 111).status == ChallengeStatus.PENDING


@pytest.mark.parametrize(("status", "actor"), [
    (ChallengeStatus.ACCEPTED, 111), (ChallengeStatus.DECLINED, 111),
    (ChallengeStatus.CANCELLED, 222), (ChallengeStatus.ACCEPTED, 333),
    (ChallengeStatus.DECLINED, 333), (ChallengeStatus.CANCELLED, 333),
])
def test_unauthorized_transitions(service: ChallengeService, status: ChallengeStatus, actor: int) -> None:
    original = service.create_challenge(111, 222)
    with pytest.raises(ChallengeValidationError, match="not authorized"):
        service.transition_challenge(actor, original.id, status)
    assert service.challenges.get_challenge(original.id) == original


@pytest.mark.parametrize("final_status", [ChallengeStatus.ACCEPTED, ChallengeStatus.DECLINED, ChallengeStatus.CANCELLED])
@pytest.mark.parametrize(("next_status", "actor"), [
    (ChallengeStatus.ACCEPTED, 222), (ChallengeStatus.DECLINED, 222), (ChallengeStatus.CANCELLED, 111),
])
def test_terminal_states_cannot_transition(
    service: ChallengeService, final_status: ChallengeStatus, next_status: ChallengeStatus, actor: int,
) -> None:
    original = service.create_challenge(111, 222)
    final_actor = 111 if final_status == ChallengeStatus.CANCELLED else 222
    final = service.transition_challenge(final_actor, original.id, final_status)
    with pytest.raises(ChallengeValidationError, match="no longer pending"):
        service.transition_challenge(actor, original.id, next_status)
    assert service.challenges.get_challenge(original.id) == final


def test_missing_unregistered_and_invalid_actions(service: ChallengeService) -> None:
    original = service.create_challenge(111, 222)
    with pytest.raises(ChallengeValidationError, match="register before managing"):
        service.transition_challenge(444, original.id, ChallengeStatus.ACCEPTED)
    with pytest.raises(ChallengeValidationError, match="not found"):
        service.transition_challenge(222, 999, ChallengeStatus.ACCEPTED)
    for challenge_id in (0, -1, 2**63):
        with pytest.raises(ChallengeValidationError, match="valid challenge ID"):
            service.transition_challenge(222, challenge_id, ChallengeStatus.ACCEPTED)
    with pytest.raises(ChallengeValidationError, match="Unsupported"):
        service.transition_challenge(222, original.id, ChallengeStatus.PENDING)


def test_database_constraints_and_repository_authorization(service: ChallengeService) -> None:
    original = service.create_challenge(111, 222)
    with database_connection(get_settings().database_url) as connection:
        for challenger, opponent, status in ((1, 1, "PENDING"), (1, 999, "PENDING"), (1, 2, "BOGUS")):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO challenges (challenger_player_id, opponent_player_id, status) VALUES (?, ?, ?)",
                    (challenger, opponent, status),
                )
    assert service.challenges.transition_challenge(original.id, 3, ChallengeStatus.ACCEPTED) is None
    assert service.challenges.transition_challenge(original.id, 1, ChallengeStatus.ACCEPTED) is None
    assert service.challenges.transition_challenge(original.id, 2, ChallengeStatus.CANCELLED) is None
    with pytest.raises(ValueError):
        service.challenges.transition_challenge(original.id, 2, ChallengeStatus.PENDING)
    with pytest.raises(DuplicatePendingChallengeError):
        service.challenges.create_challenge(2, 1)
    assert service.challenges.get_challenge(original.id) == original


def test_concurrent_opposing_creation_has_one_winner(service: ChallengeService) -> None:
    barrier = threading.Barrier(2)
    def attempt(challenger: int, opponent: int) -> Challenge | None:
        barrier.wait(timeout=5)
        try:
            return service.create_challenge(challenger, opponent)
        except ChallengeValidationError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(attempt, 111, 222), executor.submit(attempt, 222, 111)]
        results = [future.result(timeout=10) for future in futures]
    assert sum(result is not None for result in results) == 1
    assert len(service.challenges.list_challenges(1)) == 1


def test_concurrent_accept_decline_has_one_atomic_winner(service: ChallengeService) -> None:
    original = service.create_challenge(111, 222)
    barrier = threading.Barrier(2)
    class RacingRepository(ChallengeRepository):
        def get_challenge(self, challenge_id: int) -> Challenge | None:
            result = super().get_challenge(challenge_id)
            if result is not None and result.status == ChallengeStatus.PENDING:
                barrier.wait(timeout=5)
            return result
    racing = ChallengeService(players=service.players, challenges=RacingRepository(get_settings().database_url))
    def attempt(status: ChallengeStatus) -> Challenge | None:
        try:
            return racing.transition_challenge(222, original.id, status)
        except ChallengeValidationError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(attempt, status) for status in (ChallengeStatus.ACCEPTED, ChallengeStatus.DECLINED)]
        results = [future.result(timeout=10) for future in futures]
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert service.challenges.get_challenge(original.id) == winners[0]


def test_version_two_migration_and_repeat_preserve_players_and_challenges(service: ChallengeService) -> None:
    # Recreate the previous schema state using a separate temporary database.
    url = "sqlite:///old.db"
    with database_connection(url, create=True) as connection:
        connection.execute("CREATE TABLE _rallybot_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO _rallybot_metadata VALUES ('schema_version', '2')")
        connection.execute("CREATE TABLE players (id INTEGER PRIMARY KEY, discord_user_id TEXT UNIQUE, display_name TEXT NOT NULL, points INTEGER NOT NULL DEFAULT 1000, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
        connection.execute("INSERT INTO players VALUES (1, '111', 'Preserved', 1400, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')")
    initialize_database(url)
    assert PlayerRepository(url).get_player(1).points == 1400
    initialize_database(url)
    challenge = service.create_challenge(111, 222)
    initialize_database(get_settings().database_url)
    assert service.challenges.get_challenge(challenge.id) == challenge
    with database_connection(url) as connection:
        assert connection.execute("SELECT value FROM _rallybot_metadata WHERE key='schema_version'").fetchone()[0] == "4"


def test_missing_database_is_not_created() -> None:
    with pytest.raises(DatabaseError):
        ChallengeService().create_challenge(111, 222)
    assert not Path("challenges.db").exists()
