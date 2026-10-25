"""Two-player integration scenarios using real services and temporary SQLite.

Synthetic Discord identities exist only in this test database. No settings,
Discord client, network operation, or developer database is used.
"""

import logging
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from rallybot.challenges.models import Challenge, ChallengeStatus
from rallybot.challenges.repository import ChallengeRepository
from rallybot.challenges.service import ChallengeService, ChallengeValidationError
from rallybot.database.connection import initialize_database
from rallybot.ladder.models import Player
from rallybot.ladder.repository import PlayerRepository


CHALLENGER_DISCORD_ID = 900000000000000001
OPPONENT_DISCORD_ID = 900000000000000002


@dataclass(frozen=True)
class Simulation:
    database_url: str
    service: ChallengeService
    challenger: Player
    opponent: Player

    def reopen(self) -> ChallengeService:
        return ChallengeService(
            players=PlayerRepository(self.database_url),
            challenges=ChallengeRepository(self.database_url),
        )


@pytest.fixture
def simulation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Simulation]:
    def forbid_settings() -> None:
        pytest.fail("Integration scenarios must not load developer configuration")
    monkeypatch.setattr("rallybot.challenges.service.get_settings", forbid_settings)

    # Isolate service logs from handlers left by unrelated CLI tests.
    logger = logging.getLogger("rallybot")
    handlers, propagate = list(logger.handlers), logger.propagate
    for handler in handlers:
        logger.removeHandler(handler)
    sink = logging.NullHandler()
    logger.addHandler(sink)
    logger.propagate = False
    try:
        database_url = "sqlite:///" + (tmp_path / "two-player-integration.db").as_posix()
        initialize_database(database_url)
        players = PlayerRepository(database_url)
        challenger = players.create_player(
            "Test Challenger", discord_user_id=str(CHALLENGER_DISCORD_ID),
        )
        opponent = players.create_player(
            "Test Opponent", discord_user_id=str(OPPONENT_DISCORD_ID),
        )
        service = ChallengeService(players=players, challenges=ChallengeRepository(database_url))
        yield Simulation(database_url, service, challenger, opponent)
    finally:
        logger.removeHandler(sink)
        sink.close()
        for handler in handlers:
            logger.addHandler(handler)
        logger.propagate = propagate


def test_create_accept_and_persist_after_reopening(simulation: Simulation) -> None:
    service = simulation.service
    pending = service.create_challenge(CHALLENGER_DISCORD_ID, OPPONENT_DISCORD_ID)
    assert pending.status == ChallengeStatus.PENDING
    assert pending.challenger_player_id == simulation.challenger.id
    assert pending.opponent_player_id == simulation.opponent.id
    assert service.challenges.get_challenge(pending.id) == pending

    accepted = service.transition_challenge(OPPONENT_DISCORD_ID, pending.id, ChallengeStatus.ACCEPTED)
    assert accepted.status == ChallengeStatus.ACCEPTED
    assert accepted.id == pending.id

    reopened = simulation.reopen()
    assert reopened.challenges.get_challenge(pending.id) == accepted
    assert reopened.challenges.list_challenges(simulation.challenger.id) == [accepted]
    assert reopened.challenges.list_challenges(simulation.opponent.id) == [accepted]
    assert reopened.players.list_players() == [simulation.challenger, simulation.opponent]


def test_opponent_declines_separate_pending_challenge(simulation: Simulation) -> None:
    service = simulation.service
    accepted = service.create_challenge(CHALLENGER_DISCORD_ID, OPPONENT_DISCORD_ID)
    service.transition_challenge(OPPONENT_DISCORD_ID, accepted.id, ChallengeStatus.ACCEPTED)
    separate = service.create_challenge(CHALLENGER_DISCORD_ID, OPPONENT_DISCORD_ID)
    assert separate.id != accepted.id
    assert separate.status == ChallengeStatus.PENDING
    declined = service.transition_challenge(OPPONENT_DISCORD_ID, separate.id, ChallengeStatus.DECLINED)
    assert simulation.reopen().challenges.get_challenge(separate.id) == declined
    assert declined.status == ChallengeStatus.DECLINED
    assert service.challenges.get_challenge(accepted.id).status == ChallengeStatus.ACCEPTED


def test_challenger_cancels_separate_pending_challenge(simulation: Simulation) -> None:
    service = simulation.service
    declined = service.create_challenge(CHALLENGER_DISCORD_ID, OPPONENT_DISCORD_ID)
    service.transition_challenge(OPPONENT_DISCORD_ID, declined.id, ChallengeStatus.DECLINED)
    separate = service.create_challenge(CHALLENGER_DISCORD_ID, OPPONENT_DISCORD_ID)
    assert separate.id != declined.id
    assert separate.status == ChallengeStatus.PENDING
    cancelled = service.transition_challenge(CHALLENGER_DISCORD_ID, separate.id, ChallengeStatus.CANCELLED)
    assert simulation.reopen().challenges.get_challenge(separate.id) == cancelled
    assert cancelled.status == ChallengeStatus.CANCELLED
    assert service.challenges.get_challenge(declined.id).status == ChallengeStatus.DECLINED


@pytest.mark.parametrize(("actor", "action"), [
    (CHALLENGER_DISCORD_ID, ChallengeStatus.ACCEPTED),
    (OPPONENT_DISCORD_ID, ChallengeStatus.CANCELLED),
])
def test_wrong_role_cannot_modify_challenge(
    simulation: Simulation, actor: int, action: ChallengeStatus,
) -> None:
    pending = simulation.service.create_challenge(CHALLENGER_DISCORD_ID, OPPONENT_DISCORD_ID)
    with pytest.raises(ChallengeValidationError, match="not authorized"):
        simulation.service.transition_challenge(actor, pending.id, action)
    assert simulation.reopen().challenges.get_challenge(pending.id) == pending


@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_pending_challenges_are_rejected(simulation: Simulation, reverse: bool) -> None:
    pending = simulation.service.create_challenge(CHALLENGER_DISCORD_ID, OPPONENT_DISCORD_ID)
    challenger, opponent = (OPPONENT_DISCORD_ID, CHALLENGER_DISCORD_ID) if reverse else (
        CHALLENGER_DISCORD_ID, OPPONENT_DISCORD_ID,
    )
    with pytest.raises(ChallengeValidationError, match="pending challenge already exists"):
        simulation.service.create_challenge(challenger, opponent)
    assert simulation.reopen().challenges.list_challenges(simulation.challenger.id) == [pending]


@pytest.mark.parametrize("actor", [CHALLENGER_DISCORD_ID, OPPONENT_DISCORD_ID])
def test_self_challenges_are_rejected(simulation: Simulation, actor: int) -> None:
    with pytest.raises(ChallengeValidationError, match="cannot challenge yourself"):
        simulation.service.create_challenge(actor, actor)
    assert simulation.reopen().challenges.list_challenges(simulation.challenger.id) == []


@pytest.mark.parametrize("terminal", [ChallengeStatus.ACCEPTED, ChallengeStatus.DECLINED, ChallengeStatus.CANCELLED])
@pytest.mark.parametrize(("actor", "action"), [
    (OPPONENT_DISCORD_ID, ChallengeStatus.ACCEPTED),
    (OPPONENT_DISCORD_ID, ChallengeStatus.DECLINED),
    (CHALLENGER_DISCORD_ID, ChallengeStatus.CANCELLED),
])
def test_terminal_states_remain_unchanged(
    simulation: Simulation, terminal: ChallengeStatus, actor: int, action: ChallengeStatus,
) -> None:
    pending = simulation.service.create_challenge(CHALLENGER_DISCORD_ID, OPPONENT_DISCORD_ID)
    permitted_actor = CHALLENGER_DISCORD_ID if terminal == ChallengeStatus.CANCELLED else OPPONENT_DISCORD_ID
    final = simulation.service.transition_challenge(permitted_actor, pending.id, terminal)
    with pytest.raises(ChallengeValidationError, match="no longer pending"):
        simulation.service.transition_challenge(actor, pending.id, action)
    assert simulation.reopen().challenges.get_challenge(pending.id) == final


def test_concurrent_accept_decline_cannot_both_succeed(
    simulation: Simulation, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = simulation.service
    pending = service.create_challenge(CHALLENGER_DISCORD_ID, OPPONENT_DISCORD_ID)
    barrier = threading.Barrier(2)
    original_lookup = service.challenges.get_challenge

    def synchronized_lookup(challenge_id: int) -> Challenge | None:
        # Synchronize only the lookup timing. Every read, rule, and write still
        # executes in the existing production repository and service.
        result = original_lookup(challenge_id)
        if result is not None and result.status == ChallengeStatus.PENDING:
            barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(service.challenges, "get_challenge", synchronized_lookup)

    def attempt(action: ChallengeStatus) -> Challenge | ChallengeValidationError:
        try:
            return service.transition_challenge(OPPONENT_DISCORD_ID, pending.id, action)
        except ChallengeValidationError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(attempt, action) for action in (
            ChallengeStatus.ACCEPTED, ChallengeStatus.DECLINED,
        )]
        results = [future.result(timeout=10) for future in futures]

    successes = [result for result in results if isinstance(result, Challenge)]
    failures = [result for result in results if isinstance(result, ChallengeValidationError)]
    assert len(successes) == len(failures) == 1
    assert "no longer pending" in str(failures[0])
    assert simulation.reopen().challenges.get_challenge(pending.id) == successes[0]
    assert simulation.reopen().players.list_players() == [simulation.challenger, simulation.opponent]
