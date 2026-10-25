"""Registered-player authorization and deterministic challenge rules."""

import logging
from typing import NoReturn

from rallybot.challenges.models import Challenge, ChallengeStatus
from rallybot.challenges.repository import ChallengeRepository, DuplicatePendingChallengeError
from rallybot.config import get_settings
from rallybot.ladder.models import Player
from rallybot.ladder.repository import PlayerRepository


class ChallengeValidationError(ValueError):
    """A safe validation failure suitable for a private Discord response."""


def _reject(message: str) -> NoReturn:
    logging.getLogger(__name__).warning("Challenge validation failed")
    raise ChallengeValidationError(message)


class ChallengeService:
    def __init__(
        self, *, players: PlayerRepository | None = None, challenges: ChallengeRepository | None = None
    ) -> None:
        if players is None or challenges is None:
            database_url = get_settings().database_url
        self.players = players if players is not None else PlayerRepository(database_url)
        self.challenges = challenges if challenges is not None else ChallengeRepository(database_url)

    def prepare_challenge(self, requester_id: int, opponent_id: int) -> tuple[Player, Player]:
        """Validate internal identities supplied by a trusted caller, without writes."""
        for player_id in (requester_id, opponent_id):
            if type(player_id) is not int or not 1 <= player_id <= 9223372036854775807:
                _reject("A valid registered player ID is required.")
        requester, opponent = self.players.get_player(requester_id), self.players.get_player(opponent_id)
        if requester is None or opponent is None:
            _reject("Both players must be registered.")
        if requester.id == opponent.id:
            _reject("You cannot challenge yourself.")
        if any(item.status == ChallengeStatus.PENDING and
               {item.challenger_player_id, item.opponent_player_id} == {requester.id, opponent.id}
               for item in self.challenges.list_challenges(requester.id)):
            _reject("A pending challenge already exists between these players.")
        return requester, opponent

    def create_approved_challenge(self, requester_id: int, opponent_id: int, workflow_id: str) -> Challenge:
        """Trusted approval entry point; never expose this directly to an LLM."""
        from uuid import UUID

        try:
            if str(UUID(workflow_id)) != workflow_id:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            _reject("Invalid approval workflow.")
        # Repository enforces pending-pair uniqueness and persists a replay receipt
        # in the same transaction. Replays may return an already-terminal challenge.
        for player_id in (requester_id, opponent_id):
            if type(player_id) is not int or not 1 <= player_id <= 9223372036854775807:
                _reject("A valid registered player ID is required.")
            if self.players.get_player(player_id) is None:
                _reject("Both players must be registered.")
        if requester_id == opponent_id:
            _reject("You cannot challenge yourself.")
        try:
            challenge = self.challenges.create_approved_challenge(requester_id, opponent_id, workflow_id)
        except DuplicatePendingChallengeError:
            _reject("A pending challenge already exists between these players.")
        logging.getLogger(__name__).info("Human-approved challenge creation completed")
        return challenge

    def list_challenges_for_player(self, player_id: int) -> list[Challenge]:
        """Read challenges for an internal identity supplied by trusted context."""
        if type(player_id) is not int or not 1 <= player_id <= 9223372036854775807:
            _reject("A valid requester player ID is required.")
        if self.players.get_player(player_id) is None:
            _reject("Requester player is not registered.")
        return self.challenges.list_challenges(player_id)

    def _registered_player(self, discord_user_id: int, message: str) -> Player:
        if type(discord_user_id) is not int or discord_user_id <= 0:
            _reject("Unable to verify Discord identity.")
        player = self.players.get_player_by_discord_user_id(str(discord_user_id))
        if player is None:
            _reject(message)
        return player

    def create_challenge(self, challenger_discord_id: int, opponent_discord_id: int) -> Challenge:
        challenger = self._registered_player(challenger_discord_id, "Please register before creating a challenge.")
        if challenger_discord_id == opponent_discord_id:
            _reject("You cannot challenge yourself.")
        opponent = self._registered_player(opponent_discord_id, "Your opponent must register before being challenged.")
        try:
            challenge = self.challenges.create_challenge(challenger.id, opponent.id)
        except DuplicatePendingChallengeError:
            _reject("A pending challenge already exists between these players.")
        logging.getLogger(__name__).info("Challenge created")
        return challenge

    def transition_challenge(
        self, actor_discord_id: int, challenge_id: int, status: ChallengeStatus
    ) -> Challenge:
        actor = self._registered_player(actor_discord_id, "Please register before managing challenges.")
        if type(challenge_id) is not int or not 1 <= challenge_id <= 9223372036854775807:
            _reject("Please provide a valid challenge ID.")
        challenge = self.challenges.get_challenge(challenge_id)
        if challenge is None:
            _reject("Challenge not found.")
        if status in (ChallengeStatus.ACCEPTED, ChallengeStatus.DECLINED):
            authorized = actor.id == challenge.opponent_player_id
        elif status == ChallengeStatus.CANCELLED:
            authorized = actor.id == challenge.challenger_player_id
        else:
            _reject("Unsupported challenge action.")
        if not authorized:
            _reject("You are not authorized to perform this challenge action.")
        if challenge.status != ChallengeStatus.PENDING:
            _reject("This challenge is no longer pending.")
        updated = self.challenges.transition_challenge(challenge_id, actor.id, status)
        if updated is None:
            _reject("This challenge is no longer pending.")
        logging.getLogger(__name__).info("Challenge %s", status.value.lower())
        return updated
