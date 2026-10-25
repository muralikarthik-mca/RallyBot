"""Player registration rules independent of Discord presentation."""

from dataclasses import dataclass
from enum import StrEnum

from rallybot.config import get_settings
from rallybot.database.connection import DatabaseError
from rallybot.ladder.models import Player
from rallybot.ladder.repository import DuplicateDiscordUserError, PlayerRepository


class RegistrationStatus(StrEnum):
    SUCCESS = "SUCCESS"
    ALREADY_REGISTERED = "ALREADY_REGISTERED"
    INVALID_NAME = "INVALID_NAME"


@dataclass(frozen=True)
class RegistrationResult:
    status: RegistrationStatus
    player: Player | None = None


def register_player(
    discord_user_id: int, display_name: str, repository: PlayerRepository | None = None
) -> RegistrationResult:
    """Register one authenticated identity, preserving existing names and points."""
    name = display_name.strip()
    if not 1 <= len(name) <= 50:
        return RegistrationResult(RegistrationStatus.INVALID_NAME)
    if type(discord_user_id) is not int or discord_user_id <= 0:
        raise ValueError("A valid authenticated Discord identity is required.")
    repository = repository if repository is not None else PlayerRepository(get_settings().database_url)
    identity = str(discord_user_id)
    existing = repository.get_player_by_discord_user_id(identity)
    if existing is not None:
        return RegistrationResult(RegistrationStatus.ALREADY_REGISTERED, existing)
    try:
        player = repository.create_player(name, points=1000, discord_user_id=identity)
    except DuplicateDiscordUserError:
        # Another transaction may have won after our initial lookup. Its insert
        # is committed before our UNIQUE failure; fetch using a fresh connection.
        existing = repository.get_player_by_discord_user_id(identity)
        if existing is None:
            raise DatabaseError("Unable to verify existing registration.") from None
        return RegistrationResult(RegistrationStatus.ALREADY_REGISTERED, existing)
    return RegistrationResult(RegistrationStatus.SUCCESS, player)
