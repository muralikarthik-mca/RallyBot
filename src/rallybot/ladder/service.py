"""Ladder retrieval and explicitly requested development seeding."""

from rallybot.config import Settings, get_settings
from rallybot.ladder.models import Player, PlayerRanking, SeedResult
from rallybot.ladder.repository import PlayerRepository


class SeedNotAllowedError(ValueError):
    """Development seed data is not permitted in the selected environment."""


class PlayerResolutionError(ValueError):
    """A fixed unknown/ambiguous player resolution category."""


def resolve_player(repository: PlayerRepository, identity: int | str) -> Player:
    """Resolve an internal ID or an exact case-insensitive name without writes."""
    if type(identity) is int and 1 <= identity <= 9223372036854775807:
        player = repository.get_player(identity)
        if player is not None:
            return player
    elif isinstance(identity, str) and identity.strip():
        matches = [player for player in repository.list_players()
                   if player.display_name.casefold() == identity.strip().casefold()]
        if len(matches) > 1:
            raise PlayerResolutionError("AMBIGUOUS_OPPONENT")
        if matches:
            return matches[0]
    raise PlayerResolutionError("UNKNOWN_OPPONENT")


def get_ladder(repository: PlayerRepository | None = None) -> list[PlayerRanking]:
    """Retrieve persisted rankings; an empty database produces an empty ladder."""
    repository = repository if repository is not None else PlayerRepository(get_settings().database_url)
    return repository.list_rankings()


def seed_development_ladder(settings: Settings, repository: PlayerRepository | None = None) -> SeedResult:
    """Seed samples explicitly, preserving all existing players and points."""
    if settings.app_env != "development":
        raise SeedNotAllowedError("Ladder seeding is only allowed in development mode.")
    repository = repository if repository is not None else PlayerRepository(settings.database_url)
    return repository.seed_players(get_sample_ladder())


def get_sample_ladder() -> list[PlayerRanking]:
    """Return development seed data only; never used for production retrieval."""
    return [
        PlayerRanking(1, "Alex Johnson", 1200),
        PlayerRanking(2, "Sam Wilson", 1100),
        PlayerRanking(3, "Chris Lee", 1000),
        PlayerRanking(4, "Jordan Davis", 900),
        PlayerRanking(5, "Taylor Brown", 800),
    ]
