"""Read-only player tools; SQL and connection ownership stay in the repository."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rallybot.config import get_settings
from rallybot.ladder.repository import PlayerRepository


class UnknownPlayerError(ValueError):
    """The requested internal player ID does not exist."""


@dataclass(frozen=True)
class PlayerSummary:
    id: int
    name: str
    points: int


@dataclass(frozen=True)
class RankedPlayer(PlayerSummary):
    rank: int


def get_registered_player(player_id: int, repository: PlayerRepository | None = None) -> PlayerSummary:
    if type(player_id) is not int or not 1 <= player_id <= 9223372036854775807:
        raise ValueError("Player ID must be a positive SQLite integer.")
    repository = repository if repository is not None else PlayerRepository(get_settings().database_url)
    player = repository.get_player(player_id)
    if player is None:
        raise UnknownPlayerError("The requested player is not registered.")
    return PlayerSummary(player.id, player.display_name, player.points)


def list_potential_opponents(player_id: int, repository: PlayerRepository | None = None) -> list[RankedPlayer]:
    repository = repository if repository is not None else PlayerRepository(get_settings().database_url)
    get_registered_player(player_id, repository)
    return [RankedPlayer(player.id, player.display_name, player.points, rank)
            for rank, player in enumerate(repository.list_ranked_players(), start=1)
            if player.id != player_id]


def filter_opponents(
    player: PlayerSummary, opponents: Sequence[RankedPlayer], max_difference: int = 200
) -> list[RankedPlayer]:
    """Filter inclusively and order by distance, ladder rank, then player ID."""
    if type(max_difference) is not int or max_difference < 0:
        raise ValueError("Maximum point difference must be a nonnegative integer.")
    return sorted(
        (opponent for opponent in opponents
         if opponent.id != player.id and abs(opponent.points - player.points) <= max_difference),
        key=lambda opponent: (abs(opponent.points - player.points), opponent.rank, opponent.id),
    )


@dataclass(frozen=True)
class PlayerToolset:
    get_player: Callable[[int], PlayerSummary]
    list_opponents: Callable[[int], list[RankedPlayer]]
    filter_opponents: Callable[[PlayerSummary, Sequence[RankedPlayer]], list[RankedPlayer]]


def create_player_tools(repository: PlayerRepository | None = None) -> PlayerToolset:
    repository = repository if repository is not None else PlayerRepository(get_settings().database_url)
    return PlayerToolset(
        get_player=lambda player_id: get_registered_player(player_id, repository),
        list_opponents=lambda player_id: list_potential_opponents(player_id, repository),
        filter_opponents=filter_opponents,
    )
