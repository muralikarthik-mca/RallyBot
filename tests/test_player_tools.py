"""Read-only player tools with isolated databases and explicit cleanup checks."""

import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

from rallybot.database.connection import initialize_database
from rallybot.ladder.repository import PlayerRepository
from rallybot.tools.player_tools import (
    PlayerSummary, RankedPlayer, UnknownPlayerError, create_player_tools,
    filter_opponents, get_registered_player, list_potential_opponents,
)


@pytest.fixture
def repository(tmp_path: Path) -> PlayerRepository:
    url = "sqlite:///" + (tmp_path / "player-tools.db").as_posix()
    initialize_database(url)
    return PlayerRepository(url)


def test_player_retrieval_does_not_expose_discord_identity(repository: PlayerRepository) -> None:
    player = repository.create_player("Name", 1000, "private-discord-identity")
    summary = get_registered_player(player.id, repository)
    assert asdict(summary) == {"id": player.id, "name": "Name", "points": 1000}
    assert "private-discord-identity" not in repr(summary)


def test_unknown_player_is_clear_and_safe(repository: PlayerRepository) -> None:
    for call in (get_registered_player, list_potential_opponents):
        with pytest.raises(UnknownPlayerError, match="not registered"):
            call(999, repository)


@pytest.mark.parametrize("player_id", [0, -1, True, "1", 1.5, 2**63])
def test_invalid_player_ids(repository: PlayerRepository, player_id: object) -> None:
    with pytest.raises(ValueError):
        get_registered_player(player_id, repository)


def test_self_exclusion_duplicate_names_and_ranking_reuse(repository: PlayerRepository) -> None:
    first = repository.create_player("Same Name", 1000)
    second = repository.create_player("Same Name", 1100)
    third = repository.create_player("Other", 900)
    result = list_potential_opponents(first.id, repository)
    assert result == [RankedPlayer(second.id, "Same Name", 1100, 1), RankedPlayer(third.id, "Other", 900, 3)]
    assert [p.name for p in result] == [r.player_name for r in repository.list_rankings() if r.rank != 2]


def test_inclusive_boundary_and_id_fallback() -> None:
    requester = PlayerSummary(1, "Same", 1000)
    candidates = [RankedPlayer(1, "Same", 1000, 1), RankedPlayer(4, "High", 1200, 2),
                  RankedPlayer(3, "Same", 800, 2), RankedPlayer(5, "Low", 799, 4),
                  RankedPlayer(6, "High", 1201, 1)]
    assert [p.id for p in filter_opponents(requester, candidates)] == [3, 4]
    assert len(candidates) == 5


@pytest.mark.parametrize("difference", [-1, True, 1.5])
def test_invalid_difference(difference: object) -> None:
    with pytest.raises(ValueError):
        filter_opponents(PlayerSummary(1, "Name", 1000), [], difference)


def test_tools_close_every_connection(repository: PlayerRepository, monkeypatch: pytest.MonkeyPatch) -> None:
    player = repository.create_player("Requester")
    repository.create_player("Opponent", 1100)
    original_connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []
    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)
        opened.append(connection)
        return connection
    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    tools = create_player_tools(repository)
    summary = tools.get_player(player.id)
    assert tools.filter_opponents(summary, tools.list_opponents(player.id))
    with pytest.raises(UnknownPlayerError):
        tools.get_player(999)
    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
