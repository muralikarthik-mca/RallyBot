"""Isolated persistence, migration, service, and development-seed tests."""

import importlib
import os
import sqlite3
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from rallybot.config import Settings, get_settings
from rallybot.database.connection import DatabaseError, database_connection, initialize_database
from rallybot.ladder.models import Player, SeedResult
from rallybot.ladder.repository import DuplicateDiscordUserError, PlayerRepository
from rallybot.ladder.service import SeedNotAllowedError, get_ladder, seed_development_ladder


@pytest.fixture(autouse=True)
def isolated_players(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields:
            monkeypatch.delenv(name)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///players.db")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest.fixture
def repository() -> PlayerRepository:
    url = get_settings().database_url
    initialize_database(url)
    return PlayerRepository(url)


def test_player_creation_retrieval_and_defaults(repository: PlayerRepository) -> None:
    player = repository.create_player("Alex", discord_user_id="test-discord-id")
    assert isinstance(player, Player)
    assert player.id > 0
    assert player.points == 1000
    assert player.discord_user_id == "test-discord-id"
    assert player.created_at.utcoffset() == timedelta(0)
    assert player.updated_at == player.created_at
    assert repository.get_player(player.id) == player
    assert repository.get_player(player.id + 1) is None
    assert repository.list_players() == [player]


def test_duplicate_discord_id_is_rejected_without_overwriting(repository: PlayerRepository) -> None:
    original = repository.create_player("Original", 1200, "private-test-id")
    with pytest.raises(DuplicateDiscordUserError) as error:
        repository.create_player("Replacement", 2000, "private-test-id")
    assert "private-test-id" not in str(error.value)
    assert repository.list_players() == [original]
    repository.create_player("Guest One")
    repository.create_player("Guest Two")
    assert len(repository.list_players()) == 3


@pytest.mark.parametrize("points", [-1, 1.5, True, "1000", 2**63])
def test_nonnegative_integer_points_validation(repository: PlayerRepository, points: object) -> None:
    with pytest.raises(ValueError, match="points"):
        repository.create_player("Invalid", points)
    assert repository.list_players() == []


@pytest.mark.parametrize("name", ["", "   "])
def test_required_display_name(repository: PlayerRepository, name: str) -> None:
    with pytest.raises(ValueError):
        repository.create_player(name)


def test_database_constraints_enforce_points_and_identity(repository: PlayerRepository) -> None:
    with database_connection(repository.database_url) as connection:
        for points in (-1, 1.5):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute("INSERT INTO players (display_name, points) VALUES (?, ?)", ("Invalid", points))
        connection.execute("INSERT INTO players (display_name) VALUES (?)", ("Default",))
        assert connection.execute("SELECT points FROM players").fetchone()[0] == 1000
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_ranking_order_and_stable_tie_breaking(repository: PlayerRepository) -> None:
    repository.create_player("First tie", 1200)
    repository.create_player("Lower", 0)
    repository.create_player("Second tie", 1200)
    repository.create_player("Leader", 1300)
    expected = [(1, "Leader", 1300), (2, "First tie", 1200),
                (3, "Second tie", 1200), (4, "Lower", 0)]
    for _ in range(2):
        assert [(p.rank, p.player_name, p.points) for p in repository.list_rankings()] == expected


def test_empty_ladder_and_service_injection(repository: PlayerRepository) -> None:
    assert get_ladder(repository) == []
    repository.create_player("Persisted", 1500)
    assert get_ladder(repository)[0].player_name == "Persisted"
    assert get_ladder()[0].points == 1500


def test_parameterized_sql_preserves_literal_names(repository: PlayerRepository) -> None:
    name = "Robert'); DROP TABLE players; --"
    player = repository.create_player(name)
    assert repository.get_player(player.id).display_name == name
    assert len(repository.list_players()) == 1


def test_additive_v1_migration_preserves_existing_data() -> None:
    url = get_settings().database_url
    with database_connection(url, create=True) as connection:
        connection.execute("CREATE TABLE _rallybot_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO _rallybot_metadata VALUES ('schema_version', '1')")
        connection.execute("CREATE TABLE existing_data (value TEXT)")
        connection.execute("INSERT INTO existing_data VALUES ('preserved')")
    initialize_database(url)
    repository = PlayerRepository(url)
    repository.create_player("Preserved player")
    initialize_database(url)
    with database_connection(url) as connection:
        assert connection.execute("SELECT value FROM existing_data").fetchone()[0] == "preserved"
        assert connection.execute("SELECT value FROM _rallybot_metadata WHERE key='schema_version'").fetchone()[0] == "4"
    assert len(repository.list_players()) == 1


def test_unknown_schema_version_is_not_downgraded(repository: PlayerRepository) -> None:
    with database_connection(repository.database_url) as connection:
        connection.execute("UPDATE _rallybot_metadata SET value='99' WHERE key='schema_version'")
    with pytest.raises(DatabaseError, match="schema version"):
        initialize_database(repository.database_url)
    with database_connection(repository.database_url) as connection:
        assert connection.execute("SELECT value FROM _rallybot_metadata WHERE key='schema_version'").fetchone()[0] == "99"


def test_seed_is_idempotent_without_overwrites(repository: PlayerRepository) -> None:
    original = repository.create_player("Alex Johnson", 42, "real-test-identity")
    assert seed_development_ladder(get_settings(), repository) == SeedResult(4, 1)
    assert seed_development_ladder(get_settings(), repository) == SeedResult(0, 5)
    players = repository.list_players()
    assert len(players) == 5
    assert repository.get_player(original.id) == original
    assert all(player.discord_user_id is None for player in players if player.id != original.id)


@pytest.mark.parametrize("environment", ["testing", "production"])
def test_seed_restricted_to_development(environment: str) -> None:
    settings = Settings(app_env=environment)
    with pytest.raises(SeedNotAllowedError):
        seed_development_ladder(settings)
    assert not Path("players.db").exists()


def test_seed_cli_counts_and_repeat(
    repository: PlayerRepository, capsys: pytest.CaptureFixture[str]
) -> None:
    module = importlib.import_module("rallybot.main")
    assert module.main(["ladder-seed"]) == 0
    assert "5 added, 0 skipped" in capsys.readouterr().out
    assert module.main(["ladder-seed"]) == 0
    assert "0 added, 5 skipped" in capsys.readouterr().out
    assert len(repository.list_players()) == 5


@pytest.mark.parametrize("environment", ["testing", "production"])
def test_seed_cli_restrictions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], environment: str
) -> None:
    module = importlib.import_module("rallybot.main")
    monkeypatch.setenv("APP_ENV", environment)
    assert module.main(["ladder-seed"]) == 1
    assert "only allowed in development" in capsys.readouterr().err
    assert not Path("players.db").exists()


def test_missing_database_is_not_created_by_retrieval() -> None:
    with pytest.raises(DatabaseError):
        get_ladder()
    assert not Path("players.db").exists()
