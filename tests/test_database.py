"""SQLite infrastructure tests using only private temporary databases."""

import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rallybot.config import Settings, get_settings
from rallybot.database.connection import (
    DatabaseError,
    InvalidDatabaseURL,
    check_database_health,
    connect_database,
    database_connection,
    database_path,
    initialize_database,
)


@pytest.fixture(autouse=True)
def isolated_database_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields:
            monkeypatch.delenv(name)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest.fixture
def database_url() -> str:
    return "sqlite:///./nested/test.db"


def test_connection_foreign_keys_rows_and_timeout(database_url: str) -> None:
    initialize_database(database_url)
    with database_connection(database_url) as connection:
        assert connection.execute("SELECT 1 AS value").fetchone()["value"] == 1
        assert connection.row_factory is sqlite3.Row
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connection.execute("CREATE TABLE child (parent_id INTEGER REFERENCES parent(id))")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO child VALUES (99)")
    with database_connection(database_url) as second:
        assert second.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_successful_transaction_commits(database_url: str) -> None:
    initialize_database(database_url)
    with database_connection(database_url) as connection:
        connection.execute("CREATE TABLE sample (value TEXT)")
        connection.execute("INSERT INTO sample VALUES (?)", ("committed",))
    with database_connection(database_url) as connection:
        assert connection.execute("SELECT value FROM sample").fetchone()[0] == "committed"


def test_failed_transaction_rolls_back_and_closes(database_url: str) -> None:
    initialize_database(database_url)
    with database_connection(database_url) as connection:
        connection.execute("CREATE TABLE sample (value TEXT)")
    with pytest.raises(RuntimeError, match="Abort"):
        with database_connection(database_url) as failed:
            failed.execute("INSERT INTO sample VALUES ('discard')")
            raise RuntimeError("Abort")
    with pytest.raises(sqlite3.ProgrammingError):
        failed.execute("SELECT 1")
    with database_connection(database_url) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sample").fetchone()[0] == 0


def test_connection_closes_after_success() -> None:
    with database_connection("sqlite:///:memory:") as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_commit_failure_rolls_back(database_url: str) -> None:
    initialize_database(database_url)
    with database_connection(database_url) as connection:
        connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE child (id INTEGER REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED)"
        )
    with pytest.raises(sqlite3.IntegrityError):
        with database_connection(database_url) as failed:
            failed.execute("INSERT INTO child VALUES (999)")
    with pytest.raises(sqlite3.ProgrammingError):
        failed.execute("SELECT 1")
    with database_connection(database_url) as connection:
        assert connection.execute("SELECT COUNT(*) FROM child").fetchone()[0] == 0


def test_initialization_is_idempotent_and_creates_directories(database_url: str) -> None:
    assert not Path("nested").exists()
    initialize_database(database_url)
    assert Path("nested/test.db").is_file()
    with database_connection(database_url) as connection:
        connection.execute("INSERT INTO _rallybot_metadata VALUES ('test', 'preserved')")
    initialize_database(database_url)
    with database_connection(database_url) as connection:
        assert connection.execute(
            "SELECT value FROM _rallybot_metadata WHERE key='test'"
        ).fetchone()[0] == "preserved"
        assert connection.execute(
            "SELECT COUNT(*) FROM _rallybot_metadata WHERE key='schema_version'"
        ).fetchone()[0] == 1
        assert [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type='table'"
        )] == ["_rallybot_metadata", "players", "challenges", "challenge_approval_receipts"]


def test_absolute_and_relative_paths(tmp_path: Path) -> None:
    assert database_path("sqlite:///./test.db") == tmp_path / "test.db"
    absolute_url = "sqlite:///" + (tmp_path / "absolute space.db").as_posix()
    initialize_database(absolute_url)
    assert check_database_health(absolute_url)
    assert (tmp_path / "absolute space.db").is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows drive path semantics")
def test_windows_paths(tmp_path: Path) -> None:
    assert database_path("sqlite:///" + str(tmp_path / "test.db")) == tmp_path / "test.db"
    assert database_path("sqlite:////" + (tmp_path / "test.db").as_posix()) == tmp_path / "test.db"


def test_in_memory_connections_are_private() -> None:
    url = "sqlite:///:memory:"
    assert database_path(url) is None
    initialize_database(url)
    assert check_database_health(url)
    with database_connection(url) as first:
        first.execute("CREATE TABLE sample (value TEXT)")
        with database_connection(url) as second:
            assert second.execute("SELECT COUNT(*) FROM sqlite_schema").fetchone()[0] == 0
    assert list(Path.cwd().iterdir()) == []


@pytest.mark.parametrize("url", [
    "postgresql://user:password@host/database", "sqlite://host/db", "sqlite:///",
    "sqlite:///file:other.db", "sqlite:///db?mode=memory", "sqlite:///db#fragment",
    "sqlite:///C:relative.db", "sqlite:///db%20name", "sqlite:///:memory:/bad",
])
def test_invalid_urls_are_rejected_without_leaking_input(url: str) -> None:
    with pytest.raises(InvalidDatabaseURL) as error:
        connect_database(url)
    assert url not in str(error.value)
    assert "password" not in str(error.value)
    assert not check_database_health(url)


def test_health_check_succeeds(database_url: str) -> None:
    initialize_database(database_url)
    assert check_database_health(database_url) is True


def test_missing_database_is_not_silently_created() -> None:
    url = "sqlite:///missing.db"
    assert check_database_health(url) is False
    with pytest.raises(DatabaseError):
        connect_database(url)
    assert not Path("missing.db").exists()


def test_corrupt_database_fails_safely(tmp_path: Path) -> None:
    path = tmp_path / "private.db"
    original = b"This is not a SQLite database" * 20
    path.write_bytes(original)
    url = "sqlite:///" + path.as_posix()
    assert check_database_health(url) is False
    with pytest.raises(DatabaseError) as error:
        initialize_database(url)
    assert str(path) not in str(error.value)
    assert path.read_bytes() == original


def test_directory_as_database_fails_safely(tmp_path: Path) -> None:
    assert not check_database_health("sqlite:///" + tmp_path.as_posix())


def test_connections_cannot_cross_threads() -> None:
    connection = connect_database("sqlite:///:memory:")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(sqlite3.ProgrammingError):
                executor.submit(connection.execute, "SELECT 1").result()
    finally:
        connection.close()


def run_cli(command: str, url: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ, DATABASE_URL=url, LOG_LEVEL="INFO")
    executable = Path(sys.executable).with_name("rallybot.exe")
    return subprocess.run([str(executable), command], env=environment,
                          capture_output=True, text=True, check=False)


def test_cli_init_and_check(database_url: str) -> None:
    initialized = run_cli("db-init", database_url)
    assert initialized.returncode == 0
    assert initialized.stdout == "Database initialized successfully.\n"
    assert json.loads(initialized.stderr)["message"] == "Database initialization completed"
    checked = run_cli("db-check", database_url)
    assert checked.returncode == 0
    assert checked.stdout == "Database health check passed.\n"
    assert json.loads(checked.stderr)["message"] == "Database health check passed"


@pytest.mark.parametrize("command", ["db-init", "db-check"])
def test_cli_errors_are_safe(command: str) -> None:
    result = run_cli(command, "postgresql://user:secret-password@private-host/database")
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Database command failed" in result.stderr
    for value in ("secret-password", "private-host", str(Path.cwd()), "Traceback"):
        assert value not in result.stderr


def test_cli_check_does_not_create_missing_file() -> None:
    result = run_cli("db-check", "sqlite:///missing.db")
    assert result.returncode == 1
    assert not Path("missing.db").exists()
