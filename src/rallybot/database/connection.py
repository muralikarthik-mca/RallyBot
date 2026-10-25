"""Short-lived SQLite connections with explicit transaction boundaries."""

import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class InvalidDatabaseURL(ValueError):
    """The configured database URL is unsupported or ambiguous."""


class DatabaseError(RuntimeError):
    """A database operation failed; the message is safe for CLI display."""


def database_path(database_url: str) -> Path | None:
    """Resolve a literal SQLite path against cwd; None denotes private memory."""
    if not database_url.startswith("sqlite:///"):
        raise InvalidDatabaseURL("Expected a SQLite URL starting with sqlite:///.")
    value = database_url.removeprefix("sqlite:///")
    if value == ":memory:":
        return None
    if (not value or not value.strip() or any(char in value for char in "?#%\x00")
            or value.startswith("file:") or ":memory:" in value):
        raise InvalidDatabaseURL("Unsupported or ambiguous SQLite file URL.")
    # Accept both sqlite:///C:/... and sqlite:////C:/... on Windows.
    if re.match(r"^/[A-Za-z]:[/\\]", value):
        value = value[1:]
    if re.match(r"^[A-Za-z]:", value):
        if os.name != "nt" or not re.match(r"^[A-Za-z]:[/\\]", value):
            raise InvalidDatabaseURL("An absolute Windows SQLite path is required.")
    elif ":" in value:
        raise InvalidDatabaseURL("Unsupported SQLite file path.")
    if os.name == "nt" and value.startswith("\\") and not value.startswith("\\\\"):
        raise InvalidDatabaseURL("Drive-relative SQLite paths are unsupported.")
    return Path(value).resolve()


def connect_database(database_url: str, *, create: bool = False) -> sqlite3.Connection:
    """Open a thread-bound connection; caller must close it. Files must exist by default."""
    try:
        path = database_path(database_url)
        target = ":memory:" if path is None else path.as_uri() + ("?mode=rwc" if create else "?mode=rw")
        connection = sqlite3.connect(target, uri=True, timeout=5.0, autocommit=True)
        try:
            connection.row_factory = sqlite3.Row
            # This must precede opening a transaction: SQLite otherwise ignores it.
            connection.execute("PRAGMA foreign_keys = ON")
            if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                raise DatabaseError("SQLite foreign key enforcement is unavailable.")
            connection.execute("SELECT name FROM sqlite_schema LIMIT 1").fetchall()
            connection.autocommit = False
            return connection
        except BaseException:
            connection.close()
            raise
    except (sqlite3.Error, OSError):
        raise DatabaseError("Unable to open the configured database.") from None


@contextmanager
def database_connection(database_url: str, *, create: bool = False) -> Iterator[sqlite3.Connection]:
    """Commit on success, roll back on failure, and always close the connection."""
    connection = connect_database(database_url, create=create)
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(database_url: str) -> None:
    """Initialize metadata and the additive players schema without replacing data."""
    try:
        path = database_path(database_url)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        with database_connection(database_url, create=True) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS _rallybot_metadata "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO _rallybot_metadata (key, value) VALUES (?, ?)",
                ("schema_version", "1"),
            )
            version = connection.execute(
                "SELECT value FROM _rallybot_metadata WHERE key = ?", ("schema_version",)
            ).fetchone()[0]
            if version not in ("1", "2", "3", "4"):
                raise DatabaseError("Unsupported database schema version.")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY,
                    discord_user_id TEXT UNIQUE,
                    display_name TEXT NOT NULL CHECK (length(trim(display_name)) > 0),
                    points INTEGER NOT NULL DEFAULT 1000
                        CHECK (typeof(points) = 'integer' AND points >= 0),
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS challenges (
                    id INTEGER PRIMARY KEY,
                    challenger_player_id INTEGER NOT NULL REFERENCES players(id),
                    opponent_player_id INTEGER NOT NULL REFERENCES players(id),
                    status TEXT NOT NULL DEFAULT 'PENDING'
                        CHECK (status IN ('PENDING', 'ACCEPTED', 'DECLINED', 'CANCELLED')),
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    CHECK (challenger_player_id <> opponent_player_id)
                )
            """)
            connection.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS challenges_pending_pair
                ON challenges (
                    min(challenger_player_id, opponent_player_id),
                    max(challenger_player_id, opponent_player_id)
                ) WHERE status = 'PENDING'
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS challenge_approval_receipts (
                    workflow_id TEXT PRIMARY KEY,
                    challenge_id INTEGER NOT NULL UNIQUE REFERENCES challenges(id)
                )
            """)
            connection.execute(
                "UPDATE _rallybot_metadata SET value = ? WHERE key = ? AND value IN (?, ?, ?)",
                ("4", "schema_version", "1", "2", "3"),
            )
    except (sqlite3.Error, OSError):
        raise DatabaseError("Unable to initialize the configured database.") from None


def check_database_health(database_url: str) -> bool:
    """Check connectivity without creating a missing file; return False on failure."""
    try:
        with database_connection(database_url) as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1
    except (DatabaseError, InvalidDatabaseURL, sqlite3.Error, OSError):
        return False
