"""Parameterized player persistence using short-lived SQLite connections."""

import sqlite3
from collections.abc import Sequence
from datetime import datetime

from rallybot.database.connection import DatabaseError, database_connection
from rallybot.ladder.models import Player, PlayerRanking, SeedResult


class DuplicateDiscordUserError(ValueError):
    """A player already owns the supplied Discord identity."""


def _player(row: sqlite3.Row) -> Player:
    return Player(
        id=row["id"], discord_user_id=row["discord_user_id"],
        display_name=row["display_name"], points=row["points"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _validate(display_name: str, points: int, discord_user_id: str | None = None) -> None:
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError("Player display name must not be empty.")
    if type(points) is not int or not 0 <= points <= 9223372036854775807:
        raise ValueError("Player points must be a nonnegative SQLite integer.")
    if discord_user_id is not None and (not isinstance(discord_user_id, str) or not discord_user_id.strip()):
        raise ValueError("Discord user ID must be a nonempty string when provided.")


class PlayerRepository:
    """Persist players without retaining connections or exposing SQLite rows."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def create_player(
        self, display_name: str, points: int = 1000, discord_user_id: str | None = None
    ) -> Player:
        _validate(display_name, points, discord_user_id)
        try:
            with database_connection(self.database_url) as connection:
                row = connection.execute(
                    "INSERT INTO players (discord_user_id, display_name, points) VALUES (?, ?, ?) RETURNING *",
                    (discord_user_id, display_name.strip(), points),
                ).fetchone()
                return _player(row)
        except sqlite3.IntegrityError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE:
                raise DuplicateDiscordUserError("A player with that Discord user ID already exists.") from None
            raise DatabaseError("Unable to create player.") from None
        except sqlite3.Error:
            raise DatabaseError("Unable to create player.") from None

    def get_player(self, player_id: int) -> Player | None:
        try:
            with database_connection(self.database_url) as connection:
                row = connection.execute("SELECT * FROM players WHERE id = ?", (player_id,)).fetchone()
                return _player(row) if row is not None else None
        except sqlite3.Error:
            raise DatabaseError("Unable to retrieve player.") from None

    def list_players(self) -> list[Player]:
        try:
            with database_connection(self.database_url) as connection:
                return [_player(row) for row in connection.execute("SELECT * FROM players ORDER BY id ASC")]
        except sqlite3.Error:
            raise DatabaseError("Unable to list players.") from None

    def get_player_by_discord_user_id(self, discord_user_id: str) -> Player | None:
        """Look up only the player associated with the supplied identity."""
        try:
            with database_connection(self.database_url) as connection:
                row = connection.execute(
                    "SELECT * FROM players WHERE discord_user_id = ?", (discord_user_id,)
                ).fetchone()
                return _player(row) if row is not None else None
        except sqlite3.Error:
            raise DatabaseError("Unable to retrieve registered player.") from None

    def list_rankings(self) -> list[PlayerRanking]:
        return [PlayerRanking(rank, player.display_name, player.points)
                for rank, player in enumerate(self.list_ranked_players(), start=1)]

    def list_ranked_players(self) -> list[Player]:
        """Return identities in the same ordering used by the tennis ladder."""
        try:
            with database_connection(self.database_url) as connection:
                return [_player(row) for row in connection.execute("SELECT * FROM players ORDER BY points DESC, id ASC")]
        except sqlite3.Error:
            raise DatabaseError("Unable to retrieve ladder rankings.") from None

    def seed_players(self, samples: Sequence[PlayerRanking]) -> SeedResult:
        """Atomically add samples whose display names do not already exist."""
        for sample in samples:
            _validate(sample.player_name, sample.points)
        added = 0
        try:
            with database_connection(self.database_url) as connection:
                # Claim the write lock before checking names so concurrent seeds
                # cannot both observe the same sample as absent.
                connection.execute("UPDATE players SET points = points WHERE 0")
                for sample in samples:
                    if connection.execute(
                        "SELECT 1 FROM players WHERE display_name = ?", (sample.player_name,)
                    ).fetchone() is None:
                        connection.execute(
                            "INSERT INTO players (display_name, points) VALUES (?, ?)",
                            (sample.player_name, sample.points),
                        )
                        added += 1
            return SeedResult(added, len(samples) - added)
        except sqlite3.Error:
            raise DatabaseError("Unable to seed development players.") from None
