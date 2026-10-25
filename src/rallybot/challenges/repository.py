"""SQLite challenge operations with database-enforced race protection."""

import sqlite3
from datetime import datetime

from rallybot.challenges.models import Challenge, ChallengeStatus
from rallybot.database.connection import DatabaseError, database_connection


class DuplicatePendingChallengeError(ValueError):
    """An unordered player pair already has a pending challenge."""


def _challenge(row: sqlite3.Row) -> Challenge:
    return Challenge(
        id=row["id"], challenger_player_id=row["challenger_player_id"],
        opponent_player_id=row["opponent_player_id"], status=ChallengeStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


class ChallengeRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def create_approved_challenge(self, challenger_id: int, opponent_id: int, workflow_id: str) -> Challenge:
        """Atomically commit the action and its durable replay receipt."""
        try:
            with database_connection(self.database_url) as connection:
                # Acquire a write reservation before reading the receipt.
                connection.execute("UPDATE challenge_approval_receipts SET challenge_id = challenge_id WHERE 0")
                row = connection.execute(
                    "SELECT c.* FROM challenges c JOIN challenge_approval_receipts r "
                    "ON r.challenge_id = c.id WHERE r.workflow_id = ?", (workflow_id,),
                ).fetchone()
                if row is not None:
                    if row["challenger_player_id"] != challenger_id or row["opponent_player_id"] != opponent_id:
                        raise ValueError("Approval does not match the original request.")
                    return _challenge(row)
                row = connection.execute(
                    "INSERT INTO challenges (challenger_player_id, opponent_player_id) VALUES (?, ?) RETURNING *",
                    (challenger_id, opponent_id),
                ).fetchone()
                connection.execute(
                    "INSERT INTO challenge_approval_receipts (workflow_id, challenge_id) VALUES (?, ?)",
                    (workflow_id, row["id"]),
                )
                return _challenge(row)
        except sqlite3.IntegrityError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE:
                raise DuplicatePendingChallengeError("A pending challenge already exists between these players.") from None
            raise DatabaseError("Unable to create approved challenge.") from None
        except sqlite3.Error:
            raise DatabaseError("Unable to create approved challenge.") from None

    def create_challenge(self, challenger_player_id: int, opponent_player_id: int) -> Challenge:
        try:
            with database_connection(self.database_url) as connection:
                row = connection.execute(
                    "INSERT INTO challenges (challenger_player_id, opponent_player_id) VALUES (?, ?) RETURNING *",
                    (challenger_player_id, opponent_player_id),
                ).fetchone()
                return _challenge(row)
        except sqlite3.IntegrityError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE:
                raise DuplicatePendingChallengeError("A pending challenge already exists between these players.") from None
            raise DatabaseError("Unable to create challenge.") from None
        except sqlite3.Error:
            raise DatabaseError("Unable to create challenge.") from None

    def get_challenge(self, challenge_id: int) -> Challenge | None:
        try:
            with database_connection(self.database_url) as connection:
                row = connection.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
                return _challenge(row) if row is not None else None
        except sqlite3.Error:
            raise DatabaseError("Unable to retrieve challenge.") from None

    def list_challenges(self, player_id: int) -> list[Challenge]:
        try:
            with database_connection(self.database_url) as connection:
                return [_challenge(row) for row in connection.execute(
                    "SELECT * FROM challenges WHERE challenger_player_id = ? OR opponent_player_id = ? ORDER BY id ASC",
                    (player_id, player_id),
                )]
        except sqlite3.Error:
            raise DatabaseError("Unable to list challenges.") from None

    def transition_challenge(
        self, challenge_id: int, actor_player_id: int, status: ChallengeStatus
    ) -> Challenge | None:
        """Change only PENDING rows belonging to the actor's permitted role."""
        if status not in (ChallengeStatus.ACCEPTED, ChallengeStatus.DECLINED, ChallengeStatus.CANCELLED):
            raise ValueError("Unsupported challenge transition.")
        try:
            with database_connection(self.database_url) as connection:
                row = connection.execute("""
                    UPDATE challenges
                    SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE id = ? AND status = 'PENDING'
                      AND ((? IN ('ACCEPTED', 'DECLINED') AND opponent_player_id = ?)
                           OR (? = 'CANCELLED' AND challenger_player_id = ?))
                    RETURNING *
                """, (status.value, challenge_id, status.value, actor_player_id, status.value, actor_player_id)).fetchone()
                return _challenge(row) if row is not None else None
        except sqlite3.Error:
            raise DatabaseError("Unable to update challenge.") from None
