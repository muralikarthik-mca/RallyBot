"""Typed tennis ladder data."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Player:
    """A persisted player; Discord identity is optional."""

    id: int
    discord_user_id: str | None
    display_name: str
    points: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class SeedResult:
    added: int
    skipped: int


@dataclass(frozen=True)
class PlayerRanking:
    """A player's position and points in a ladder."""

    rank: int
    player_name: str
    points: int
