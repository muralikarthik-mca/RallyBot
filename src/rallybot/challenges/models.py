"""Typed challenge lifecycle data."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class ChallengeStatus(StrEnum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    DECLINED = "DECLINED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class Challenge:
    id: int
    challenger_player_id: int
    opponent_player_id: int
    status: ChallengeStatus
    created_at: datetime
    updated_at: datetime
