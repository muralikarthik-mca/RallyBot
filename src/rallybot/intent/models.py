"""Validated intent values and literal extracted request fields."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IntentType(StrEnum):
    FIND_OPPONENT = "FIND_OPPONENT"
    VIEW_LADDER = "VIEW_LADDER"
    CHALLENGE_PLAYER = "CHALLENGE_PLAYER"
    VIEW_CHALLENGES = "VIEW_CHALLENGES"
    COURT_REQUEST = "COURT_REQUEST"
    UNKNOWN = "UNKNOWN"


class ClassifiedIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)

    intent: IntentType
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    player_name: str | None = Field(default=None, max_length=100)
    opponent_name: str | None = Field(default=None, max_length=100,
        description="Literal opponent name explicitly targeted in the request; no database resolution required.")
    requested_date: str | None = Field(default=None, max_length=100)
    requested_time: str | None = Field(default=None, max_length=100)
    clarification_needed: bool

    @field_validator("player_name", "opponent_name", "requested_date", "requested_time")
    @classmethod
    def clean_literal_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("Extracted fields must be plain single-line text.")
        return value or None
