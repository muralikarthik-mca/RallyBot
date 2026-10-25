"""Validated contracts shared by the specialized agent graphs."""

import logging
from typing import Literal, TypedDict
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rallybot.intent.models import IntentType
from rallybot.tools.player_tools import UnknownPlayerError

AgentName = Literal["ladder", "matchmaking", "challenge"]
AgentStatus = Literal["COMPLETED", "CLARIFICATION_NEEDED", "ERROR"]


class RequestContext(BaseModel):
    """Allowlisted correlation metadata; no free-form user payload or credentials."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)
    request_id: UUID | None = None


class AgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)

    requester_player_id: int | None = Field(default=None, gt=0, le=9223372036854775807)
    intent: IntentType
    opponent_name: str | None = Field(default=None, max_length=100)
    request_context: RequestContext = Field(default_factory=RequestContext)

    @field_validator("opponent_name")
    @classmethod
    def clean_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("Opponent name must be single-line text.")
        return value.strip() or None


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)

    agent_name: AgentName
    status: AgentStatus
    message: str
    data: dict[str, object] = Field(default_factory=dict)
    error_code: str | None = None


class AgentState(TypedDict):
    request: AgentInput
    result: AgentOutput | None


class AgentUpdate(TypedDict, total=False):
    request: AgentInput
    result: AgentOutput | None


def agent_error(agent_name: AgentName, error: Exception | None = None) -> AgentOutput:
    code = "REQUESTER_NOT_FOUND" if isinstance(error, UnknownPlayerError) else "CAPABILITY_UNAVAILABLE"
    logging.getLogger(__name__).warning("Specialized agent failed: %s", code)
    return AgentOutput(agent_name=agent_name, status="ERROR", error_code=code,
        message="Requester player is not registered." if code == "REQUESTER_NOT_FOUND"
        else "This capability is temporarily unavailable.")


def validate_request(state: AgentState, *, agent_name: AgentName, intent: IntentType) -> AgentUpdate:
    try:
        request = state["request"]
        if not isinstance(request, AgentInput):
            raise ValueError
        request = AgentInput.model_validate(request.model_dump(warnings=False))
        if request.intent != intent:
            raise ValueError
        return {"request": request, "result": None}
    except Exception:
        logging.getLogger(__name__).warning("Specialized agent input rejected: INVALID_AGENT_INPUT")
        return {"result": AgentOutput(agent_name=agent_name, status="ERROR",
            message="A valid structured request for this agent is required.", error_code="INVALID_AGENT_INPUT")}


def invoke_agent(graph: CompiledStateGraph, request: AgentInput, *, agent_name: AgentName) -> AgentOutput:
    """No checkpoints/traces; only a validated result crosses the agent boundary."""
    logger = logging.getLogger(__name__)
    logger.info("Specialized agent started: %s", agent_name)
    try:
        with tracing_context(enabled=False):
            state = graph.invoke({"request": request, "result": None})
        result = state["result"]
        if not isinstance(result, AgentOutput):
            raise ValueError
        result = AgentOutput.model_validate(result.model_dump(warnings=False))
        if result.agent_name != agent_name:
            raise ValueError
    except Exception as error:
        result = agent_error(agent_name, error)
    logger.info("Specialized agent completed: %s (%s)", agent_name, result.status)
    return result
