"""Allowlisted LangChain tools bound to a trusted requester outside model schemas."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rallybot.config import get_settings
from rallybot.ladder.repository import PlayerRepository
from rallybot.ladder.service import get_ladder
from rallybot.tools.player_tools import PlayerSummary, UnknownPlayerError, create_player_tools
from rallybot.workflows.matchmaking_graph import run_matchmaking

ToolName = Literal["get_player_profile", "list_eligible_opponents", "get_ladder_rankings"]
TOOL_NAMES = ("get_player_profile", "list_eligible_opponents", "get_ladder_rankings")


class ProfileInput(BaseModel):
    """No model-selected identity or other parameters are permitted."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class RankingInput(ProfileInput):
    limit: int = Field(default=5, ge=1, le=20, description="Maximum records to return, from 1 to 20.")


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    tool: ToolName
    status: Literal["OK", "ERROR"]
    data: dict[str, object] = Field(default_factory=dict)
    error_code: str | None = None


class RankingEntry(ProfileInput):
    player_name: str = Field(max_length=100)
    points: int = Field(ge=0)
    rank: int = Field(ge=1)


class OpponentData(ProfileInput):
    opponents: list[RankingEntry] = Field(max_length=20)
    max_point_difference: Literal[200]


class LadderData(ProfileInput):
    rankings: list[RankingEntry] = Field(max_length=20)


@dataclass(frozen=True)
class ReadOnlyTools:
    tools: tuple[StructuredTool, ...]
    validate_requester: Callable[[], PlayerSummary]
    requester_player_id: int


def create_agent_tools(requester_player_id: int, repository: PlayerRepository | None = None) -> ReadOnlyTools:
    """Caller supplies an authenticated ID; the LLM cannot supply or override it."""
    repository = repository if repository is not None else PlayerRepository(get_settings().database_url)
    players = create_player_tools(repository)

    def safe_read(name: ToolName, read: Callable[[], dict[str, object]]) -> dict[str, object]:
        try:
            players.get_player(requester_player_id)
            return ToolResult(tool=name, status="OK", data=read()).model_dump()
        except Exception as error:
            return ToolResult(tool=name, status="ERROR", error_code=(
                "REQUESTER_NOT_FOUND" if isinstance(error, UnknownPlayerError) else "TOOL_UNAVAILABLE"
            )).model_dump()

    def get_player_profile() -> dict[str, object]:
        def read() -> dict[str, object]:
            player = players.get_player(requester_player_id)
            rank = next(rank for rank, item in enumerate(repository.list_ranked_players(), 1)
                        if item.id == player.id)
            return {"player_name": player.name[:100], "points": player.points, "rank": rank}
        return safe_read("get_player_profile", read)

    def list_eligible_opponents(limit: int = 5) -> dict[str, object]:
        def read() -> dict[str, object]:
            result = run_matchmaking(requester_player_id, tools=players)
            return {"opponents": [{"player_name": item["name"][:100], "points": item["points"], "rank": item["rank"]}
                                  for item in result["available_players"][:limit]], "max_point_difference": 200}
        return safe_read("list_eligible_opponents", read)

    def get_ladder_rankings(limit: int = 5) -> dict[str, object]:
        return safe_read("get_ladder_rankings", lambda: {"rankings": [
            {"rank": item.rank, "player_name": item.player_name[:100], "points": item.points}
            for item in get_ladder(repository)[:limit]
        ]})

    tools = (
        StructuredTool.from_function(get_player_profile, args_schema=ProfileInput,
            description="Read the authenticated requester's player name, points and ladder rank. Accepts no identity arguments."),
        StructuredTool.from_function(list_eligible_opponents, args_schema=RankingInput,
            description="Read opponents within 200 points of the authenticated requester, excluding self; ordered by existing matchmaking rules. Optional limit 1-20."),
        StructuredTool.from_function(get_ladder_rankings, args_schema=RankingInput,
            description="Read the current ladder ordered by points and existing ranking rules. Optional limit 1-20, default 5."),
    )
    return ReadOnlyTools(tools, lambda: players.get_player(requester_player_id), requester_player_id)


def validate_tool_arguments(tools: ReadOnlyTools, name: str, arguments: object) -> dict[str, object]:
    """Validate once against the allowlist schema and include declared defaults."""
    tool = next((item for item in tools.tools if item.name == name and name in TOOL_NAMES), None)
    if tool is None:
        raise ValueError("UNKNOWN_TOOL")
    try:
        if type(arguments) is not dict:
            raise ValueError
        validated = tool.args_schema.model_validate(arguments)
    except (ValidationError, ValueError, TypeError):
        raise ValueError("INVALID_TOOL_ARGUMENTS") from None
    return validated.model_dump()


def execute_tool(tools: ReadOnlyTools, name: str, arguments: object) -> ToolResult:
    """All model calls pass through strict argument validation and this allowlist."""
    normalized = validate_tool_arguments(tools, name, arguments)
    tool = next(item for item in tools.tools if item.name == name)
    try:
        result = ToolResult.model_validate(tool.invoke(normalized))
        if result.tool != name:
            raise ValueError
        if result.status == "ERROR":
            return ToolResult(tool=name, status="ERROR", error_code="TOOL_UNAVAILABLE")
        schema = {"get_player_profile": RankingEntry, "list_eligible_opponents": OpponentData,
                  "get_ladder_rankings": LadderData}[name]
        return ToolResult(tool=name, status="OK", data=schema.model_validate(result.data).model_dump())
    except Exception:
        return ToolResult(tool=name, status="ERROR", error_code="TOOL_UNAVAILABLE")
