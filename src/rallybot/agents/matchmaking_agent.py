"""Read-only delegation to the existing deterministic matchmaking workflow."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rallybot.agents.models import AgentInput, AgentOutput, AgentState, AgentUpdate, agent_error, invoke_agent, validate_request
from rallybot.config import Settings, get_settings
from rallybot.intent.models import IntentType
from rallybot.ladder.repository import PlayerRepository
from rallybot.llm.matchmaking_explainer import MatchmakingExplainer
from rallybot.tools.player_tools import PlayerSummary, create_player_tools
from rallybot.workflows.matchmaking_graph import MatchmakingState, run_matchmaking


@dataclass(frozen=True)
class MatchmakingAgentDependencies:
    get_player: Callable[[int], PlayerSummary]
    matchmaking: Callable[[int], MatchmakingState]
    explanation_enabled: bool = False


def create_matchmaking_dependencies(
    settings: Settings | None = None, *, enable_ai_explanation: bool = False,
    explainer: MatchmakingExplainer | None = None,
) -> MatchmakingAgentDependencies:
    settings = settings if settings is not None else get_settings()
    tools = create_player_tools(PlayerRepository(settings.database_url))
    active_explainer = (explainer if explainer is not None else MatchmakingExplainer(settings)) if enable_ai_explanation else None
    return MatchmakingAgentDependencies(tools.get_player,
        lambda player_id: run_matchmaking(player_id, tools=tools, explainer=active_explainer), enable_ai_explanation)


def recommend_opponent(state: AgentState, *, dependencies: MatchmakingAgentDependencies) -> AgentUpdate:
    logging.getLogger(__name__).info("Matchmaking agent executing deterministic workflow")
    player_id = state["request"].requester_player_id
    if player_id is None:
        return {"result": AgentOutput(agent_name="matchmaking", status="CLARIFICATION_NEEDED",
            message="A registered requester from trusted application context is required.", error_code="MISSING_REQUESTER")}
    try:
        dependencies.get_player(player_id)
        result = dependencies.matchmaking(player_id)
        data: dict[str, object] = {
            "requester_player_id": player_id, "eligible_opponents": result["available_players"],
            "recommended_opponent": result["recommended_opponent"],
            "recommended_opponent_id": result["recommended_opponent_id"], "matchmaking_status": result["status"],
        }
        if dependencies.explanation_enabled:
            data.update(explanation=result.get("explanation"), explanation_source=result.get("explanation_source"))
        return {"result": AgentOutput(agent_name="matchmaking", status="COMPLETED", data=data,
            message="Read-only matchmaking completed." if result["recommended_opponent_id"] is not None
            else "No eligible opponents were found within 200 points.")}
    except Exception as error:
        return {"result": agent_error("matchmaking", error)}


def build_matchmaking_agent(dependencies: MatchmakingAgentDependencies | None = None) -> CompiledStateGraph:
    dependencies = dependencies if dependencies is not None else create_matchmaking_dependencies()
    graph = StateGraph(AgentState)
    graph.add_node("validate_request", partial(validate_request, agent_name="matchmaking", intent=IntentType.FIND_OPPONENT))
    graph.add_node("recommend_opponent", partial(recommend_opponent, dependencies=dependencies))
    graph.add_edge(START, "validate_request")
    graph.add_conditional_edges("validate_request", lambda state: state["result"] is None,
                                {True: "recommend_opponent", False: END})
    graph.add_edge("recommend_opponent", END)
    return graph.compile()


def run_matchmaking_agent(request: AgentInput, *, dependencies: MatchmakingAgentDependencies | None = None) -> AgentOutput:
    return invoke_agent(build_matchmaking_agent(dependencies), request, agent_name="matchmaking")
