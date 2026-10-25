"""Current ladder retrieval as a deterministic, read-only LangGraph agent."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rallybot.agents.models import AgentInput, AgentOutput, AgentState, AgentUpdate, agent_error, invoke_agent, validate_request
from rallybot.intent.models import IntentType
from rallybot.ladder.models import PlayerRanking
from rallybot.ladder.repository import PlayerRepository
from rallybot.ladder.service import get_ladder


@dataclass(frozen=True)
class LadderAgentDependencies:
    get_rankings: Callable[[], list[PlayerRanking]]


def create_ladder_dependencies(repository: PlayerRepository | None = None) -> LadderAgentDependencies:
    return LadderAgentDependencies(get_rankings=lambda: get_ladder(repository))


def retrieve_rankings(state: AgentState, *, dependencies: LadderAgentDependencies) -> AgentUpdate:
    logging.getLogger(__name__).info("Ladder agent retrieving rankings")
    try:
        rankings = dependencies.get_rankings()
        return {"result": AgentOutput(agent_name="ladder", status="COMPLETED",
            message="Current ladder rankings." if rankings else "The ladder is empty. Register a player to get started.",
            data={"rankings": [{"rank": item.rank, "player_name": item.player_name, "points": item.points}
                               for item in rankings]})}
    except Exception as error:
        return {"result": agent_error("ladder", error)}


def build_ladder_agent(dependencies: LadderAgentDependencies | None = None) -> CompiledStateGraph:
    dependencies = dependencies if dependencies is not None else create_ladder_dependencies()
    graph = StateGraph(AgentState)
    graph.add_node("validate_request", partial(validate_request, agent_name="ladder", intent=IntentType.VIEW_LADDER))
    graph.add_node("retrieve_rankings", partial(retrieve_rankings, dependencies=dependencies))
    graph.add_edge(START, "validate_request")
    graph.add_conditional_edges("validate_request", lambda state: state["result"] is None,
                                {True: "retrieve_rankings", False: END})
    graph.add_edge("retrieve_rankings", END)
    return graph.compile()


def run_ladder_agent(request: AgentInput, *, dependencies: LadderAgentDependencies | None = None) -> AgentOutput:
    return invoke_agent(build_ladder_agent(dependencies), request, agent_name="ladder")
