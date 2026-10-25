"""Resolve and validate a challenge preview; never start/resume an approval."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rallybot.agents.models import AgentInput, AgentOutput, AgentState, AgentUpdate, agent_error, invoke_agent, validate_request
from rallybot.challenges.repository import ChallengeRepository
from rallybot.challenges.service import ChallengeService, ChallengeValidationError
from rallybot.config import Settings, get_settings
from rallybot.intent.models import IntentType
from rallybot.ladder.repository import PlayerRepository
from rallybot.ladder.service import PlayerResolutionError, resolve_player
from rallybot.tools.player_tools import PlayerSummary, create_player_tools


@dataclass(frozen=True)
class ChallengeAgentDependencies:
    get_player: Callable[[int], PlayerSummary]
    resolve_opponent: Callable[[str], PlayerSummary]
    prepare_challenge: Callable[[int, int], None]


def create_challenge_dependencies(settings: Settings | None = None) -> ChallengeAgentDependencies:
    settings = settings if settings is not None else get_settings()
    players = PlayerRepository(settings.database_url)
    service = ChallengeService(players=players, challenges=ChallengeRepository(settings.database_url))

    def resolve(name: str) -> PlayerSummary:
        player = resolve_player(players, name)
        return PlayerSummary(player.id, player.display_name, player.points)

    def prepare(requester_id: int, opponent_id: int) -> None:
        service.prepare_challenge(requester_id, opponent_id)

    return ChallengeAgentDependencies(create_player_tools(players).get_player, resolve, prepare)


def prepare_preview(state: AgentState, *, dependencies: ChallengeAgentDependencies) -> AgentUpdate:
    logging.getLogger(__name__).info("Challenge agent preparing read-only preview")
    request = state["request"]
    if request.requester_player_id is None or request.opponent_name is None:
        return {"result": AgentOutput(agent_name="challenge", status="CLARIFICATION_NEEDED",
            message="Provide a trusted registered requester and an explicit opponent name.",
            error_code="MISSING_REQUESTER" if request.requester_player_id is None else "MISSING_OPPONENT")}
    try:
        dependencies.get_player(request.requester_player_id)
        opponent = dependencies.resolve_opponent(request.opponent_name)
        if opponent.id == request.requester_player_id:
            return {"result": AgentOutput(agent_name="challenge", status="ERROR",
                message="You cannot challenge yourself.", error_code="SELF_CHALLENGE")}
        dependencies.prepare_challenge(request.requester_player_id, opponent.id)
        return {"result": AgentOutput(agent_name="challenge", status="COMPLETED",
            message="Preview only: challenge creation requires human confirmation. No challenge was created.",
            data={"requester_player_id": request.requester_player_id, "opponent_player_id": opponent.id,
                  "opponent_name": opponent.name, "opponent_resolved": True,
                  "requires_confirmation": True, "challenge_created": False})}
    except PlayerResolutionError as error:
        code = "AMBIGUOUS_OPPONENT" if str(error) == "AMBIGUOUS_OPPONENT" else "UNKNOWN_OPPONENT"
        logging.getLogger(__name__).info("Challenge agent needs clarification: %s", code)
        return {"result": AgentOutput(agent_name="challenge", status="CLARIFICATION_NEEDED", error_code=code,
            message="More than one registered player matches. Please clarify the opponent."
            if code == "AMBIGUOUS_OPPONENT" else "No registered opponent matches that name. Please clarify the opponent.")}
    except ChallengeValidationError:
        logging.getLogger(__name__).info("Challenge agent rejected preview: CHALLENGE_INELIGIBLE")
        return {"result": AgentOutput(agent_name="challenge", status="ERROR", error_code="CHALLENGE_INELIGIBLE",
            message="This challenge is not eligible. Check player registration and existing pending challenges.")}
    except Exception as error:
        return {"result": agent_error("challenge", error)}


def build_challenge_agent(dependencies: ChallengeAgentDependencies | None = None) -> CompiledStateGraph:
    dependencies = dependencies if dependencies is not None else create_challenge_dependencies()
    graph = StateGraph(AgentState)
    graph.add_node("validate_request", partial(validate_request, agent_name="challenge", intent=IntentType.CHALLENGE_PLAYER))
    graph.add_node("prepare_preview", partial(prepare_preview, dependencies=dependencies))
    graph.add_edge(START, "validate_request")
    graph.add_conditional_edges("validate_request", lambda state: state["result"] is None,
                                {True: "prepare_preview", False: END})
    graph.add_edge("prepare_preview", END)
    return graph.compile()


def run_challenge_agent(request: AgentInput, *, dependencies: ChallengeAgentDependencies | None = None) -> AgentOutput:
    return invoke_agent(build_challenge_agent(dependencies), request, agent_name="challenge")
