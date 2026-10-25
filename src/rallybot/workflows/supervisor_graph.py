"""Validated intent routing to fixed, read-only application capabilities."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import NotRequired, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import tracing_context

from rallybot.agents.challenge_agent import ChallengeAgentDependencies, create_challenge_dependencies, run_challenge_agent
from rallybot.agents.ladder_agent import LadderAgentDependencies, run_ladder_agent
from rallybot.agents.matchmaking_agent import MatchmakingAgentDependencies, run_matchmaking_agent
from rallybot.agents.models import AgentInput, AgentOutput
from rallybot.challenges.models import Challenge
from rallybot.challenges.repository import ChallengeRepository
from rallybot.challenges.service import ChallengeService
from rallybot.config import Settings, get_settings
from rallybot.intent.classifier import IntentClassifier
from rallybot.intent.models import ClassifiedIntent, IntentType
from rallybot.ladder.models import PlayerRanking
from rallybot.ladder.repository import PlayerRepository
from rallybot.ladder.service import get_ladder
from rallybot.tools.player_tools import PlayerSummary, UnknownPlayerError, create_player_tools
from rallybot.workflows.matchmaking_graph import MatchmakingState, run_matchmaking


class IntentProvider(Protocol):
    def classify(self, request: str, *, authenticated_identity_expected: bool = True) -> ClassifiedIntent: ...


class SupervisorResult(TypedDict):
    capability: str
    message: str
    data: dict[str, object]
    agent_name: NotRequired[str]


class SupervisorState(TypedDict):
    user_message: str
    requester_player_id: int | None
    classified_intent: ClassifiedIntent | None
    route: str | None
    result: SupervisorResult | None
    status: str
    error_code: str | None


class SupervisorUpdate(TypedDict, total=False):
    classified_intent: ClassifiedIntent | None
    route: str | None
    result: SupervisorResult | None
    status: str
    error_code: str | None


@dataclass(frozen=True)
class SupervisorDependencies:
    classifier: IntentProvider
    get_player: Callable[[int], PlayerSummary]
    matchmaking: Callable[[int], MatchmakingState]
    ladder: Callable[[], list[PlayerRanking]]
    challenges: Callable[[int], list[Challenge]]
    challenge_agent: ChallengeAgentDependencies | None = None


def create_supervisor_dependencies(settings: Settings | None = None) -> SupervisorDependencies:
    settings = settings if settings is not None else get_settings()
    players = PlayerRepository(settings.database_url)
    tools = create_player_tools(players)
    challenges = ChallengeService(players=players, challenges=ChallengeRepository(settings.database_url))
    return SupervisorDependencies(
        classifier=IntentClassifier(settings), get_player=tools.get_player,
        matchmaking=lambda player_id: run_matchmaking(player_id, tools=tools),
        ladder=lambda: get_ladder(players), challenges=challenges.list_challenges_for_player,
        challenge_agent=create_challenge_dependencies(settings),
    )


def _valid_id(value: object) -> bool:
    return type(value) is int and 1 <= value <= 9223372036854775807


def classify_intent(state: SupervisorState, *, dependencies: SupervisorDependencies) -> SupervisorUpdate:
    logger = logging.getLogger(__name__)
    try:
        candidate = dependencies.classifier.classify(
            state["user_message"], authenticated_identity_expected=_valid_id(state["requester_player_id"]),
        )
        if not isinstance(candidate, ClassifiedIntent):
            raise ValueError("Classifier must return a structured intent")
        # Revalidate even a model_construct-created object; no string-only routing.
        intent = ClassifiedIntent.model_validate(candidate.model_dump(warnings=False))
    except Exception:
        logger.warning("Supervisor classification failed: CLASSIFICATION_FAILED")
        return {"classified_intent": None, "result": None, "route": None,
                "status": "CLASSIFICATION_FAILED", "error_code": "CLASSIFICATION_FAILED"}
    logger.info("Supervisor classified intent: %s", intent.intent.value)
    return {"classified_intent": intent, "result": None, "route": None,
            "status": "CLASSIFIED", "error_code": None}


def route_request(state: SupervisorState) -> SupervisorUpdate:
    intent = state["classified_intent"]
    error = state["error_code"]
    if intent is None or intent.intent == IntentType.UNKNOWN:
        route = "unsupported"
    elif intent.clarification_needed:
        route = "clarification"
        if intent.intent in (IntentType.FIND_OPPONENT, IntentType.VIEW_CHALLENGES, IntentType.CHALLENGE_PLAYER) and not _valid_id(state["requester_player_id"]):
            error = "MISSING_REQUESTER"
    elif intent.intent in (IntentType.FIND_OPPONENT, IntentType.VIEW_CHALLENGES, IntentType.CHALLENGE_PLAYER) and not _valid_id(state["requester_player_id"]):
        route, error = "clarification", "MISSING_REQUESTER"
    elif intent.intent == IntentType.CHALLENGE_PLAYER and intent.opponent_name is None:
        route, error = "clarification", "MISSING_OPPONENT"
    else:
        route = {
            IntentType.FIND_OPPONENT: "matchmaking", IntentType.VIEW_LADDER: "ladder",
            IntentType.VIEW_CHALLENGES: "challenges", IntentType.CHALLENGE_PLAYER: "challenge_preview",
            IntentType.COURT_REQUEST: "court",
        }[intent.intent]
    logging.getLogger(__name__).info("Supervisor selected route: %s", route)
    return {"route": route, "status": "ROUTED", "error_code": error}


def _result(capability: str, message: str, data: dict[str, object] | None = None,
            *, status: str = "COMPLETED", error_code: str | None = None) -> SupervisorUpdate:
    return {"result": {"capability": capability, "message": message, "data": data or {}},
            "status": status, "error_code": error_code}


def _handler_error(capability: str, error: Exception) -> SupervisorUpdate:
    code = "REQUESTER_NOT_FOUND" if isinstance(error, UnknownPlayerError) else "CAPABILITY_UNAVAILABLE"
    logging.getLogger(__name__).warning("Supervisor handler failed: %s", code)
    message = "Requester player is not registered." if code == "REQUESTER_NOT_FOUND" else "This capability is temporarily unavailable."
    return _result(capability, message, status="ERROR", error_code=code)


def matchmaking_handler(state: SupervisorState, *, dependencies: SupervisorDependencies) -> SupervisorUpdate:
    logging.getLogger(__name__).info("Supervisor handler executing: matchmaking")
    result = run_matchmaking_agent(_agent_request(state), dependencies=MatchmakingAgentDependencies(
        dependencies.get_player, dependencies.matchmaking))
    return _agent_result("matchmaking", result)


def ladder_handler(state: SupervisorState, *, dependencies: SupervisorDependencies) -> SupervisorUpdate:
    logging.getLogger(__name__).info("Supervisor handler executing: ladder")
    result = run_ladder_agent(_agent_request(state), dependencies=LadderAgentDependencies(dependencies.ladder))
    return _agent_result("ladder", result)


def _agent_request(state: SupervisorState) -> AgentInput:
    intent = state["classified_intent"]
    return AgentInput(requester_player_id=state["requester_player_id"], intent=intent.intent,
                      opponent_name=intent.opponent_name)


def _agent_result(capability: str, result: AgentOutput) -> SupervisorUpdate:
    return {"result": {"capability": capability, "agent_name": result.agent_name,
                       "message": result.message, "data": result.data},
            "status": result.status, "error_code": result.error_code}


def challenge_handler(state: SupervisorState, *, dependencies: SupervisorDependencies) -> SupervisorUpdate:
    logging.getLogger(__name__).info("Supervisor handler executing: challenges")
    intent = state["classified_intent"]
    if intent.intent == IntentType.CHALLENGE_PLAYER:
        if dependencies.challenge_agent is None:
            return _handler_error("challenge_preview", RuntimeError())
        result = run_challenge_agent(_agent_request(state), dependencies=dependencies.challenge_agent)
        return _agent_result("challenge_preview", result)
    try:
        player_id = state["requester_player_id"]
        if not _valid_id(player_id):
            return _result("challenges", "A trusted requester player ID is required.", status="CLARIFICATION_NEEDED", error_code="MISSING_REQUESTER")
        dependencies.get_player(player_id)
        challenges = dependencies.challenges(player_id)
        return _result("challenges", "Your challenges (read-only).", {"challenges": [
            {"id": item.id, "status": item.status.value,
             "role": "challenger" if item.challenger_player_id == player_id else "opponent",
             "created_at": item.created_at.isoformat(), "updated_at": item.updated_at.isoformat()}
            for item in challenges
        ]})
    except Exception as error:
        return _handler_error("challenges", error)


def court_handler(state: SupervisorState) -> SupervisorUpdate:
    logging.getLogger(__name__).info("Supervisor handler executing: court")
    return _result("court", "Court availability and booking are not implemented.", status="NOT_IMPLEMENTED")


def clarification_handler(state: SupervisorState) -> SupervisorUpdate:
    logging.getLogger(__name__).info("Supervisor handler executing: clarification")
    message = "Please clarify the missing or ambiguous request details."
    if state["error_code"] == "MISSING_REQUESTER":
        message = "Provide a registered requester through trusted application context; a name in the request is not an identity."
    elif state["error_code"] == "MISSING_OPPONENT":
        message = "Please explicitly name the opponent for the preview."
    return _result("clarification", message, status="CLARIFICATION_NEEDED", error_code=state["error_code"])


def unsupported_handler(state: SupervisorState) -> SupervisorUpdate:
    logging.getLogger(__name__).info("Supervisor handler executing: unsupported")
    return _result("unsupported", "This request is unsupported or could not be classified safely.",
                   status="UNSUPPORTED", error_code=state["error_code"])


def build_supervisor_graph(dependencies: SupervisorDependencies | None = None) -> CompiledStateGraph:
    dependencies = dependencies if dependencies is not None else create_supervisor_dependencies()
    graph = StateGraph(SupervisorState)
    graph.add_node("classify_intent", partial(classify_intent, dependencies=dependencies))
    graph.add_node("route_request", route_request)
    for name, handler in (("matchmaking_handler", matchmaking_handler), ("ladder_handler", ladder_handler),
                          ("challenge_handler", challenge_handler)):
        graph.add_node(name, partial(handler, dependencies=dependencies))
    for name, handler in (("court_handler", court_handler), ("clarification_handler", clarification_handler),
                          ("unsupported_handler", unsupported_handler)):
        graph.add_node(name, handler)
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "route_request")
    graph.add_conditional_edges("route_request", lambda state: state["route"], {
        "matchmaking": "matchmaking_handler", "ladder": "ladder_handler",
        "challenges": "challenge_handler", "challenge_preview": "challenge_handler",
        "court": "court_handler", "clarification": "clarification_handler", "unsupported": "unsupported_handler",
    })
    for name in ("matchmaking_handler", "ladder_handler", "challenge_handler", "court_handler",
                 "clarification_handler", "unsupported_handler"):
        graph.add_edge(name, END)
    return graph.compile()


def run_supervisor(user_message: str, requester_player_id: int | None = None, *,
                   dependencies: SupervisorDependencies | None = None) -> SupervisorState:
    logger = logging.getLogger(__name__)
    logger.info("Supervisor workflow started")
    state: SupervisorState = {"user_message": user_message, "requester_player_id": requester_player_id,
        "classified_intent": None, "route": None, "result": None, "status": "STARTED", "error_code": None}
    try:
        with tracing_context(enabled=False):
            result = build_supervisor_graph(dependencies).invoke(state)
    except Exception:
        logger.warning("Supervisor workflow failed: SUPERVISOR_FAILED")
        state.update(_result("supervisor", "The supervisor is temporarily unavailable.", status="ERROR", error_code="SUPERVISOR_FAILED"))
        result = state
    logger.info("Supervisor workflow completed: %s", result["status"])
    return cast(SupervisorState, result)
