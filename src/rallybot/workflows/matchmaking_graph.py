"""Deterministic database-backed matchmaking through injectable player tools."""

import logging
from functools import partial
from typing import Literal, NotRequired, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import tracing_context

from rallybot.tools.player_tools import PlayerSummary, PlayerToolset, create_player_tools
from rallybot.tools.player_tools import RankedPlayer
from rallybot.llm.matchmaking_explainer import ExplanationResult, MatchmakingExplainer, fallback_explanation


class Opponent(TypedDict):
    id: int
    name: str
    points: int
    rank: int


class MatchmakingState(TypedDict):
    player_id: int
    player_name: str
    player_points: int
    available_players: list[Opponent]
    recommended_opponent: str | None
    recommended_opponent_id: int | None
    status: str
    explanation: NotRequired[str | None]
    explanation_source: NotRequired[Literal["OLLAMA", "FALLBACK"] | None]


class MatchmakingUpdate(TypedDict, total=False):
    player_name: str
    player_points: int
    available_players: list[Opponent]
    recommended_opponent: str | None
    recommended_opponent_id: int | None
    status: str
    explanation: str | None
    explanation_source: Literal["OLLAMA", "FALLBACK"] | None


def load_player(state: MatchmakingState, *, tools: PlayerToolset) -> MatchmakingUpdate:
    logging.getLogger(__name__).info("Matchmaking node executing: load_player")
    player_id = state.get("player_id")
    if type(player_id) is not int or not 1 <= player_id <= 9223372036854775807:
        raise ValueError("Player ID must be a positive SQLite integer.")
    player = tools.get_player(player_id)
    return {"player_name": player.name, "player_points": player.points,
            "available_players": [], "recommended_opponent": None,
            "recommended_opponent_id": None, "explanation": None,
            "explanation_source": None, "status": "PLAYER_LOADED"}


def find_opponents(state: MatchmakingState, *, tools: PlayerToolset) -> MatchmakingUpdate:
    logging.getLogger(__name__).info("Matchmaking node executing: find_opponents")
    player = PlayerSummary(state["player_id"], state["player_name"], state["player_points"])
    eligible = tools.filter_opponents(player, tools.list_opponents(player.id))
    opponents: list[Opponent] = [
        {"id": candidate.id, "name": candidate.name, "points": candidate.points, "rank": candidate.rank}
        for candidate in eligible
    ]
    return {"available_players": opponents, "status": "OPPONENTS_FOUND"}


def recommend_match(state: MatchmakingState) -> MatchmakingUpdate:
    """Select from already ordered tool results; this node performs no I/O."""
    logging.getLogger(__name__).info("Matchmaking node executing: recommend_match")
    if not state["available_players"]:
        return {"recommended_opponent": None, "recommended_opponent_id": None, "status": "NO_OPPONENTS"}
    opponent = state["available_players"][0]
    return {"recommended_opponent": opponent["name"], "recommended_opponent_id": opponent["id"],
            "status": "MATCH_RECOMMENDED"}


def explain_recommendation(
    state: MatchmakingState, *, explainer: MatchmakingExplainer | None = None
) -> MatchmakingUpdate:
    """Only explanation fields can change; selection and ranking stay authoritative."""
    logger = logging.getLogger(__name__)
    logger.info("Matchmaking node executing: explain_recommendation")
    if explainer is None or state["recommended_opponent_id"] is None:
        return {"explanation": None, "explanation_source": None}
    player = PlayerSummary(state["player_id"], state["player_name"], state["player_points"])
    opponents = [RankedPlayer(p["id"], p["name"], p["points"], p["rank"]) for p in state["available_players"]]
    selected = next((p for p in opponents if p.id == state["recommended_opponent_id"]), None)
    if selected is None:
        logger.warning("AI explanation skipped because verified recommendation context is unavailable")
        return {"explanation": "The deterministic recommendation remains unchanged.", "explanation_source": "FALLBACK"}
    try:
        result = explainer.explain(player, opponents, selected)
        if (not isinstance(result, ExplanationResult) or result.explanation_source not in ("OLLAMA", "FALLBACK")
                or not isinstance(result.explanation, str) or not result.explanation.strip()):
            raise ValueError("Invalid explanation result")
    except Exception:
        logger.warning("AI explanation failed; preserving deterministic recommendation")
        result = fallback_explanation(player, selected)
    return {"explanation": result.explanation, "explanation_source": result.explanation_source}


def build_matchmaking_graph(
    tools: PlayerToolset | None = None, *, explainer: MatchmakingExplainer | None = None
) -> CompiledStateGraph:
    tools = tools if tools is not None else create_player_tools()
    graph = StateGraph(MatchmakingState)
    graph.add_node("load_player", partial(load_player, tools=tools))
    graph.add_node("find_opponents", partial(find_opponents, tools=tools))
    graph.add_node("recommend_match", recommend_match)
    graph.add_node("explain_recommendation", partial(explain_recommendation, explainer=explainer))
    graph.add_edge(START, "load_player")
    graph.add_edge("load_player", "find_opponents")
    graph.add_edge("find_opponents", "recommend_match")
    graph.add_edge("recommend_match", "explain_recommendation")
    graph.add_edge("explain_recommendation", END)
    return graph.compile()


def run_matchmaking(
    player_id: int, *, tools: PlayerToolset | None = None, explainer: MatchmakingExplainer | None = None
) -> MatchmakingState:
    logger = logging.getLogger(__name__)
    logger.info("Matchmaking workflow started")
    state: MatchmakingState = {
        "player_id": player_id, "player_name": "", "player_points": 0,
        "available_players": [], "recommended_opponent": None,
        "recommended_opponent_id": None, "status": "STARTED",
    }
    try:
        with tracing_context(enabled=False):
            result = build_matchmaking_graph(tools, explainer=explainer).invoke(state)
    except Exception:
        logger.error("Matchmaking workflow failed")
        raise
    logger.info("Matchmaking workflow completed")
    return cast(MatchmakingState, result)
