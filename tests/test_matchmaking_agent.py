"""Read-only deterministic matchmaking agent and explicit AI explanation opt-in."""

import importlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from rallybot.agents.matchmaking_agent import (
    MatchmakingAgentDependencies, build_matchmaking_agent, create_matchmaking_dependencies, run_matchmaking_agent,
)
from rallybot.agents.models import AgentInput, AgentOutput
from rallybot.config import get_settings
from rallybot.intent.models import ClassifiedIntent, IntentType
from rallybot.ladder.repository import PlayerRepository
from rallybot.llm.matchmaking_explainer import ExplanationResult
from rallybot.tools.player_tools import PlayerSummary, UnknownPlayerError
from rallybot.workflows.supervisor_graph import create_supervisor_dependencies, run_supervisor

pytestmark = pytest.mark.usefixtures("agent_environment")


def request(player_id: int | None = 1) -> AgentInput:
    return AgentInput(intent=IntentType.FIND_OPPONENT, requester_player_id=player_id)


def test_graph_compilation() -> None:
    graph = build_matchmaking_agent(MatchmakingAgentDependencies(Mock(), Mock()))
    assert set(graph.get_graph().nodes) == {"__start__", "__end__", "validate_request", "recommend_opponent"}
    assert graph.checkpointer is None


def test_existing_workflow_delegation(agent_players: PlayerRepository, monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("rallybot.agents.matchmaking_agent")
    original = module.run_matchmaking
    run = Mock(wraps=original)
    monkeypatch.setattr(module, "run_matchmaking", run)
    dependencies = create_matchmaking_dependencies(get_settings())
    result = run_matchmaking_agent(request(), dependencies=dependencies)
    assert isinstance(result, AgentOutput)
    assert result.status == "COMPLETED"
    assert result.data["recommended_opponent_id"] == 2
    assert result.data["recommended_opponent"] == "Sam Wilson"
    assert result.data["eligible_opponents"] == [{"id": 2, "name": "Sam Wilson", "points": 1100, "rank": 1}]
    assert run.call_args.args == (1,)
    assert run.call_args.kwargs["explainer"] is None
    assert "explanation" not in result.data
    assert "private-discord" not in result.model_dump_json()


def test_read_only_and_connection_cleanup(agent_players: PlayerRepository, agent_no_writes: None) -> None:
    result = run_matchmaking_agent(request(), dependencies=create_matchmaking_dependencies())
    assert result.status == "COMPLETED"
    Path("agents.db").rename("closed.db")
    Path("closed.db").rename("agents.db")


def test_no_eligible_opponents(agent_players: PlayerRepository) -> None:
    far = agent_players.create_player("Far Away", 5000)
    result = run_matchmaking_agent(request(far.id), dependencies=create_matchmaking_dependencies())
    assert result.status == "COMPLETED"
    assert result.data["recommended_opponent_id"] is None
    assert result.data["eligible_opponents"] == []
    assert result.data["matchmaking_status"] == "NO_OPPONENTS"
    assert "No eligible" in result.message


def test_unknown_requester_prevents_matchmaking() -> None:
    recommend = Mock()
    dependencies = MatchmakingAgentDependencies(Mock(side_effect=UnknownPlayerError("private-id")), recommend)
    result = run_matchmaking_agent(request(), dependencies=dependencies)
    assert result.error_code == "REQUESTER_NOT_FOUND"
    recommend.assert_not_called()
    assert "private-id" not in result.model_dump_json()


def test_real_unknown_requester(agent_players: PlayerRepository) -> None:
    assert run_matchmaking_agent(request(999), dependencies=create_matchmaking_dependencies()).error_code == "REQUESTER_NOT_FOUND"


def test_missing_identity_cannot_be_inferred_from_name() -> None:
    lookup, recommend = Mock(), Mock()
    result = run_matchmaking_agent(AgentInput(intent=IntentType.FIND_OPPONENT,
        opponent_name="Agent Requester"), dependencies=MatchmakingAgentDependencies(lookup, recommend))
    assert result.status == "CLARIFICATION_NEEDED"
    assert result.error_code == "MISSING_REQUESTER"
    lookup.assert_not_called()
    recommend.assert_not_called()


def test_database_failure_is_sanitized() -> None:
    dependencies = MatchmakingAgentDependencies(Mock(return_value=PlayerSummary(1, "Requester", 1000)),
        Mock(side_effect=RuntimeError("password-in-db-error")))
    result = run_matchmaking_agent(request(), dependencies=dependencies)
    assert result.status == "ERROR"
    assert result.error_code == "CAPABILITY_UNAVAILABLE"
    assert "password-in-db-error" not in result.model_dump_json()


def test_ai_is_disabled_even_when_explainer_is_supplied(agent_players: PlayerRepository) -> None:
    explainer = Mock()
    dependencies = create_matchmaking_dependencies(explainer=explainer)
    result = run_matchmaking_agent(request(), dependencies=dependencies)
    explainer.explain.assert_not_called()
    assert "explanation" not in result.data
    assert result.data["recommended_opponent_id"] == 2


def test_explicit_ai_opt_in_reuses_explanation_workflow(agent_players: PlayerRepository) -> None:
    explainer = Mock(explain=Mock(return_value=ExplanationResult("Verified explanation.", "OLLAMA")))
    dependencies = create_matchmaking_dependencies(enable_ai_explanation=True, explainer=explainer)
    result = run_matchmaking_agent(request(), dependencies=dependencies)
    explainer.explain.assert_called_once()
    assert result.data["explanation"] == "Verified explanation."
    assert result.data["explanation_source"] == "OLLAMA"
    assert result.data["recommended_opponent_id"] == 2


def test_ai_failure_preserves_deterministic_result(agent_players: PlayerRepository) -> None:
    explainer = Mock(explain=Mock(side_effect=TimeoutError("private-token")))
    result = run_matchmaking_agent(request(), dependencies=create_matchmaking_dependencies(
        enable_ai_explanation=True, explainer=explainer))
    assert result.status == "COMPLETED"
    assert result.data["explanation_source"] == "FALLBACK"
    assert result.data["recommended_opponent_id"] == 2
    assert "private-token" not in result.model_dump_json()


def test_no_eligible_opponent_does_not_call_ai(agent_players: PlayerRepository) -> None:
    player = agent_players.create_player("Far Away", 9000)
    explainer = Mock()
    result = run_matchmaking_agent(request(player.id), dependencies=create_matchmaking_dependencies(
        enable_ai_explanation=True, explainer=explainer))
    explainer.explain.assert_not_called()
    assert result.data["explanation"] is None


def test_supervisor_delegates_matchmaking(monkeypatch: pytest.MonkeyPatch, agent_players: PlayerRepository) -> None:
    module = importlib.import_module("rallybot.workflows.supervisor_graph")
    agent = Mock(return_value=AgentOutput(agent_name="matchmaking", status="COMPLETED", message="Delegated"))
    monkeypatch.setattr(module, "run_matchmaking_agent", agent)
    classifier = Mock(classify=Mock(return_value=ClassifiedIntent(intent=IntentType.FIND_OPPONENT,
        confidence=0.9, player_name="Someone Else", clarification_needed=False)))
    result = run_supervisor("Use Someone Else's identity", 1,
        dependencies=replace(create_supervisor_dependencies(), classifier=classifier))
    assert agent.call_args.args[0].requester_player_id == 1
    assert agent.call_args.kwargs["dependencies"].explanation_enabled is False
    assert result["result"]["agent_name"] == "matchmaking"
