"""Ladder graph, shared contracts, safe failures, and supervisor delegation."""

import importlib
import logging
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from rallybot.agents.ladder_agent import LadderAgentDependencies, build_ladder_agent, create_ladder_dependencies, run_ladder_agent
from rallybot.agents.models import AgentInput, AgentOutput, RequestContext
from rallybot.config import get_settings
from rallybot.intent.models import ClassifiedIntent, IntentType
from rallybot.ladder.models import PlayerRanking
from rallybot.ladder.repository import PlayerRepository
from rallybot.workflows.supervisor_graph import create_supervisor_dependencies, run_supervisor

pytestmark = pytest.mark.usefixtures("agent_environment")


def request() -> AgentInput:
    return AgentInput(intent=IntentType.VIEW_LADDER)


def test_graph_compilation_and_typed_output() -> None:
    getter = Mock(return_value=[PlayerRanking(1, "Test Player", 1000)])
    graph = build_ladder_agent(LadderAgentDependencies(getter))
    assert set(graph.get_graph().nodes) == {"__start__", "__end__", "validate_request", "retrieve_rankings"}
    assert graph.checkpointer is None
    initial = {"request": request(), "result": None}
    result = graph.invoke(initial)["result"]
    assert isinstance(result, AgentOutput)
    assert result.agent_name == "ladder"
    assert result.data == {"rankings": [{"rank": 1, "player_name": "Test Player", "points": 1000}]}
    getter.assert_called_once_with()
    assert initial["result"] is None


def test_ladder_service_delegation(monkeypatch: pytest.MonkeyPatch, agent_players: PlayerRepository) -> None:
    module = importlib.import_module("rallybot.agents.ladder_agent")
    getter = Mock(return_value=[])
    monkeypatch.setattr(module, "get_ladder", getter)
    result = run_ladder_agent(request(), dependencies=create_ladder_dependencies(agent_players))
    getter.assert_called_once_with(agent_players)
    assert result.status == "COMPLETED"
    assert result.data == {"rankings": []}
    assert "empty" in result.message.lower()


def test_database_rankings_are_read_only(agent_players: PlayerRepository, agent_no_writes: None) -> None:
    result = run_ladder_agent(request(), dependencies=create_ladder_dependencies(agent_players))
    assert [entry["player_name"] for entry in result.data["rankings"]] == ["Sam Wilson", "Agent Requester"]
    assert "private-discord" not in result.model_dump_json()
    # No retained connection prevents a Windows rename after execution.
    Path("agents.db").rename("closed.db")
    Path("closed.db").rename("agents.db")


def test_database_error_is_sanitized(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        result = run_ladder_agent(request(), dependencies=LadderAgentDependencies(
            Mock(side_effect=RuntimeError("private-token-and-filesystem"))))
    assert result.status == "ERROR"
    assert result.error_code == "CAPABILITY_UNAVAILABLE"
    assert "private-token" not in result.model_dump_json() + caplog.text


@pytest.mark.parametrize("bad", ["VIEW_LADDER", {"intent": "VIEW_LADDER"},
    AgentInput.model_construct(intent="VIEW_LADDER", requester_player_id=None),
    AgentInput(intent=IntentType.FIND_OPPONENT)])
def test_unvalidated_or_wrong_intent_cannot_run_service(bad: object) -> None:
    getter = Mock()
    result = run_ladder_agent(bad, dependencies=LadderAgentDependencies(getter))
    assert result.error_code == "INVALID_AGENT_INPUT"
    getter.assert_not_called()


@pytest.mark.parametrize("invalid_id", [0, -1, True, "1", 2**63])
def test_input_identity_validation(invalid_id: object) -> None:
    with pytest.raises(ValidationError):
        AgentInput(intent=IntentType.VIEW_LADDER, requester_player_id=invalid_id)


def test_input_output_contracts_and_context_allowlist() -> None:
    request_id = uuid4()
    value = AgentInput(intent=IntentType.VIEW_LADDER, request_context=RequestContext(request_id=request_id))
    assert value.request_context.request_id == request_id
    with pytest.raises(ValidationError):
        AgentInput(intent="VIEW_LADDER")
    with pytest.raises(ValidationError):
        RequestContext(discord_user_id="private-discord")
    with pytest.raises(ValidationError):
        AgentInput(intent=IntentType.VIEW_LADDER, discord_bot_token="secret")
    with pytest.raises(ValidationError):
        AgentOutput(agent_name="ladder", status="WRITE_ALLOWED", message="invalid")


def test_model_construct_cannot_bypass_input_validation(caplog: pytest.LogCaptureFixture) -> None:
    getter = Mock()
    bad = AgentInput.model_construct(intent=IntentType.VIEW_LADDER, requester_player_id="private-token")
    result = run_ladder_agent(bad, dependencies=LadderAgentDependencies(getter))
    assert result.error_code == "INVALID_AGENT_INPUT"
    assert "private-token" not in caplog.text + result.model_dump_json()
    getter.assert_not_called()


def test_supervisor_delegates_to_ladder_agent(monkeypatch: pytest.MonkeyPatch, agent_players: PlayerRepository) -> None:
    module = importlib.import_module("rallybot.workflows.supervisor_graph")
    agent = Mock(return_value=AgentOutput(agent_name="ladder", status="COMPLETED", message="Delegated", data={"rankings": []}))
    monkeypatch.setattr(module, "run_ladder_agent", agent)
    classifier = Mock(classify=Mock(return_value=ClassifiedIntent(intent=IntentType.VIEW_LADDER,
        confidence=0.9, clarification_needed=False)))
    dependencies = replace(create_supervisor_dependencies(get_settings()), classifier=classifier)
    result = run_supervisor("Private original message", 1, dependencies=dependencies)
    passed = agent.call_args.args[0]
    assert isinstance(passed, AgentInput)
    assert passed.intent is IntentType.VIEW_LADDER
    assert "Private original message" not in passed.model_dump_json()
    assert result["result"]["agent_name"] == "ladder"
    classifier.classify.assert_called_once()
