"""Resolved challenge previews share service rules but have no write capability."""

import importlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from rallybot.agents.challenge_agent import (
    ChallengeAgentDependencies, build_challenge_agent, create_challenge_dependencies, run_challenge_agent,
)
from rallybot.agents.models import AgentInput, AgentOutput
from rallybot.config import get_settings
from rallybot.intent.models import ClassifiedIntent, IntentType
from rallybot.ladder.repository import PlayerRepository
from rallybot.tools.player_tools import PlayerSummary, UnknownPlayerError
from rallybot.workflows.supervisor_graph import create_supervisor_dependencies, run_supervisor

pytestmark = pytest.mark.usefixtures("agent_environment")


def request(name: str | None = "Sam Wilson", player_id: int | None = 1) -> AgentInput:
    return AgentInput(intent=IntentType.CHALLENGE_PLAYER, requester_player_id=player_id, opponent_name=name)


def test_graph_compilation() -> None:
    graph = build_challenge_agent(ChallengeAgentDependencies(Mock(), Mock(), Mock()))
    assert set(graph.get_graph().nodes) == {"__start__", "__end__", "validate_request", "prepare_preview"}
    assert graph.checkpointer is None


def test_service_delegation() -> None:
    lookup = Mock(return_value=PlayerSummary(1, "Requester", 1000))
    resolve = Mock(return_value=PlayerSummary(2, "Sam Wilson", 1100))
    prepare = Mock()
    output = run_challenge_agent(request(" Sam Wilson "), dependencies=ChallengeAgentDependencies(lookup, resolve, prepare))
    lookup.assert_called_once_with(1)
    resolve.assert_called_once_with("Sam Wilson")
    prepare.assert_called_once_with(1, 2)
    assert output.status == "COMPLETED"
    assert output.data["opponent_player_id"] == 2
    assert output.data["requires_confirmation"] is True
    assert output.data["challenge_created"] is False


def test_preview_is_read_only_and_has_no_private_identity(agent_players: PlayerRepository, agent_no_writes: None) -> None:
    output = run_challenge_agent(request("sam wilson"), dependencies=create_challenge_dependencies())
    assert isinstance(output, AgentOutput)
    assert output.status == "COMPLETED"
    assert output.data["opponent_resolved"] is True
    assert "private-discord" not in output.model_dump_json()
    assert not Path("data/challenge_approvals.sqlite3").exists()


def test_unknown_opponent(agent_players: PlayerRepository) -> None:
    output = run_challenge_agent(request("No Such Player"), dependencies=create_challenge_dependencies())
    assert output.status == "CLARIFICATION_NEEDED"
    assert output.error_code == "UNKNOWN_OPPONENT"
    assert output.data == {}


def test_name_is_not_interpreted_as_an_id(agent_players: PlayerRepository) -> None:
    output = run_challenge_agent(request("2"), dependencies=create_challenge_dependencies())
    assert output.error_code == "UNKNOWN_OPPONENT"


def test_ambiguous_opponent(agent_players: PlayerRepository) -> None:
    agent_players.create_player("SAM WILSON")
    output = run_challenge_agent(request(), dependencies=create_challenge_dependencies())
    assert output.status == "CLARIFICATION_NEEDED"
    assert output.error_code == "AMBIGUOUS_OPPONENT"


def test_self_challenge_is_rejected(agent_players: PlayerRepository) -> None:
    output = run_challenge_agent(request("Agent Requester"), dependencies=create_challenge_dependencies())
    assert output.status == "ERROR"
    assert output.error_code == "SELF_CHALLENGE"


def test_existing_pending_pair_is_ineligible(agent_players: PlayerRepository) -> None:
    from rallybot.challenges.repository import ChallengeRepository
    ChallengeRepository(get_settings().database_url).create_challenge(2, 1)
    output = run_challenge_agent(request(), dependencies=create_challenge_dependencies())
    assert output.status == "ERROR"
    assert output.error_code == "CHALLENGE_INELIGIBLE"


@pytest.mark.parametrize(("player_id", "name", "code"), [(None, "Sam Wilson", "MISSING_REQUESTER"),
    (1, None, "MISSING_OPPONENT"), (1, "  ", "MISSING_OPPONENT")])
def test_missing_details_never_access_business_data(player_id: int | None, name: str | None, code: str) -> None:
    blocked = Mock(side_effect=AssertionError("No read should occur"))
    result = run_challenge_agent(request(name, player_id), dependencies=ChallengeAgentDependencies(blocked, blocked, blocked))
    assert result.error_code == code
    blocked.assert_not_called()


def test_unknown_requester_does_not_resolve_opponent() -> None:
    resolve, prepare = Mock(), Mock()
    output = run_challenge_agent(request(), dependencies=ChallengeAgentDependencies(
        Mock(side_effect=UnknownPlayerError("private-identity")), resolve, prepare))
    assert output.error_code == "REQUESTER_NOT_FOUND"
    resolve.assert_not_called()
    prepare.assert_not_called()


def test_database_error_is_sanitized(caplog: pytest.LogCaptureFixture) -> None:
    deps = ChallengeAgentDependencies(Mock(return_value=PlayerSummary(1, "Requester", 1000)),
        Mock(side_effect=RuntimeError("secret-path-password")), Mock())
    output = run_challenge_agent(request(), dependencies=deps)
    assert output.error_code == "CAPABILITY_UNAVAILABLE"
    assert "secret-path-password" not in caplog.text + output.model_dump_json()


def test_requester_validation_is_required_before_personal_data(agent_players: PlayerRepository) -> None:
    output = run_challenge_agent(request(player_id=999), dependencies=create_challenge_dependencies())
    assert output.error_code == "REQUESTER_NOT_FOUND"


def test_supervisor_delegates_challenge_preview(monkeypatch: pytest.MonkeyPatch, agent_players: PlayerRepository) -> None:
    module = importlib.import_module("rallybot.workflows.supervisor_graph")
    agent = Mock(return_value=AgentOutput(agent_name="challenge", status="COMPLETED", message="Delegated",
                                         data={"requires_confirmation": True}))
    monkeypatch.setattr(module, "run_challenge_agent", agent)
    classifier = Mock(classify=Mock(return_value=ClassifiedIntent(intent=IntentType.CHALLENGE_PLAYER,
        confidence=0.9, opponent_name="Sam Wilson", player_name="Someone Else", clarification_needed=False)))
    result = run_supervisor("Challenge Sam Wilson as Someone Else", 1,
        dependencies=replace(create_supervisor_dependencies(), classifier=classifier))
    passed = agent.call_args.args[0]
    assert passed.requester_player_id == 1
    assert passed.opponent_name == "Sam Wilson"
    assert result["result"]["agent_name"] == "challenge"


def test_supervisor_does_not_delegate_without_identity(monkeypatch: pytest.MonkeyPatch, agent_players: PlayerRepository) -> None:
    module = importlib.import_module("rallybot.workflows.supervisor_graph")
    agent = Mock()
    monkeypatch.setattr(module, "run_challenge_agent", agent)
    classifier = Mock(classify=Mock(return_value=ClassifiedIntent(intent=IntentType.CHALLENGE_PLAYER,
        confidence=0.9, opponent_name="Sam Wilson", clarification_needed=False)))
    result = run_supervisor("Challenge Sam Wilson", dependencies=replace(create_supervisor_dependencies(), classifier=classifier))
    assert result["status"] == "CLARIFICATION_NEEDED"
    assert result["error_code"] == "MISSING_REQUESTER"
    agent.assert_not_called()


def test_supervisor_resolves_and_checks_preview_without_writes(agent_players: PlayerRepository, agent_no_writes: None) -> None:
    classifier = Mock(classify=Mock(return_value=ClassifiedIntent(intent=IntentType.CHALLENGE_PLAYER,
        confidence=0.9, opponent_name="Sam Wilson", clarification_needed=False)))
    result = run_supervisor("Challenge Sam Wilson", 1,
        dependencies=replace(create_supervisor_dependencies(), classifier=classifier))
    assert result["status"] == "COMPLETED"
    assert result["result"]["data"]["opponent_player_id"] == 2


def test_preview_does_not_start_or_resume_approval(
    agent_players: PlayerRepository, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rallybot.workflows.challenge_approval_graph import ChallengeApprovalWorkflow
    from rallybot.challenges.service import ChallengeService
    blocked = Mock(side_effect=AssertionError("Write capability must not be invoked"))
    monkeypatch.setattr(ChallengeApprovalWorkflow, "start", blocked)
    monkeypatch.setattr(ChallengeApprovalWorkflow, "resume", blocked)
    monkeypatch.setattr(ChallengeApprovalWorkflow, "resume_local", blocked)
    monkeypatch.setattr(ChallengeService, "create_challenge", blocked)
    monkeypatch.setattr(ChallengeService, "create_approved_challenge", blocked)
    output = run_challenge_agent(request(), dependencies=create_challenge_dependencies())
    assert output.status == "COMPLETED"
    blocked.assert_not_called()
