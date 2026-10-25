"""Human approval integration using only isolated business/checkpoint databases."""

import importlib
import os
import socket
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from rallybot.challenges.models import ChallengeStatus
from rallybot.challenges.repository import ChallengeRepository
from rallybot.challenges.service import ChallengeService, ChallengeValidationError
from rallybot.config import Settings, get_settings
from rallybot.database.connection import database_connection, initialize_database
from rallybot.ladder.repository import PlayerRepository
from rallybot.workflows.challenge_approval_graph import (
    ApprovalActor, ApprovalError, ChallengeApprovalWorkflow, build_challenge_approval_graph,
    local_principal, resolve_opponent,
)


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields or name.startswith(("LANGSMITH_", "LANGCHAIN_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///business.db")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("Network forbidden")))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def workflow() -> ChallengeApprovalWorkflow:
    initialize_database("sqlite:///business.db")
    players = PlayerRepository("sqlite:///business.db")
    players.create_player("Test Challenger", discord_user_id="synthetic-111")
    players.create_player("Test Opponent", discord_user_id="synthetic-222")
    return ChallengeApprovalWorkflow("sqlite:///business.db", Path("checkpoints.sqlite3"))


@pytest.fixture
def actor() -> ApprovalActor:
    return ApprovalActor(1, "trusted-test-principal")


def count_challenges() -> int:
    with database_connection("sqlite:///business.db") as connection:
        return connection.execute("SELECT COUNT(*) FROM challenges").fetchone()[0]


def test_unique_opponent_resolution(workflow: ChallengeApprovalWorkflow) -> None:
    assert resolve_opponent(workflow.players, "  test opponent ").id == 2
    assert resolve_opponent(workflow.players, 2).display_name == "Test Opponent"


@pytest.mark.parametrize("opponent", ["Unknown", "", 999, 0, True, None])
def test_unknown_opponent(workflow: ChallengeApprovalWorkflow, opponent: object) -> None:
    with pytest.raises(ApprovalError, match="UNKNOWN_OPPONENT"):
        resolve_opponent(workflow.players, opponent)
    assert count_challenges() == 0


def test_ambiguous_opponent(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    workflow.players.create_player("TEST OPPONENT")
    with pytest.raises(ApprovalError, match="AMBIGUOUS_OPPONENT"):
        workflow.start(actor, "Test Opponent")
    # Explicit immutable IDs still resolve uniquely.
    assert workflow.start(actor, 2)[1]["workflow_status"] == "AWAITING_APPROVAL"


def test_self_challenge(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    with pytest.raises(ApprovalError, match="SELF_CHALLENGE"):
        workflow.start(actor, 1)


def test_preparation_is_read_only_and_interrupt_payload_is_safe(
    workflow: ChallengeApprovalWorkflow, actor: ApprovalActor,
) -> None:
    before = Path("business.db").read_bytes()
    workflow_id, result = workflow.start(actor, 2)
    assert len(workflow_id) == 36
    assert result["workflow_status"] == "AWAITING_APPROVAL"
    assert result["approval_status"] == "PENDING"
    assert result["challenge_id"] is None
    payload = result["__interrupt__"][0].value
    assert payload == {"requester_player_id": 1, "opponent_player_id": 2,
        "opponent_display_name": "Test Opponent", "proposed_action": "CREATE_CHALLENGE",
        "question": "Create a pending challenge to Test Opponent?"}
    assert "synthetic-" not in repr(result)
    assert "principal" not in repr(result)
    assert Path("business.db").read_bytes() == before
    assert count_challenges() == 0


def test_approval_creates_one_challenge(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    workflow_id, _ = workflow.start(actor, "Test Opponent")
    result = workflow.resume(workflow_id, "approve", actor=actor)
    assert result["workflow_status"] == "COMPLETED"
    assert result["approval_status"] == "APPROVED"
    challenge = workflow.service.challenges.get_challenge(result["challenge_id"])
    assert (challenge.challenger_player_id, challenge.opponent_player_id) == (1, 2)
    assert challenge.status == ChallengeStatus.PENDING
    assert count_challenges() == 1


def test_rejection(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    result = workflow.resume(workflow_id, "reject", actor=actor)
    assert result["workflow_status"] == result["approval_status"] == "REJECTED"
    assert result["challenge_id"] is None
    assert count_challenges() == 0
    assert workflow.resume(workflow_id, "approve", actor=actor)["workflow_status"] == "REJECTED"


@pytest.mark.parametrize("decision", ["yes", "APPROVE", True, None, {"approved": True}, "secret-token"])
def test_invalid_decision(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor, decision: object) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    result = workflow.resume(workflow_id, decision, actor=actor)
    assert result["workflow_status"] == "ERROR"
    assert result["error_code"] == "INVALID_DECISION"
    assert "secret-token" not in repr(result)
    assert count_challenges() == 0


def test_reopen_checkpoint_and_resume(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    restarted = ChallengeApprovalWorkflow("sqlite:///business.db", Path("checkpoints.sqlite3"))
    result = restarted.resume(workflow_id, "approve", actor=actor)
    assert result["workflow_status"] == "COMPLETED"
    assert ChallengeRepository("sqlite:///business.db").get_challenge(result["challenge_id"]) is not None
    # Connections have all closed; databases can be renamed on Windows.
    Path("checkpoints.sqlite3").rename("reopened.sqlite3")


def test_replay_after_terminal_challenge_does_not_create_another(
    workflow: ChallengeApprovalWorkflow, actor: ApprovalActor,
) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    first = workflow.resume(workflow_id, "approve", actor=actor)
    workflow.service.challenges.transition_challenge(first["challenge_id"], 2, ChallengeStatus.ACCEPTED)
    assert workflow.resume(workflow_id, "approve", actor=actor) == first
    assert count_challenges() == 1


def test_atomic_receipt_survives_business_commit_before_checkpoint(
    workflow: ChallengeApprovalWorkflow, actor: ApprovalActor,
) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    # Simulate the business commit preceding a process crash/checkpoint save.
    committed = workflow.service.create_approved_challenge(1, 2, workflow_id)
    workflow.service.challenges.transition_challenge(committed.id, 2, ChallengeStatus.ACCEPTED)
    result = workflow.resume(workflow_id, "approve", actor=actor)
    assert result["challenge_id"] == committed.id
    assert result["workflow_status"] == "COMPLETED"
    assert count_challenges() == 1


def test_concurrent_resumes(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    def resume() -> dict:
        fresh = ChallengeApprovalWorkflow("sqlite:///business.db", Path("checkpoints.sqlite3"))
        return fresh.resume(workflow_id, "approve", actor=actor)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: resume(), range(2)))
    assert {item["challenge_id"] for item in results} == {1}
    assert count_challenges() == 1


def test_crash_after_commit_restarts_create_node_without_duplicate(
    workflow: ChallengeApprovalWorkflow, actor: ApprovalActor, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    original = workflow.service.create_approved_challenge
    def crash_after_commit(*args: object):
        original(*args)
        raise SystemExit("simulated process exit")
    monkeypatch.setattr(workflow.service, "create_approved_challenge", crash_after_commit)
    with pytest.raises(SystemExit):
        workflow.resume(workflow_id, "approve", actor=actor)
    assert count_challenges() == 1
    restarted = ChallengeApprovalWorkflow("sqlite:///business.db", Path("checkpoints.sqlite3"))
    result = restarted.resume(workflow_id, "approve", actor=actor)
    assert result["workflow_status"] == "COMPLETED"
    assert result["challenge_id"] == 1
    assert count_challenges() == 1


def test_two_processes_resume_only_one_business_write(workflow: ChallengeApprovalWorkflow) -> None:
    workflow_id, _ = workflow.start(ApprovalActor(1, local_principal()), 2)
    cli = Path(sys.executable).with_name("rallybot.exe")
    args = [str(cli), "challenge-approval-demo", "--workflow-id", workflow_id,
            "--decision", "approve", "--checkpoint-file", "checkpoints.sqlite3"]
    children = [subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    try:
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, stderr
            assert "Challenge ID: 1" in stdout
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
    assert count_challenges() == 1


@pytest.mark.parametrize("unauthorized", [ApprovalActor(2, "trusted-test-principal"),
    ApprovalActor(1, "other-principal"), ApprovalActor(True, "trusted-test-principal")])
def test_unauthorized_resume(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor, unauthorized: ApprovalActor) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    with pytest.raises(ApprovalError, match="UNAUTHORIZED"):
        workflow.resume(workflow_id, "approve", actor=unauthorized)
    assert count_challenges() == 0
    assert workflow.resume(workflow_id, "reject", actor=actor)["workflow_status"] == "REJECTED"


def test_local_identity_mismatch(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    with pytest.raises(ApprovalError, match="UNAUTHORIZED"):
        workflow.resume_local(workflow_id, "approve")


def test_eligibility_rechecked_after_pause(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    other = workflow.service.challenges.create_challenge(2, 1)
    result = workflow.resume(workflow_id, "approve", actor=actor)
    assert result["workflow_status"] == "ERROR"
    assert result["error_code"] == "ELIGIBILITY_CHANGED"
    assert count_challenges() == 1
    assert workflow.service.challenges.get_challenge(other.id).status == ChallengeStatus.PENDING


def test_preexisting_pending_challenge_is_rejected(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    workflow.service.challenges.create_challenge(2, 1)
    _, result = workflow.start(actor, 2)
    assert result["workflow_status"] == "ERROR"
    assert "__interrupt__" not in result
    assert count_challenges() == 1


def test_requester_must_exist(workflow: ChallengeApprovalWorkflow) -> None:
    _, result = workflow.start(ApprovalActor(999, "trusted-principal"), 2)
    assert result["error_code"] == "ELIGIBILITY_FAILED"
    assert count_challenges() == 0


def test_deleted_opponent_cannot_be_approved(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    with database_connection("sqlite:///business.db") as connection:
        connection.execute("DELETE FROM players WHERE id = ?", (2,))
    result = workflow.resume(workflow_id, "approve", actor=actor)
    assert result["error_code"] == "ELIGIBILITY_CHANGED"
    assert count_challenges() == 0


def test_checkpoint_separate_from_business(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    with pytest.raises(ApprovalError, match="SEPARATE_FILE_STORAGE_REQUIRED"):
        ChallengeApprovalWorkflow("sqlite:///business.db", Path("business.db"))
    workflow.start(actor, 2)
    with database_connection("sqlite:///business.db") as connection:
        assert connection.execute("SELECT name FROM sqlite_schema WHERE name='checkpoints'").fetchone() is None


def test_missing_or_malformed_workflow(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    for workflow_id in ("bad-id", str(uuid4())):
        with pytest.raises(ApprovalError, match="WORKFLOW_NOT_AVAILABLE"):
            workflow.resume(workflow_id, "approve", actor=actor)


def test_database_binding(workflow: ChallengeApprovalWorkflow, actor: ApprovalActor) -> None:
    workflow_id, _ = workflow.start(actor, 2)
    initialize_database("sqlite:///other.db")
    other = ChallengeApprovalWorkflow("sqlite:///other.db", Path("checkpoints.sqlite3"))
    with pytest.raises(ApprovalError, match="UNAUTHORIZED"):
        other.resume(workflow_id, "approve", actor=actor)


def test_version_three_upgrade_preserves_challenges(workflow: ChallengeApprovalWorkflow) -> None:
    existing = workflow.service.challenges.create_challenge(1, 2)
    with database_connection("sqlite:///business.db") as connection:
        connection.execute("DROP TABLE challenge_approval_receipts")
        connection.execute("UPDATE _rallybot_metadata SET value = '3' WHERE key = 'schema_version'")
    initialize_database("sqlite:///business.db")
    initialize_database("sqlite:///business.db")
    assert workflow.service.challenges.get_challenge(existing.id) == existing
    assert workflow.players.get_player(1).display_name == "Test Challenger"
    with database_connection("sqlite:///business.db") as connection:
        assert connection.execute("SELECT value FROM _rallybot_metadata WHERE key='schema_version'").fetchone()[0] == "4"
        assert connection.execute("SELECT COUNT(*) FROM challenge_approval_receipts").fetchone()[0] == 0


def test_command_resume_api_and_recovery_from_approved_node(workflow: ChallengeApprovalWorkflow) -> None:
    # Low-level API use in a trusted test also demonstrates supported LangGraph APIs.
    workflow_id = str(uuid4())
    config = {"configurable": {"thread_id": workflow_id}}
    with SqliteSaver.from_conn_string("lowlevel.sqlite3") as saver:
        graph = build_challenge_approval_graph(workflow.service, saver)
        initial = {"requester_player_id": 1, "opponent_player_id": 2,
            "opponent_display_name": "", "approval_status": "PENDING", "challenge_id": None,
            "workflow_status": "PREPARING", "error_code": None}
        assert graph.invoke(initial, config)["__interrupt__"]
        result = graph.invoke(Command(resume="approve"), config)
        assert result["workflow_status"] == "COMPLETED"


def test_cli_restart_and_resume_in_separate_processes(workflow: ChallengeApprovalWorkflow) -> None:
    cli = Path(sys.executable).with_name("rallybot.exe")
    started = subprocess.run([str(cli), "challenge-approval-demo", "--requester-id", "1", "--opponent-id", "2"],
                             capture_output=True, text=True, check=True)
    assert "Development-only" in started.stdout
    assert "AWAITING_APPROVAL" in started.stdout
    workflow_id = next(line.split(": ", 1)[1] for line in started.stdout.splitlines() if line.startswith("Workflow ID:"))
    resumed = subprocess.run([str(cli), "challenge-approval-demo", "--workflow-id", workflow_id, "--decision", "approve"],
                             capture_output=True, text=True, check=True)
    assert "Status: COMPLETED" in resumed.stdout
    assert "Challenge ID: 1" in resumed.stdout
    assert "synthetic-" not in started.stdout + resumed.stdout + started.stderr + resumed.stderr
    assert count_challenges() == 1


def test_cli_development_only(workflow: ChallengeApprovalWorkflow, monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    module = importlib.import_module("rallybot.main")
    assert module.main(["challenge-approval-demo", "--requester-id", "1", "--opponent-id", "2"]) == 1
    assert "only available in development" in capsys.readouterr().err
    assert not Path("data/challenge_approvals.sqlite3").exists()


def test_cli_rejects_actor_identity_on_resume() -> None:
    module = importlib.import_module("rallybot.main")
    with pytest.raises(SystemExit) as error:
        module.main(["challenge-approval-demo", "--workflow-id", str(uuid4()), "--decision", "approve", "--requester-id", "1"])
    assert error.value.code == 2


def test_supervisor_challenge_preview_still_read_only(workflow: ChallengeApprovalWorkflow) -> None:
    from dataclasses import replace
    from rallybot.intent.models import ClassifiedIntent, IntentType
    from rallybot.workflows.supervisor_graph import create_supervisor_dependencies, run_supervisor

    classifier = Mock(classify=Mock(return_value=ClassifiedIntent(intent=IntentType.CHALLENGE_PLAYER,
        confidence=0.9, opponent_name="Test Opponent", clarification_needed=False)))
    dependencies = replace(create_supervisor_dependencies(get_settings()), classifier=classifier)
    result = run_supervisor("Challenge Test Opponent", 1, dependencies=dependencies)
    assert result["result"]["data"]["requires_confirmation"] is True
    assert count_challenges() == 0
