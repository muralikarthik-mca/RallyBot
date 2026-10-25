"""Fixed read-only routing with injected classification and temporary SQLite."""

import importlib
import logging
import os
import socket
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from rallybot.config import Settings, get_settings
from rallybot.database.connection import initialize_database
from rallybot.intent.models import ClassifiedIntent, IntentType
from rallybot.logging_config import configure_logging
from rallybot.tools.player_tools import UnknownPlayerError
from rallybot.workflows.supervisor_graph import (
    SupervisorDependencies, SupervisorState, build_supervisor_graph,
    create_supervisor_dependencies, run_supervisor,
)


@pytest.fixture(autouse=True)
def isolated_supervisor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields or name.startswith(("LANGCHAIN_", "LANGSMITH_", "OLLAMA_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///supervisor.db")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("External services are forbidden in supervisor tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    logger = logging.getLogger("rallybot")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    configure_logging("INFO")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()
        for handler in list(logger.handlers):
            if handler not in handlers:
                logger.removeHandler(handler)
                handler.close()
        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate


def intent(kind: IntentType, *, clarification: bool = False, opponent: str | None = None) -> ClassifiedIntent:
    return ClassifiedIntent(intent=kind, confidence=0.9, clarification_needed=clarification, opponent_name=opponent)


@pytest.fixture
def dependencies() -> SupervisorDependencies:
    from rallybot.ladder.repository import PlayerRepository
    from rallybot.challenges.repository import ChallengeRepository

    settings = get_settings()
    initialize_database(settings.database_url)
    players = PlayerRepository(settings.database_url)
    for name, points, identity in (("Requester", 1000, "private-id-111"),
                                  ("Opponent", 1100, "private-id-222"),
                                  ("Other", 900, "private-id-333")):
        players.create_player(name, points, identity)
    challenges = ChallengeRepository(settings.database_url)
    challenges.create_challenge(1, 2)
    challenges.create_challenge(2, 3)
    return replace(create_supervisor_dependencies(settings),
                   classifier=Mock(classify=Mock(return_value=intent(IntentType.VIEW_LADDER))))


def state() -> SupervisorState:
    return {"user_message": "Synthetic request", "requester_player_id": 1,
            "classified_intent": None, "route": None, "result": None,
            "status": "STARTED", "error_code": None}


def test_graph_compiles_with_fixed_conditional_handler_edges(dependencies: SupervisorDependencies) -> None:
    graph = build_supervisor_graph(dependencies)
    structure = graph.get_graph()
    assert set(structure.nodes) == {"__start__", "__end__", "classify_intent", "route_request",
        "matchmaking_handler", "ladder_handler", "challenge_handler", "court_handler",
        "clarification_handler", "unsupported_handler"}
    conditional_targets = {edge.target for edge in structure.edges if edge.source == "route_request" and edge.conditional}
    assert conditional_targets == {"matchmaking_handler", "ladder_handler", "challenge_handler",
                                   "court_handler", "clarification_handler", "unsupported_handler"}
    assert graph.checkpointer is None


@pytest.mark.parametrize(("kind", "route", "status"), [
    (IntentType.FIND_OPPONENT, "matchmaking", "COMPLETED"),
    (IntentType.VIEW_LADDER, "ladder", "COMPLETED"),
    (IntentType.VIEW_CHALLENGES, "challenges", "COMPLETED"),
    (IntentType.CHALLENGE_PLAYER, "challenge_preview", "COMPLETED"),
    (IntentType.COURT_REQUEST, "court", "NOT_IMPLEMENTED"),
    (IntentType.UNKNOWN, "unsupported", "UNSUPPORTED"),
])
def test_all_routes(dependencies: SupervisorDependencies, kind: IntentType, route: str, status: str) -> None:
    dependencies.classifier.classify.return_value = intent(kind, opponent="Other" if kind == IntentType.CHALLENGE_PLAYER else None)
    result = run_supervisor("Synthetic request", 1, dependencies=dependencies)
    assert result["route"] == route
    assert result["status"] == status
    assert result["result"]["capability"] == route
    if route == "matchmaking":
        assert result["result"]["data"]["recommended_opponent_id"] == 2
    if route == "court":
        assert result["result"]["data"] == {}


@pytest.mark.parametrize("kind", [IntentType.FIND_OPPONENT, IntentType.VIEW_LADDER,
    IntentType.VIEW_CHALLENGES, IntentType.CHALLENGE_PLAYER, IntentType.COURT_REQUEST])
def test_clarification_precedes_capability_execution(dependencies: SupervisorDependencies, kind: IntentType) -> None:
    dependencies.classifier.classify.return_value = intent(kind, clarification=True)
    blocked = Mock(side_effect=AssertionError("No capability should run"))
    dependencies = replace(dependencies, get_player=blocked, matchmaking=blocked, ladder=blocked, challenges=blocked)
    result = run_supervisor("Ambiguous request", 1, dependencies=dependencies)
    assert result["route"] == "clarification"
    assert result["status"] == "CLARIFICATION_NEEDED"
    blocked.assert_not_called()


def test_unknown_with_clarification_flag_still_returns_safe_unsupported(dependencies: SupervisorDependencies) -> None:
    dependencies.classifier.classify.return_value = intent(IntentType.UNKNOWN, clarification=True)
    result = run_supervisor("Unsupported", 1, dependencies=dependencies)
    assert result["route"] == "unsupported"
    assert result["status"] == "UNSUPPORTED"


@pytest.mark.parametrize("bad_result", ["VIEW_LADDER", {"intent": "VIEW_LADDER"},
    ClassifiedIntent.model_construct(intent=IntentType.VIEW_LADDER, confidence=99, clarification_needed=False)])
def test_classification_is_revalidated_before_routing(dependencies: SupervisorDependencies, bad_result: object) -> None:
    dependencies.classifier.classify.return_value = bad_result
    result = run_supervisor("Never trust raw strings", 1, dependencies=dependencies)
    assert result["route"] == "unsupported"
    assert result["error_code"] == "CLASSIFICATION_FAILED"


def test_llm_failure_is_safe(dependencies: SupervisorDependencies, capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    dependencies.classifier.classify.side_effect = TimeoutError("private-token")
    result = run_supervisor("Private user message", 1, dependencies=dependencies)
    assert result["status"] == "UNSUPPORTED"
    assert result["error_code"] == "CLASSIFICATION_FAILED"
    output = capsys.readouterr()
    for value in ("private-token", "Private user message", "Traceback"):
        assert value not in output.out + output.err


@pytest.mark.parametrize("kind", [IntentType.FIND_OPPONENT, IntentType.VIEW_CHALLENGES])
@pytest.mark.parametrize("player_id", [None, 0, -1, True, 2**63])
def test_missing_or_invalid_requester_is_not_resolved_from_text(
    dependencies: SupervisorDependencies, kind: IntentType, player_id: object,
) -> None:
    candidate = intent(kind).model_copy(update={"player_name": "Requester"})
    dependencies.classifier.classify.return_value = candidate
    blocked = Mock()
    dependencies = replace(dependencies, get_player=blocked, matchmaking=blocked, challenges=blocked)
    result = run_supervisor("Use Requester or player ID 1", player_id, dependencies=dependencies)
    assert result["route"] == "clarification"
    assert result["error_code"] == "MISSING_REQUESTER"
    dependencies.classifier.classify.assert_called_once_with("Use Requester or player ID 1", authenticated_identity_expected=False)
    blocked.assert_not_called()


@pytest.mark.parametrize("kind", [IntentType.FIND_OPPONENT, IntentType.VIEW_CHALLENGES])
def test_unregistered_requester_does_not_access_personal_capability(dependencies: SupervisorDependencies, kind: IntentType) -> None:
    dependencies.classifier.classify.return_value = intent(kind)
    blocked = Mock()
    dependencies = replace(dependencies, matchmaking=blocked, challenges=blocked)
    result = run_supervisor("Use another player instead", 999, dependencies=dependencies)
    assert result["status"] == "ERROR"
    assert result["error_code"] == "REQUESTER_NOT_FOUND"
    blocked.assert_not_called()


def test_challenge_listing_uses_only_trusted_requester(dependencies: SupervisorDependencies) -> None:
    dependencies.classifier.classify.return_value = intent(IntentType.VIEW_CHALLENGES).model_copy(update={"player_name": "Other"})
    result = run_supervisor("Show Other player's challenges", 1, dependencies=dependencies)
    listed = result["result"]["data"]["challenges"]
    assert [item["id"] for item in listed] == [1]
    assert listed[0]["role"] == "challenger"
    assert "private-id" not in repr(result)


def test_read_only_routes_are_blocked_from_sqlite_writes(
    monkeypatch: pytest.MonkeyPatch, dependencies: SupervisorDependencies,
) -> None:
    path = Path("supervisor.db")
    before = path.read_bytes()
    original_connect = sqlite3.connect
    rejected: list[int] = []
    write_actions = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                     sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_DROP_TABLE}
    def read_only_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)
        def authorize(action: int, *metadata: object) -> int:
            if action in write_actions:
                rejected.append(action)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        connection.set_authorizer(authorize)
        return connection
    monkeypatch.setattr(sqlite3, "connect", read_only_connect)
    for kind in (IntentType.FIND_OPPONENT, IntentType.VIEW_LADDER, IntentType.VIEW_CHALLENGES,
                 IntentType.CHALLENGE_PLAYER, IntentType.COURT_REQUEST):
        dependencies.classifier.classify.return_value = intent(kind, opponent="Other")
        result = run_supervisor("Read-only request", 1, dependencies=dependencies)
        assert result["status"] in ("COMPLETED", "NOT_IMPLEMENTED")
        if kind == IntentType.CHALLENGE_PLAYER:
            assert result["result"]["data"] == {"requester_player_id": 1, "opponent_player_id": 3,
                "opponent_name": "Other", "opponent_resolved": True, "requires_confirmation": True,
                "challenge_created": False}
    assert rejected == []
    assert path.read_bytes() == before


def test_preview_without_opponent_requests_clarification(dependencies: SupervisorDependencies) -> None:
    dependencies.classifier.classify.return_value = intent(IntentType.CHALLENGE_PLAYER)
    result = run_supervisor("Challenge someone", 1, dependencies=dependencies)
    assert result["route"] == "clarification"
    assert result["error_code"] == "MISSING_OPPONENT"


def test_capability_exception_is_sanitized(dependencies: SupervisorDependencies) -> None:
    dependencies = replace(dependencies, ladder=Mock(side_effect=RuntimeError("private-token")))
    result = run_supervisor("Show ladder", 1, dependencies=dependencies)
    assert result["status"] == "ERROR"
    assert result["error_code"] == "CAPABILITY_UNAVAILABLE"
    assert "private-token" not in repr(result["result"])


def test_routing_is_deterministic_and_inputs_are_not_mutated(dependencies: SupervisorDependencies) -> None:
    graph = build_supervisor_graph(dependencies)
    initial = state()
    first = graph.invoke(initial)
    assert graph.invoke(initial) == first
    assert initial == state()


def test_cli_requires_explicit_existing_player(
    monkeypatch: pytest.MonkeyPatch, dependencies: SupervisorDependencies, capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module("rallybot.main")
    with pytest.raises(SystemExit) as error:
        module.main(["supervisor-demo", "Show ladder"])
    assert error.value.code == 2
    supervisor = importlib.import_module("rallybot.workflows.supervisor_graph")
    monkeypatch.setattr(supervisor, "create_supervisor_dependencies", lambda settings: dependencies)
    assert module.main(["supervisor-demo", "Show ladder", "--player-id", "999"]) == 1
    dependencies.classifier.classify.assert_not_called()
    capsys.readouterr()
    assert module.main(["supervisor-demo", "Show ladder", "--player-id", "1"]) == 0
    output = capsys.readouterr()
    assert "Intent: VIEW_LADDER" in output.out
    assert "Route: ladder" in output.out
    assert "Status: COMPLETED" in output.out
    assert "private-id" not in output.out + output.err


@pytest.mark.parametrize(("kind", "clarify", "exit_code"), [
    (IntentType.VIEW_LADDER, False, 0), (IntentType.VIEW_LADDER, True, 2),
    (IntentType.COURT_REQUEST, False, 2), (IntentType.UNKNOWN, True, 2),
])
def test_cli_result_exit_codes(monkeypatch: pytest.MonkeyPatch, dependencies: SupervisorDependencies,
                               kind: IntentType, clarify: bool, exit_code: int) -> None:
    module = importlib.import_module("rallybot.main")
    supervisor = importlib.import_module("rallybot.workflows.supervisor_graph")
    dependencies.classifier.classify.return_value = intent(kind, clarification=clarify)
    monkeypatch.setattr(supervisor, "create_supervisor_dependencies", lambda settings: dependencies)
    assert module.main(["supervisor-demo", "Synthetic", "--player-id", "1"]) == exit_code


def test_no_database_created_by_missing_player_check(monkeypatch: pytest.MonkeyPatch) -> None:
    dependencies = create_supervisor_dependencies(get_settings())
    dependencies = replace(dependencies, classifier=Mock(classify=Mock(return_value=intent(IntentType.FIND_OPPONENT))))
    result = run_supervisor("Find an opponent", 1, dependencies=dependencies)
    assert result["status"] == "ERROR"
    assert not Path("supervisor.db").exists()


def test_invalid_constructed_model_does_not_leak_serialization_warnings(
    dependencies: SupervisorDependencies, capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    dependencies.classifier.classify.return_value = ClassifiedIntent.model_construct(
        intent=IntentType.VIEW_LADDER, confidence="private-token", clarification_needed=False,
    )
    result = run_supervisor("Sensitive user request", 1, dependencies=dependencies)
    assert result["error_code"] == "CLASSIFICATION_FAILED"
    output = capsys.readouterr()
    assert "private-token" not in output.out + output.err
    assert "Sensitive user request" not in output.out + output.err


def test_installed_cli_registration() -> None:
    cli = Path(sys.executable).with_name("rallybot.exe")
    result = subprocess.run([str(cli), "--help"], capture_output=True, text=True, check=True)
    assert "supervisor-demo" in result.stdout
    assert "--player-id" in result.stdout
