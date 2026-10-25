"""Database-backed workflow and injected-tool regression tests."""

import importlib
import json
import logging
import os
import socket
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

from rallybot.config import Settings, get_settings
from rallybot.database.connection import initialize_database
from rallybot.ladder.repository import PlayerRepository
from rallybot.logging_config import configure_logging
from rallybot.tools.player_tools import PlayerSummary, PlayerToolset, RankedPlayer, UnknownPlayerError, create_player_tools, filter_opponents
from rallybot.workflows import matchmaking_graph as workflow


@pytest.fixture(autouse=True)
def isolated_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields or name.startswith(("LANGCHAIN_", "LANGSMITH_")):
            monkeypatch.delenv(name)
    for name in ("LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
        monkeypatch.setenv(name, "false")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///workflow.db")
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network access is forbidden")
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


@pytest.fixture
def repository() -> PlayerRepository:
    initialize_database(get_settings().database_url)
    return PlayerRepository(get_settings().database_url)


@pytest.fixture
def tools(repository: PlayerRepository) -> PlayerToolset:
    for name, points in (("Requesting Player", 1000), ("Chris Lee", 1000), ("Sam", 1100),
                         ("Jordan", 900), ("Alex", 1200), ("Taylor", 800)):
        repository.create_player(name, points)
    return create_player_tools(repository)


def initial_state(player_id: int = 1) -> workflow.MatchmakingState:
    return {"player_id": player_id, "player_name": "untrusted", "player_points": 0,
            "available_players": [], "recommended_opponent": None,
            "recommended_opponent_id": None, "status": "STARTED"}


@pytest.mark.parametrize("values", [{}, {"player_name": "Name"}, {"player_id": None},
    {"player_id": 0}, {"player_id": -1}, {"player_id": True}, {"player_id": "1"},
    {"player_id": 1.5}, {"player_id": 2**63}])
def test_state_validation(values: dict[str, object], tools: PlayerToolset) -> None:
    with pytest.raises(ValueError):
        workflow.build_matchmaking_graph(tools).invoke(values)


def test_load_player_uses_database_values_and_clears_stale_results(tools: PlayerToolset) -> None:
    state = initial_state()
    state["recommended_opponent"] = "stale"
    state["recommended_opponent_id"] = 999
    original = deepcopy(state)
    update = workflow.load_player(state, tools=tools)
    assert update == {"player_name": "Requesting Player", "player_points": 1000,
                      "available_players": [], "recommended_opponent": None,
                      "recommended_opponent_id": None, "explanation": None,
                      "explanation_source": None, "status": "PLAYER_LOADED"}
    assert state == original


def test_graph_compilation_fixed_edges_and_execution(tools: PlayerToolset) -> None:
    graph = workflow.build_matchmaking_graph(tools)
    assert {(edge.source, edge.target) for edge in graph.get_graph().edges} == {
        ("__start__", "load_player"), ("load_player", "find_opponents"),
        ("find_opponents", "recommend_match"), ("recommend_match", "explain_recommendation"),
        ("explain_recommendation", "__end__"),
    }
    assert graph.checkpointer is None
    state = initial_state()
    result = graph.invoke(state)
    assert result["recommended_opponent"] == "Chris Lee"
    assert result["recommended_opponent_id"] == 2
    assert result["status"] == "MATCH_RECOMMENDED"
    assert state == initial_state()


def test_node_order_and_state_updates(tools: PlayerToolset) -> None:
    updates = list(workflow.build_matchmaking_graph(tools).stream(initial_state(), stream_mode="updates"))
    assert [next(iter(update)) for update in updates] == ["load_player", "find_opponents", "recommend_match", "explain_recommendation"]
    assert updates[0]["load_player"]["status"] == "PLAYER_LOADED"
    assert updates[1]["find_opponents"]["status"] == "OPPONENTS_FOUND"
    assert updates[2]["recommend_match"]["status"] == "MATCH_RECOMMENDED"


def test_inclusive_range_and_duplicate_names(repository: PlayerRepository) -> None:
    requester = repository.create_player("Same Name", 1000)
    for name, points in (("Same Name", 1000), ("Low", 800), ("High", 1200), ("Outside", 799), ("Outside", 1201)):
        repository.create_player(name, points)
    result = workflow.run_matchmaking(requester.id, tools=create_player_tools(repository))
    assert {p["id"] for p in result["available_players"]} == {2, 3, 4}
    assert result["recommended_opponent_id"] == 2
    assert result["recommended_opponent"] == "Same Name"


def test_tie_breaking_uses_existing_rank_not_alphabet(repository: PlayerRepository) -> None:
    requester = repository.create_player("Requester", 1000)
    low = repository.create_player("Alphabetically First", 900)
    high = repository.create_player("Zed", 1100)
    second_high = repository.create_player("Aaron", 1100)
    tools = create_player_tools(repository)
    for _ in range(2):
        result = workflow.run_matchmaking(requester.id, tools=tools)
        assert [p["id"] for p in result["available_players"]] == [high.id, second_high.id, low.id]
        assert result["recommended_opponent_id"] == high.id


def test_no_eligible_opponents(repository: PlayerRepository) -> None:
    requester = repository.create_player("Requester", 3000)
    repository.create_player("Too Far", 1000)
    result = workflow.run_matchmaking(requester.id, tools=create_player_tools(repository))
    assert result["available_players"] == []
    assert result["recommended_opponent"] is None
    assert result["recommended_opponent_id"] is None
    assert result["status"] == "NO_OPPONENTS"


def test_empty_opponent_list(repository: PlayerRepository) -> None:
    player = repository.create_player("Only Player")
    assert workflow.run_matchmaking(player.id, tools=create_player_tools(repository))["status"] == "NO_OPPONENTS"


def test_recommend_node_has_no_database_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sqlite3, "connect", Mock(side_effect=AssertionError("No DB in recommendation")))
    assert workflow.recommend_match(initial_state()) == {
        "recommended_opponent": None, "recommended_opponent_id": None, "status": "NO_OPPONENTS",
    }


def test_graph_with_injected_tools_needs_no_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sqlite3, "connect", Mock(side_effect=AssertionError("Injected tools must avoid DB")))
    get_player = Mock(return_value=PlayerSummary(10, "Injected", 1000))
    list_opponents = Mock(return_value=[RankedPlayer(20, "Opponent", 1050, 1)])
    filter_tool = Mock(wraps=filter_opponents)
    result = workflow.run_matchmaking(10, tools=PlayerToolset(get_player, list_opponents, filter_tool))
    assert result["recommended_opponent_id"] == 20
    get_player.assert_called_once_with(10)
    list_opponents.assert_called_once_with(10)
    filter_tool.assert_called_once()


def test_repeated_execution_is_independent(tools: PlayerToolset) -> None:
    first = workflow.run_matchmaking(1, tools=tools)
    second = workflow.run_matchmaking(1, tools=tools)
    assert first == second
    first["available_players"].clear()
    assert len(second["available_players"]) == 5


def test_lifecycle_logs_omit_inputs(tools: PlayerToolset, capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    workflow.run_matchmaking(1, tools=tools)
    output = capsys.readouterr()
    assert [json.loads(line)["message"] for line in output.err.splitlines()] == [
        "Matchmaking workflow started", "Matchmaking node executing: load_player",
        "Matchmaking node executing: find_opponents", "Matchmaking node executing: recommend_match",
        "Matchmaking node executing: explain_recommendation", "Matchmaking workflow completed",
    ]
    assert "Requesting Player" not in output.err


def test_cli_demo_and_startup(tools: PlayerToolset, repository: PlayerRepository, capsys: pytest.CaptureFixture[str]) -> None:
    before = repository.list_players()
    module = importlib.import_module("rallybot.main")
    assert module.main(["matchmaking-demo"]) == 0
    output = capsys.readouterr()
    assert "Requesting player: Requesting Player [ID 1] (1000 points)" in output.out
    assert "Recommended opponent: Chris Lee [ID 2]" in output.out
    assert "Workflow status: MATCH_RECOMMENDED" in output.out
    assert repository.list_players() == before
    assert module.main([]) == 0
    assert capsys.readouterr().out == "RallyBot initialized successfully!\n"


def test_cli_empty_database_setup_message(repository: PlayerRepository, capsys: pytest.CaptureFixture[str]) -> None:
    module = importlib.import_module("rallybot.main")
    assert module.main(["matchmaking-demo"]) == 1
    assert "No players found" in capsys.readouterr().out
    assert repository.list_players() == []


def test_cli_missing_database_not_created(capsys: pytest.CaptureFixture[str]) -> None:
    module = importlib.import_module("rallybot.main")
    assert module.main(["matchmaking-demo"]) == 1
    assert "Run db-init" in capsys.readouterr().err
    assert not Path("workflow.db").exists()


def test_cli_failure_is_safe(monkeypatch: pytest.MonkeyPatch, tools: PlayerToolset, capsys: pytest.CaptureFixture[str]) -> None:
    module = importlib.import_module("rallybot.main")
    monkeypatch.setattr(workflow, "find_opponents", Mock(side_effect=RuntimeError("secret-password")))
    assert module.main(["matchmaking-demo"]) == 1
    output = capsys.readouterr()
    assert "temporarily unavailable" in output.err
    assert "Matchmaking workflow failed" in output.err
    assert "secret-password" not in output.err
    assert "Traceback" not in output.err


def test_unknown_player_graph_result(tools: PlayerToolset) -> None:
    with pytest.raises(UnknownPlayerError, match="not registered"):
        workflow.run_matchmaking(999, tools=tools)


def test_cli_no_eligible_opponent_is_success(repository: PlayerRepository, capsys: pytest.CaptureFixture[str]) -> None:
    repository.create_player("Only Player")
    module = importlib.import_module("rallybot.main")
    assert module.main(["matchmaking-demo"]) == 0
    output = capsys.readouterr()
    assert "Recommended opponent: None" in output.out
    assert "Workflow status: NO_OPPONENTS" in output.out


def test_cli_registration() -> None:
    executable = Path(sys.executable).with_name("rallybot.exe")
    result = subprocess.run([str(executable), "--help"], capture_output=True, text=True, check=True)
    for command in ("matchmaking-demo", "ladder-seed", "discord-start", "health", "ollama-check", "db-init", "db-check"):
        assert command in result.stdout
