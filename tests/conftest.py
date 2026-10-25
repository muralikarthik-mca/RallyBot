"""Opt-in fixtures for specialized-agent tests; existing tests are unaffected."""

import logging
import os
import socket
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest

from rallybot.config import Settings, get_settings
from rallybot.database.connection import initialize_database
from rallybot.ladder.repository import PlayerRepository


@pytest.fixture
def agent_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields or name.startswith(("LANGCHAIN_", "LANGSMITH_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///agents.db")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("Agent tests forbid network access")))
    logger = logging.getLogger("rallybot")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    get_settings.cache_clear()
    try:
        yield
    finally:
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate
        get_settings.cache_clear()


@pytest.fixture
def agent_players(agent_environment: None) -> PlayerRepository:
    initialize_database(get_settings().database_url)
    players = PlayerRepository(get_settings().database_url)
    players.create_player("Agent Requester", 1000, "private-discord-111")
    players.create_player("Sam Wilson", 1100, "private-discord-222")
    return players


@pytest.fixture
def agent_no_writes(agent_players: PlayerRepository, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    before = Path("agents.db").read_bytes()
    original = sqlite3.connect
    attempted: list[int] = []
    write_actions = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_DROP_INDEX, sqlite3.SQLITE_ATTACH}

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = original(*args, **kwargs)
        def authorize(action: int, *metadata: object) -> int:
            if action in write_actions:
                attempted.append(action)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    yield
    assert attempted == []
    assert Path("agents.db").read_bytes() == before
