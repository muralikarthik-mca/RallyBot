"""Tests for JSON logging, isolation, and safe failure handling."""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr

from rallybot.config import Settings, get_settings
from rallybot.logging_config import configure_logging
from rallybot.main import main


@pytest.fixture(autouse=True)
def isolated_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields:
            monkeypatch.delenv(name)
    logger = logging.getLogger("rallybot")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    for handler in handlers:
        logger.removeHandler(handler)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
        get_settings.cache_clear()


def test_json_fields_and_utc(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    before = datetime.now().astimezone()
    logging.getLogger("rallybot.logging_test").info("Ready\nfor tennis")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1
    entry = json.loads(captured.err)
    assert set(entry) == {"timestamp", "level", "logger", "message"}
    assert entry["level"] == "INFO"
    assert entry["logger"] == "rallybot.logging_test"
    assert entry["message"] == "Ready\nfor tennis"
    timestamp = datetime.fromisoformat(entry["timestamp"])
    assert timestamp.utcoffset() == timedelta(0)
    assert before <= timestamp <= datetime.now().astimezone()


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_log_levels(level: str, capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level)
    logger = logging.getLogger("rallybot")
    for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        logger.log(logging.getLevelNamesMapping()[name], "Safe event")
    entries = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    expected = [name for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
                if logging.getLevelNamesMapping()[name] >= logging.getLevelNamesMapping()[level]]
    assert [entry["level"] for entry in entries] == expected


def test_exception_type_without_credentials(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    try:
        raise ValueError("password=never-log-this")
    except ValueError:
        logging.getLogger("rallybot").exception("Operation failed")
    output = capsys.readouterr().err
    assert json.loads(output)["exception"] == {"type": "ValueError"}
    assert "never-log-this" not in output
    assert "Traceback" not in output


def test_context_allowlist(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    logging.getLogger("rallybot").info("Event", extra={
        "request_id": "request-1", "agent": "dispatcher", "workflow_id": "workflow-1",
        "password": "never-log-this", "api_key": "never-log-this",
        "settings": Settings(), "discord_bot_token": SecretStr("never-log-this"),
        "timestamp": "cannot-overwrite",
    })
    output = capsys.readouterr().err
    entry = json.loads(output)
    assert entry["request_id"] == "request-1"
    assert entry["agent"] == "dispatcher"
    assert entry["workflow_id"] == "workflow-1"
    assert entry["timestamp"] != "cannot-overwrite"
    assert "password" not in entry
    assert "never-log-this" not in output
    assert "settings" not in entry


@pytest.mark.parametrize("field", ["message", "name", "levelname", "msg", "created"])
def test_reserved_record_fields_rejected(field: str) -> None:
    configure_logging("INFO")
    with pytest.raises(KeyError):
        logging.getLogger("rallybot").info("Safe event", extra={field: "override"})


def test_context_objects_are_not_stringified(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    logging.getLogger("rallybot").info("Safe event", extra={
        "request_id": SecretStr("never-log-this"), "agent": Settings(),
    })
    entry = json.loads(capsys.readouterr().err)
    assert entry["request_id"] == "[REDACTED]"
    assert entry["agent"] == "[REDACTED]"


def test_safe_formatting_failure(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    # Exercise our handler directly: pytest also attaches its own formatters,
    # whose behavior on malformed messages is outside the application contract.
    handler = next(handler for handler in logging.getLogger("rallybot").handlers
                   if handler.name == "rallybot.stderr")
    handler.handle(logging.LogRecord(
        "rallybot", logging.INFO, __file__, 0, "%d",
        ("password=never-log-this",), None,
    ))
    output = capsys.readouterr().err
    assert json.loads(output)["message"] == "Log entry could not be formatted"
    assert "never-log-this" not in output
    assert "Logging error" not in output


def test_reconfigure_without_duplicates(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING")
    configure_logging("INFO")
    logger = logging.getLogger("rallybot")
    assert sum(handler.name == "rallybot.stderr" for handler in logger.handlers) == 1
    logger.info("Once")
    assert len(capsys.readouterr().err.splitlines()) == 1


def test_other_loggers_unchanged() -> None:
    root = logging.getLogger()
    third_party = logging.getLogger("unrelated_library")
    before = [(list(logger.handlers), logger.level, logger.propagate, logger.disabled)
              for logger in (root, third_party)]
    configure_logging("DEBUG")
    after = [(list(logger.handlers), logger.level, logger.propagate, logger.disabled)
             for logger in (root, third_party)]
    assert after == before


def test_import_has_no_logging_side_effects() -> None:
    script = """
import logging
root = logging.getLogger()
app = logging.getLogger('rallybot')
before = [(list(x.handlers), x.level, x.propagate) for x in (root, app)]
import rallybot.logging_config
assert before == [(list(x.handlers), x.level, x.propagate) for x in (root, app)]
"""
    result = subprocess.run([sys.executable, "-B", "-c", script],
                            capture_output=True, text=True, check=True)
    assert result.stdout == result.stderr == ""


def test_startup_uses_settings_without_exposing_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "never-log-this")
    monkeypatch.setenv("DATABASE_URL", "scheme://user:password@host/db")
    main([])
    assert logging.getLogger("rallybot").level == logging.ERROR
    captured = capsys.readouterr()
    assert captured.out == "RallyBot initialized successfully!\n"
    assert captured.err == ""
