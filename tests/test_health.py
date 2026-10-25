"""Deterministic unified health checks with injected infrastructure dependencies."""

import importlib
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from rallybot.config import Settings, get_settings
from rallybot.database.connection import check_database_health
from rallybot.health import ComponentStatus, HealthStatus, check_health, logging_is_configured
from rallybot.llm.ollama_client import OllamaHealthResult
from rallybot.logging_config import configure_logging


def ollama_result(
    reachable: bool = True, available: bool = True, code: str | None = None
) -> OllamaHealthResult:
    return OllamaHealthResult(reachable, available, "llama3.2", [], code,
                              "untrusted diagnostic: never-print-this")


def component_failure(*args: object) -> Any:
    raise RuntimeError("password=never-print-this at a private filesystem path")


@pytest.fixture(autouse=True)
def isolated_health(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields:
            monkeypatch.delenv(name)
    def forbid_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Real HTTP requests are forbidden in health tests")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_network)
    logger = logging.getLogger("rallybot")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    get_settings.cache_clear()
    try:
        yield
    finally:
        for handler in list(logger.handlers):
            if handler not in handlers:
                logger.removeHandler(handler)
                handler.close()
        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
        get_settings.cache_clear()


@pytest.fixture
def dependencies() -> dict[str, Any]:
    return {
        "settings_loader": lambda: Settings(_env_file=None),
        "database_checker": lambda url: True,
        "ollama_checker": lambda settings: ollama_result(),
        "logging_checker": lambda: True,
    }


def test_all_components_healthy(dependencies: dict[str, Any]) -> None:
    result = check_health(**dependencies)
    assert result.app_name == "RallyBot"
    assert result.overall_status == HealthStatus.HEALTHY
    assert result.exit_code == 0
    for component in (result.configuration, result.database, result.ollama_server,
                      result.ollama_model, result.logging):
        assert component.status == ComponentStatus.PASS


@pytest.mark.parametrize(("reachable", "code"), [
    (False, "SERVER_UNAVAILABLE"), (False, "REQUEST_TIMEOUT"),
    (True, "MODEL_MISSING"), (True, "HTTP_ERROR"), (True, "INVALID_RESPONSE"),
])
def test_ollama_problems_are_degraded(
    dependencies: dict[str, Any], reachable: bool, code: str
) -> None:
    dependencies["ollama_checker"] = lambda settings: ollama_result(reachable, False, code)
    result = check_health(**dependencies)
    assert result.overall_status == HealthStatus.DEGRADED
    assert result.exit_code == 2
    assert result.ollama_server.status == (ComponentStatus.PASS if reachable else ComponentStatus.FAIL)
    assert result.ollama_model.status == ComponentStatus.FAIL
    if code == "MODEL_MISSING":
        assert result.ollama_model.message == "Configured model not installed."
    assert "never-print-this" not in repr(result)


@pytest.mark.parametrize("ai_available", [True, False])
def test_database_failure_is_unhealthy(dependencies: dict[str, Any], ai_available: bool) -> None:
    dependencies["database_checker"] = lambda url: False
    dependencies["ollama_checker"] = lambda settings: ollama_result(ai_available, ai_available)
    result = check_health(**dependencies)
    assert result.database.status == ComponentStatus.FAIL
    assert result.overall_status == HealthStatus.UNHEALTHY
    assert result.exit_code == 1


def test_invalid_configuration_skips_dependencies(dependencies: dict[str, Any]) -> None:
    def unexpected(*args: object) -> bool:
        pytest.fail("Component must not be called after configuration failure")
    result = check_health(
        settings_loader=lambda: Settings(app_env="never-print-this"),
        database_checker=unexpected, ollama_checker=unexpected,
        logging_checker=unexpected, logging_configurer=unexpected,
    )
    assert result.overall_status == HealthStatus.UNHEALTHY
    assert result.configuration.status == ComponentStatus.FAIL
    assert "never-print-this" not in repr(result)
    for component in (result.database, result.ollama_server, result.ollama_model, result.logging):
        assert component.status == ComponentStatus.SKIPPED


def test_logging_not_configured(dependencies: dict[str, Any]) -> None:
    logger = logging.getLogger("rallybot")
    original_handlers = list(logger.handlers)
    for handler in original_handlers:
        logger.removeHandler(handler)
    try:
        dependencies.pop("logging_checker")
        result = check_health(**dependencies)
        assert result.logging.status == ComponentStatus.FAIL
        assert result.overall_status == HealthStatus.UNHEALTHY
        assert not logger.handlers
    finally:
        for handler in original_handlers:
            logger.addHandler(handler)


def test_logging_check_does_not_duplicate_handlers(dependencies: dict[str, Any]) -> None:
    configure_logging("INFO")
    dependencies.pop("logging_checker")
    before = list(logging.getLogger("rallybot").handlers)
    for _ in range(2):
        assert check_health(**dependencies).logging.status == ComponentStatus.PASS
    assert logging.getLogger("rallybot").handlers == before
    assert logging_is_configured()


def test_optional_logging_setup_runs_once(dependencies: dict[str, Any]) -> None:
    levels: list[str] = []
    dependencies["logging_configurer"] = levels.append
    assert check_health(**dependencies).overall_status == HealthStatus.HEALTHY
    assert levels == ["INFO"]


@pytest.mark.parametrize("dependency", [
    "settings_loader", "database_checker", "ollama_checker", "logging_checker", "logging_configurer",
])
def test_component_exceptions_are_safe(dependencies: dict[str, Any], dependency: str) -> None:
    dependencies[dependency] = component_failure
    result = check_health(**dependencies)
    assert result.overall_status == (
        HealthStatus.DEGRADED if dependency == "ollama_checker" else HealthStatus.UNHEALTHY
    )
    assert "never-print-this" not in repr(result)
    assert "private filesystem" not in repr(result)


def test_no_unintended_database_creation(dependencies: dict[str, Any]) -> None:
    dependencies["database_checker"] = check_database_health
    result = check_health(**dependencies)
    assert result.overall_status == HealthStatus.UNHEALTHY
    assert not Path("data").exists()
    assert not list(Path.cwd().rglob("*.db"))


@pytest.mark.parametrize(("database_ok", "ai_ok", "exit_code"), [
    (True, True, 0), (False, True, 1), (True, False, 2), (False, False, 1),
])
def test_cli_status_and_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    dependencies: dict[str, Any], database_ok: bool, ai_ok: bool, exit_code: int,
) -> None:
    module = importlib.import_module("rallybot.main")
    dependencies["database_checker"] = lambda url: database_ok
    dependencies["ollama_checker"] = lambda settings: ollama_result(ai_ok, ai_ok)
    dependencies.pop("logging_checker")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "never-print-this")
    monkeypatch.setenv("DATABASE_URL", "scheme://user:never-print-this@host/db")
    monkeypatch.setattr(module, "check_health", lambda **kwargs: check_health(**dependencies, **kwargs))
    assert module.main(["health"]) == exit_code
    captured = capsys.readouterr()
    for label in ("Application:", "Overall health:", "Configuration:", "Database:",
                  "Ollama server:", "Configured model:", "Logging:"):
        assert label in captured.out
    assert "never-print-this" not in captured.out + captured.err
    assert "scheme://" not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err
    assert sum(handler.name == "rallybot.stderr"
               for handler in logging.getLogger("rallybot").handlers) == 1


def test_cli_invalid_configuration_is_safe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = importlib.import_module("rallybot.main")
    monkeypatch.setenv("APP_ENV", "password=never-print-this")
    assert module.main(["health"]) == 1
    output = capsys.readouterr()
    assert "Overall health: UNHEALTHY" in output.out
    assert "Configuration: FAIL" in output.out
    assert "Database: SKIPPED" in output.out
    assert "never-print-this" not in output.out + output.err
    assert "Traceback" not in output.out + output.err


def test_existing_cli_commands_remain_functional(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = importlib.import_module("rallybot.main")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./temporary/test.db")
    monkeypatch.setattr(module, "check_ollama_health", lambda settings: ollama_result())
    for args in ([], ["db-init"], ["db-check"], ["ollama-check"]):
        assert module.main(args) == 0
    output = capsys.readouterr()
    assert "RallyBot initialized successfully!" in output.out
    assert "Database initialized successfully." in output.out
    assert "Database health check passed." in output.out
    assert "Configured model available: yes" in output.out
