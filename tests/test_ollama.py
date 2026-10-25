"""Ollama connectivity tests; every HTTP request uses an in-process mock."""

import importlib
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from rallybot.config import Settings, get_settings
from rallybot.llm.ollama_client import check_ollama_health, get_available_models


@pytest.fixture(autouse=True)
def isolated_ollama(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    fields = {name.casefold() for name in Settings.model_fields}
    for name in list(os.environ):
        if name.casefold() in fields:
            monkeypatch.delenv(name)
    def forbid_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Real HTTP requests are forbidden in Ollama tests")
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


def response_transport(payload: object, status: int = 200) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/tags"
        return httpx.Response(status, json=payload)
    return httpx.MockTransport(respond)


def test_successful_connectivity_and_discovery() -> None:
    payload = {"models": [{"name": "llama3.2:latest"}, {"name": "other:7b"}]}
    names = get_available_models(Settings(), transport=response_transport(payload))
    assert names == ["llama3.2:latest", "other:7b"]
    result = check_ollama_health(Settings(), transport=response_transport(payload))
    assert result.server_reachable is True
    assert result.model_available is True
    assert result.configured_model == "llama3.2"
    assert result.available_models == names
    assert result.error_code is None


@pytest.mark.parametrize(("configured", "installed", "available"), [
    ("llama3.2", "llama3.2:latest", True),
    ("llama3.2:latest", "llama3.2", True),
    ("llama3.2:3b", "llama3.2:3b", True),
    ("llama3.2:3b", "llama3.2:latest", False),
    ("llama3.2", "llama3.2:3b", False),
    ("llama3.2:3b", "llama3.2:1b", False),
    ("registry:5000/library/model", "registry:5000/library/model:latest", True),
    ("llama3.2", "other:latest", False),
])
def test_model_tag_matching(configured: str, installed: str, available: bool) -> None:
    result = check_ollama_health(
        Settings(ollama_model=configured),
        transport=response_transport({"models": [{"name": installed}]}),
    )
    assert result.server_reachable is True
    assert result.model_available is available
    assert result.error_code == (None if available else "MODEL_MISSING")


def test_empty_and_duplicate_model_lists() -> None:
    result = check_ollama_health(Settings(), transport=response_transport({"models": []}))
    assert result.error_code == "MODEL_MISSING"
    assert result.available_models == []
    names = get_available_models(Settings(), transport=response_transport({
        "models": [{"name": "llama3.2"}, {"name": "llama3.2"}]
    }))
    assert names == ["llama3.2"]


@pytest.mark.parametrize(("error_type", "code"), [
    (httpx.ConnectError, "SERVER_UNAVAILABLE"),
    (httpx.ReadError, "SERVER_UNAVAILABLE"),
    (httpx.ConnectTimeout, "REQUEST_TIMEOUT"),
    (httpx.ReadTimeout, "REQUEST_TIMEOUT"),
])
def test_network_error_classification(error_type: type[httpx.RequestError], code: str) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise error_type("credentials=never-print-this", request=request)
    result = check_ollama_health(Settings(), transport=httpx.MockTransport(fail))
    assert result.server_reachable is False
    assert result.model_available is False
    assert result.error_code == code
    assert "never-print-this" not in result.message


@pytest.mark.parametrize("status", [301, 401, 500])
def test_http_errors(status: int) -> None:
    result = check_ollama_health(Settings(), transport=response_transport(
        {"error": "credentials=never-print-this"}, status,
    ))
    assert result.server_reachable is True
    assert result.error_code == "HTTP_ERROR"
    assert result.available_models == []
    assert "never-print-this" not in result.message


def test_malformed_json() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"not JSON"))
    result = check_ollama_health(Settings(), transport=transport)
    assert result.server_reachable is True
    assert result.error_code == "INVALID_RESPONSE"


@pytest.mark.parametrize("payload", [
    {}, [], {"models": None}, {"models": {}}, {"models": [None]},
    {"models": [{}]}, {"models": [{"name": 42}]}, {"models": [{"name": ""}]},
    {"models": [{"name": "llama3.2"}, {"name": "invalid\nname"}]},
    {"models": [{"name": "https://user:password@host"}]},
])
def test_invalid_model_entries_fail_closed(payload: object) -> None:
    result = check_ollama_health(Settings(), transport=response_transport(payload))
    assert result.server_reachable is True
    assert result.model_available is False
    assert result.error_code == "INVALID_RESPONSE"
    assert result.available_models == []


def test_configured_endpoint_timeout_and_client_closure() -> None:
    class TrackingTransport(httpx.MockTransport):
        closed = False

        def close(self) -> None:
            self.closed = True
            super().close()

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://example.test:1234/ollama/api/tags"
        assert request.method == "GET"
        assert request.extensions["timeout"] == {
            "connect": 2.5, "read": 2.5, "write": 2.5, "pool": 2.5,
        }
        return httpx.Response(200, json={"models": []})
    transport = TrackingTransport(respond)
    settings = Settings(ollama_base_url="http://example.test:1234/ollama/",
                        ollama_timeout_seconds=2.5)
    get_available_models(settings, transport=transport)
    assert transport.closed


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_timeouts(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(ollama_timeout_seconds=value)


def test_timeout_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "2.5")
    assert Settings().ollama_timeout_seconds == 2.5


def test_invalid_configured_model_is_not_printed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = importlib.import_module("rallybot.main")
    monkeypatch.setenv("OLLAMA_MODEL", "https://user:secret-password@host")
    assert module.main(["ollama-check"]) == 1
    output = capsys.readouterr()
    assert "[invalid model name]" in output.out
    assert "secret-password" not in output.out + output.err


@pytest.mark.parametrize("url", [
    "file:///private", "http://user:secret@host", "http://host/?token=secret",
    "http://host/#secret", "not a URL",
])
def test_invalid_base_url(url: str) -> None:
    result = check_ollama_health(Settings(ollama_base_url=url))
    assert result.error_code == "INVALID_CONFIGURATION"
    assert url not in result.message


@pytest.mark.parametrize(("payload", "status", "exit_code"), [
    ({"models": [{"name": "llama3.2:latest"}]}, 200, 0),
    ({"models": [{"name": "other:latest"}]}, 200, 1),
    ({"error": "never-print-this"}, 500, 1),
])
def test_cli_exit_codes_and_safe_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    payload: object, status: int, exit_code: int,
) -> None:
    module = importlib.import_module("rallybot.main")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "never-print-this")
    monkeypatch.setattr(module, "check_ollama_health", lambda settings: check_ollama_health(
        settings, transport=response_transport(payload, status),
    ))
    assert module.main(["ollama-check"]) == exit_code
    captured = capsys.readouterr()
    assert "Configured model: llama3.2" in captured.out
    assert "Installed models:" in captured.out
    if exit_code == 0:
        assert "Configured model available: yes" in captured.out
        assert "llama3.2:latest" in captured.out
    assert "never-print-this" not in captured.out + captured.err
    assert "http://" not in captured.out + captured.err
    assert json.loads(captured.err)["level"] == ("INFO" if exit_code == 0 else "ERROR")


def test_cli_timeout(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = importlib.import_module("rallybot.main")
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("password=never-print-this", request=request)
    monkeypatch.setattr(module, "check_ollama_health", lambda settings: check_ollama_health(
        settings, transport=httpx.MockTransport(fail),
    ))
    assert module.main(["ollama-check"]) == 1
    output = capsys.readouterr()
    assert "Ollama server reachable: no" in output.out
    assert "REQUEST_TIMEOUT" in output.err
    assert "never-print-this" not in output.out + output.err
