"""Ollama discovery and health checks; no generation or model downloads."""

import re
from dataclasses import dataclass

import httpx

from rallybot.config import Settings


@dataclass(frozen=True)
class OllamaHealthResult:
    """Connectivity and configured-model availability, with safe diagnostics."""

    server_reachable: bool
    model_available: bool
    configured_model: str
    available_models: list[str]
    error_code: str | None
    message: str


class OllamaError(RuntimeError):
    """A classified discovery error containing no raw HTTP diagnostics."""

    def __init__(self, code: str, message: str, *, server_reachable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.server_reachable = server_reachable


def valid_model_name(name: object) -> bool:
    """Accept printable model identifiers, excluding URL credentials and controls."""
    return (
        isinstance(name, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", name) is not None
        and not name.endswith((":", "/"))
    )


def _normalized_model(name: str) -> str:
    # Registry ports belong to the prefix, not to the model's tag.
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def get_available_models(
    settings: Settings, *, transport: httpx.BaseTransport | None = None
) -> list[str]:
    """Fetch and strictly validate /api/tags, closing the HTTP client afterwards."""
    try:
        base = httpx.URL(settings.ollama_base_url)
        if (base.scheme not in ("http", "https") or not base.host
                or base.username or base.password or base.query or base.fragment):
            raise ValueError
        endpoint = base.copy_with(path=base.path.rstrip("/") + "/api/tags")
    except (httpx.InvalidURL, ValueError):
        raise OllamaError("INVALID_CONFIGURATION", "Invalid Ollama base URL.") from None

    try:
        with httpx.Client(
            timeout=settings.ollama_timeout_seconds,
            transport=transport,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            response = client.get(endpoint)
    except httpx.TimeoutException:
        raise OllamaError("REQUEST_TIMEOUT", "Ollama request timed out.") from None
    except httpx.RequestError:
        raise OllamaError("SERVER_UNAVAILABLE", "Unable to reach the Ollama server.") from None

    if not 200 <= response.status_code < 300:
        raise OllamaError("HTTP_ERROR", "Ollama returned an unsuccessful HTTP status.",
                          server_reachable=True)
    try:
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ValueError
        names: list[str] = []
        for model in payload["models"]:
            if not isinstance(model, dict) or not valid_model_name(model.get("name")):
                raise ValueError
            if model["name"] not in names:
                names.append(model["name"])
        return names
    except (ValueError, UnicodeError):
        raise OllamaError("INVALID_RESPONSE", "Ollama returned an invalid model list.",
                          server_reachable=True) from None


def check_ollama_health(
    settings: Settings, *, transport: httpx.BaseTransport | None = None
) -> OllamaHealthResult:
    """Return a predictable health result without exposing request exceptions."""
    if not valid_model_name(settings.ollama_model):
        return OllamaHealthResult(False, False, settings.ollama_model, [],
                                  "INVALID_CONFIGURATION", "Invalid configured model name.")
    try:
        names = get_available_models(settings, transport=transport)
    except OllamaError as error:
        return OllamaHealthResult(error.server_reachable, False, settings.ollama_model,
                                  [], error.code, str(error))
    available = _normalized_model(settings.ollama_model) in {
        _normalized_model(name) for name in names
    }
    return OllamaHealthResult(
        True, available, settings.ollama_model, names,
        None if available else "MODEL_MISSING",
        "Ollama server and configured model are available." if available
        else "The configured model is not installed.",
    )
