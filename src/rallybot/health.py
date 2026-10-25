"""Read-only application health coordination with injectable component checks."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from rallybot.config import Settings, get_settings
from rallybot.database.connection import check_database_health
from rallybot.llm.ollama_client import OllamaHealthResult, check_ollama_health
from rallybot.logging_config import JsonFormatter


class HealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class ComponentStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ComponentHealth:
    status: ComponentStatus
    message: str


@dataclass(frozen=True)
class ApplicationHealth:
    app_name: str
    overall_status: HealthStatus
    configuration: ComponentHealth
    database: ComponentHealth
    ollama_server: ComponentHealth
    ollama_model: ComponentHealth
    logging: ComponentHealth

    @property
    def exit_code(self) -> int:
        return {HealthStatus.HEALTHY: 0, HealthStatus.UNHEALTHY: 1,
                HealthStatus.DEGRADED: 2}[self.overall_status]


def logging_is_configured() -> bool:
    """Inspect the application handler without mutating the logger or emitting logs."""
    logger = logging.getLogger("rallybot")
    return not logger.disabled and not logger.propagate and any(
        handler.name == "rallybot.stderr"
        and isinstance(handler, logging.StreamHandler)
        and isinstance(handler.formatter, JsonFormatter)
        and not getattr(handler.stream, "closed", False)
        for handler in logger.handlers
    )


def _check(checker: Callable[[], bool], success: str, failure: str) -> ComponentHealth:
    try:
        passed = checker() is True
    except Exception:
        passed = False
    return ComponentHealth(ComponentStatus.PASS if passed else ComponentStatus.FAIL,
                           success if passed else failure)


def _check_ollama(
    checker: Callable[[Settings], OllamaHealthResult], settings: Settings
) -> tuple[ComponentHealth, ComponentHealth]:
    try:
        result = checker(settings)
        server = ComponentHealth(
            ComponentStatus.PASS if result.server_reachable else ComponentStatus.FAIL,
            "Server reachable." if result.server_reachable else "Server unavailable or request timed out.",
        )
        if result.server_reachable and result.model_available and result.error_code is None:
            return server, ComponentHealth(ComponentStatus.PASS, "Configured model available.")
        message = ("Configured model not installed." if result.error_code == "MODEL_MISSING"
                   else "Configured model availability could not be verified.")
        return server, ComponentHealth(ComponentStatus.FAIL, message)
    except Exception:
        return (ComponentHealth(ComponentStatus.FAIL, "Ollama check failed."),
                ComponentHealth(ComponentStatus.SKIPPED, "Model availability not checked."))


def check_health(
    *,
    settings_loader: Callable[[], Settings] | None = None,
    database_checker: Callable[[str], bool] | None = None,
    ollama_checker: Callable[[Settings], OllamaHealthResult] | None = None,
    logging_checker: Callable[[], bool] | None = None,
    logging_configurer: Callable[[str], None] | None = None,
) -> ApplicationHealth:
    """Coordinate checks safely; optional logging setup runs once after validation."""
    try:
        settings = (settings_loader or get_settings)()
    except Exception:
        skipped = ComponentHealth(ComponentStatus.SKIPPED, "Configuration must be valid first.")
        return ApplicationHealth(
            "RallyBot", HealthStatus.UNHEALTHY,
            ComponentHealth(ComponentStatus.FAIL, "Application configuration is invalid."),
            skipped, skipped, skipped, skipped,
        )

    configuration = ComponentHealth(ComponentStatus.PASS, "Settings loaded and validated.")
    logging_setup_failed = False
    if logging_configurer is not None:
        try:
            logging_configurer(settings.log_level)
        except Exception:
            logging_setup_failed = True
    logging_health = _check(
        lambda: not logging_setup_failed and (logging_checker or logging_is_configured)(),
        "Application logger configured.", "Application logger is not configured correctly.",
    )
    database = _check(
        lambda: (database_checker or check_database_health)(settings.database_url),
        "Database connection available.", "Database connection unavailable.",
    )
    server, model = _check_ollama(ollama_checker or check_ollama_health, settings)
    if database.status != ComponentStatus.PASS or logging_health.status != ComponentStatus.PASS:
        overall = HealthStatus.UNHEALTHY
    elif server.status != ComponentStatus.PASS or model.status != ComponentStatus.PASS:
        overall = HealthStatus.DEGRADED
    else:
        overall = HealthStatus.HEALTHY
    return ApplicationHealth(settings.app_name, overall, configuration, database,
                             server, model, logging_health)
