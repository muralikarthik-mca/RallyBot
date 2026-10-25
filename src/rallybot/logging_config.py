"""Explicit JSON logging configuration scoped to the RallyBot namespace."""

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Format safe context and exception types without exception text or stacks."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a single JSON object, with a safe fallback on formatting errors."""
        try:
            entry: dict[str, object] = {
                "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            for field in ("request_id", "agent", "workflow_id"):
                if field in record.__dict__:
                    value = record.__dict__[field]
                    # Never call str/repr on arbitrary context objects or Settings.
                    entry[field] = (
                        value if type(value) in (str, int, float, bool, type(None))
                        else "[REDACTED]"
                    )
            if record.exc_info and record.exc_info[0] is not None:
                entry["exception"] = {"type": record.exc_info[0].__name__}
            return json.dumps(entry, allow_nan=False)
        except Exception:
            # Do not include the failed record, its arguments, or the error text.
            return json.dumps({
                "timestamp": datetime.now(UTC).isoformat(),
                "level": "ERROR",
                "logger": "rallybot",
                "message": "Log entry could not be formatted",
            })


def configure_logging(log_level: str) -> None:
    """Configure RallyBot's stderr handler without modifying other loggers."""
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("Unsupported log level")

    logger = logging.getLogger("rallybot")
    logger.setLevel(log_level)
    logger.propagate = False
    for handler in list(logger.handlers):
        if handler.name == "rallybot.stderr":
            logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(sys.stderr)
    handler.name = "rallybot.stderr"
    handler.setLevel(log_level)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
