"""Bounded ChatOllama adapter: capability-selected native tools or strict JSON plans."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from typing import Annotated, Protocol

import httpx
import ollama
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field

from rallybot.config import Settings
from rallybot.llm.ollama_client import valid_model_name


class SelectionMode(StrEnum):
    NATIVE_TOOL_CALLING = "NATIVE_TOOL_CALLING"
    STRUCTURED_FALLBACK = "STRUCTURED_FALLBACK"


class RequestedTool(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    calls: list[RequestedTool] = Field(max_length=10)


class FinalSelection(BaseModel):
    """Only existing evidence IDs, never unrestricted prose or new facts."""
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    result_ids: list[Annotated[int, Field(ge=1)]] = Field(min_length=1, max_length=10)


class ToolCallingModel(Protocol):
    mode: SelectionMode

    def select(self, messages: list[BaseMessage], tools: Sequence[StructuredTool]) -> object: ...
    def finalize(self, messages: list[BaseMessage]) -> object: ...


def tool_plan_schema(tools: Sequence[StructuredTool]) -> dict[str, object]:
    """Guide generation with exact per-tool schemas; executor validation stays authoritative."""
    return {"type": "object", "properties": {"calls": {"type": "array", "maxItems": 10, "items": {"oneOf": [
        {"type": "object", "properties": {"name": {"const": tool.name, "type": "string"},
         "arguments": tool.args_schema.model_json_schema()}, "required": ["name", "arguments"],
         "additionalProperties": False} for tool in tools
    ]}}}, "required": ["calls"], "additionalProperties": False}


def parse_structured(response: object, schema: type[BaseModel]) -> BaseModel:
    if isinstance(response, schema):
        return schema.model_validate(response.model_dump(warnings=False))
    if not isinstance(response, AIMessage) or response.tool_calls or response.invalid_tool_calls:
        raise ValueError("INVALID_MODEL_RESPONSE")
    if not isinstance(response.content, str) or not 0 < len(response.content) <= 16384:
        raise ValueError("INVALID_MODEL_RESPONSE")
    return schema.model_validate_json(response.content)


def model_error_code(error: Exception) -> str:
    if isinstance(error, (httpx.TimeoutException, TimeoutError)):
        return "OLLAMA_TIMEOUT"
    if isinstance(error, (httpx.TransportError, ConnectionError)):
        return "OLLAMA_CONNECTION_ERROR"
    if isinstance(error, ollama.ResponseError):
        return "OLLAMA_MODEL_ERROR"
    if isinstance(error, httpx.HTTPStatusError):
        return "OLLAMA_HTTP_ERROR"
    if isinstance(error, ValueError):
        return "OLLAMA_INVALID_CONFIGURATION_OR_RESPONSE"
    return "OLLAMA_UNEXPECTED_ERROR"


class OllamaToolModel:
    def __init__(self, model: ChatOllama, mode: SelectionMode) -> None:
        self.model, self.mode = model, mode

    def select(self, messages: list[BaseMessage], tools: Sequence[StructuredTool]) -> object:
        if self.mode == SelectionMode.NATIVE_TOOL_CALLING:
            return self.model.bind_tools(tools).invoke(messages)
        return self.model.bind(format=tool_plan_schema(tools)).invoke(messages)

    def finalize(self, messages: list[BaseMessage]) -> object:
        return self.model.bind(format=FinalSelection.model_json_schema()).invoke(messages)


@contextmanager
def open_tool_model(settings: Settings) -> Iterator[OllamaToolModel]:
    base = httpx.URL(settings.ollama_base_url)
    if (base.scheme not in ("http", "https") or not base.host or base.username or base.password
            or base.query or base.fragment or not valid_model_name(settings.ollama_model)):
        raise ValueError("INVALID_CONFIGURATION")
    endpoint = base.copy_with(path=base.path.rstrip("/") + "/api/show")
    with httpx.Client(timeout=min(settings.ollama_timeout_seconds, 30), trust_env=False, follow_redirects=False) as client:
        response = client.post(str(endpoint), json={"model": settings.ollama_model})
        if response.status_code == 404:
            raise ollama.ResponseError("Configured model unavailable", status_code=404)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("INVALID_MODEL_CAPABILITIES")
        capabilities = payload.get("capabilities")
        if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
            raise ValueError("INVALID_MODEL_CAPABILITIES")
    mode = SelectionMode.NATIVE_TOOL_CALLING if "tools" in capabilities else SelectionMode.STRUCTURED_FALLBACK
    # Own the actual synchronous transport; no persistent global model clients.
    with httpx.HTTPTransport(trust_env=False) as transport:
        model = ChatOllama(base_url=str(base), model=settings.ollama_model, temperature=0, num_predict=512,
            client_kwargs={"timeout": settings.ollama_generation_timeout_seconds,
                           "trust_env": False, "follow_redirects": False},
            sync_client_kwargs={"transport": transport})
        yield OllamaToolModel(model, mode)
