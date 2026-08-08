import os
import threading
from itertools import count
from typing import Any

import httpx
from openai import OpenAI

from state_bench.client import BaseLLMClient


class OpenAICompatibleToolClient(BaseLLMClient):
    """OpenAI-compatible Chat Completions client with native tool calls."""

    def __init__(
        self,
        *,
        base_urls: list[str],
        api_key: str,
        model: str,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        min_p: float | None,
        max_tokens: int,
    ):
        if not base_urls:
            raise ValueError("OPENAI_COMPAT_AGENT_BASE_URLS must include at least one URL")
        self.base_urls = base_urls
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self._clients = [
            OpenAI(base_url=base_url, api_key=api_key, http_client=httpx.Client(trust_env=False))
            for base_url in base_urls
        ]
        self._counter = count()
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "OpenAICompatibleToolClient":
        raw_base_urls = os.environ.get("OPENAI_COMPAT_AGENT_BASE_URLS") or os.environ.get(
            "OPENAI_COMPAT_AGENT_BASE_URL", "http://127.0.0.1:8000/v1"
        )
        return cls(
            base_urls=[url.strip() for url in raw_base_urls.split(",") if url.strip()],
            api_key=os.environ.get("OPENAI_COMPAT_AGENT_API_KEY", "EMPTY"),
            model=os.environ.get("OPENAI_COMPAT_AGENT_MODEL", "Qwen3-4B-Instruct-2507"),
            temperature=_parse_optional_float(os.environ.get("OPENAI_COMPAT_AGENT_TEMPERATURE")),
            top_p=_parse_optional_float(os.environ.get("OPENAI_COMPAT_AGENT_TOP_P")),
            top_k=_parse_optional_int(os.environ.get("OPENAI_COMPAT_AGENT_TOP_K")),
            min_p=_parse_optional_float(os.environ.get("OPENAI_COMPAT_AGENT_MIN_P")),
            max_tokens=int(os.environ.get("OPENAI_COMPAT_AGENT_MAX_TOKENS", "4096")),
        )

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self.model

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [_to_chat_completion_tool(tool) for tool in tools],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        extra_body: dict[str, Any] = {}
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if extra_body:
            kwargs["extra_body"] = extra_body

        return self._next_client().chat.completions.create(**kwargs)

    def _next_client(self) -> OpenAI:
        with self._lock:
            idx = next(self._counter) % len(self._clients)
        return self._clients[idx]


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _to_chat_completion_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if "function" in tool:
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }
