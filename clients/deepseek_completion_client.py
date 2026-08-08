import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from state_bench.client import BaseLLMClient


class DeepSeekCompletionClient(BaseLLMClient):
    """OpenAI-compatible client used by DeepSeekToolCallingAgent."""

    _rate_lock = threading.Lock()
    _request_times: list[float] = []

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._client = OpenAI(base_url=base_url, api_key=api_key, http_client=httpx.Client(trust_env=False))

    @classmethod
    def from_env(cls) -> "DeepSeekCompletionClient":
        config = _load_json_config(Path(os.environ.get("DEEPSEEK_CONFIG_JSON", "/home/hj/shixi/deepseek.json")))
        return cls(
            base_url=os.environ.get("DEEPSEEK_AGENT_BASE_URL") or config.get("base_url", ""),
            api_key=os.environ.get("DEEPSEEK_AGENT_API_KEY") or config.get("api_key", ""),
            model=os.environ.get("DEEPSEEK_AGENT_MODEL", "deepseek-v4-flash"),
            temperature=_parse_optional_float(os.environ.get("DEEPSEEK_AGENT_TEMPERATURE")),
            top_p=_parse_optional_float(os.environ.get("DEEPSEEK_AGENT_TOP_P")),
            max_tokens=int(os.environ.get("DEEPSEEK_AGENT_MAX_TOKENS", "4096")),
        )

    @property
    def provider_name(self) -> str:
        return "deepseek"

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
            "extra_body": {"thinking": {"type": os.environ.get("DEEPSEEK_AGENT_THINKING", "disabled")}},
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        self._throttle()
        return self._client.chat.completions.create(**kwargs)

    def _throttle(self) -> None:
        rpm = int(os.environ.get("DEEPSEEK_AGENT_RPM_LIMIT", "0"))
        if rpm <= 0:
            return
        while True:
            with self._rate_lock:
                now = time.monotonic()
                self._request_times[:] = [ts for ts in self._request_times if now - ts < 60]
                if len(self._request_times) < rpm:
                    self._request_times.append(now)
                    return
                sleep_for = max(0.1, 60 - (now - self._request_times[0]))
            time.sleep(sleep_for)


def _load_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


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
