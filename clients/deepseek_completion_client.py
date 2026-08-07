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
    """OpenAI-compatible completion client for using DeepSeek as a JSON agent."""

    _rate_lock = threading.Lock()
    _request_times: list[float] = []

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
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
            temperature=float(os.environ.get("DEEPSEEK_AGENT_TEMPERATURE", "0.2")),
            top_p=_parse_optional_float(os.environ.get("DEEPSEEK_AGENT_TOP_P")),
            max_tokens=int(os.environ.get("DEEPSEEK_AGENT_MAX_TOKENS", "4096")),
        )

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def model_name(self) -> str:
        return self.model

    def complete(self, prompt: str) -> tuple[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {"thinking": {"type": os.environ.get("DEEPSEEK_AGENT_THINKING", "disabled")}},
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        self._throttle()
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or "", getattr(response, "usage", None)

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
