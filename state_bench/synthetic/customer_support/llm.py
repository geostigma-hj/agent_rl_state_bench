"""Small OpenAI-compatible JSON clients for synthetic data generation and judging."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIStatusError, OpenAI


def load_provider_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return json.load(f)


class OpenAICompatibleJsonClient:
    """Minimal chat-completions JSON client.

    The class intentionally reads credentials from env/config paths only and never
    serializes credentials into generated artifacts.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 2048,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body or {}
        self._client = OpenAI(base_url=base_url, api_key=api_key, http_client=httpx.Client(trust_env=False))

    @classmethod
    def deepseek_v4_flash(cls) -> "OpenAICompatibleJsonClient":
        config = load_provider_config(os.environ.get("SYNTH_DEEPSEEK_CONFIG_JSON", "/home/hj/shixi/deepseek.json"))
        return cls(
            base_url=os.environ.get("SYNTH_DEEPSEEK_BASE_URL") or config.get("base_url", ""),
            api_key=os.environ.get("SYNTH_DEEPSEEK_API_KEY") or config.get("api_key", ""),
            model=os.environ.get("SYNTH_DEEPSEEK_MODEL", "deepseek-v4-flash"),
            temperature=_optional_float(os.environ.get("SYNTH_DEEPSEEK_TEMPERATURE")),
            max_tokens=int(os.environ.get("SYNTH_DEEPSEEK_MAX_TOKENS", "2048")),
            extra_body={"thinking": {"type": os.environ.get("SYNTH_DEEPSEEK_THINKING", "disabled")}},
        )

    @classmethod
    def step37_flash(cls) -> "OpenAICompatibleJsonClient":
        config = load_provider_config(os.environ.get("SYNTH_STEP_CONFIG_JSON", "/home/hj/shixi/step.json"))
        return cls(
            base_url=os.environ.get("SYNTH_STEP_BASE_URL") or config.get("base_url", ""),
            api_key=os.environ.get("SYNTH_STEP_API_KEY") or config.get("api_key", ""),
            model=os.environ.get("SYNTH_STEP_MODEL", "step-3.7-flash"),
            temperature=_optional_float(os.environ.get("SYNTH_STEP_TEMPERATURE")),
            max_tokens=int(os.environ.get("SYNTH_STEP_MAX_TOKENS", "2048")),
            reasoning_effort=os.environ.get("SYNTH_STEP_REASONING_EFFORT", "high"),
        )

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code != 429 or attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable retry state") from last_error


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
