import os
import threading
from typing import Any

import httpx
from openai import OpenAI

from state_bench.client import BaseLLMClient


class VLLMCompletionClient(BaseLLMClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ):
        base_urls = [item.strip() for item in base_url.split(",") if item.strip()]
        if not base_urls:
            raise ValueError("STATE_BENCH_VLLM_BASE_URL must include at least one URL")
        self.clients = [
            OpenAI(base_url=url, api_key=api_key, http_client=httpx.Client(trust_env=False)) for url in base_urls
        ]
        self._in_flight = [0] * len(self.clients)
        self._pool_lock = threading.Lock()
        self._next_client_idx = 0
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    @classmethod
    def from_env(cls):
        return cls(
            base_url=os.environ.get("STATE_BENCH_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("STATE_BENCH_VLLM_API_KEY", "EMPTY"),
            model=os.environ.get("STATE_BENCH_VLLM_MODEL", "Qwen/Qwen3-4B"),
            temperature=float(os.environ.get("STATE_BENCH_VLLM_TEMPERATURE", "0.0")),
            top_p=float(os.environ.get("STATE_BENCH_VLLM_TOP_P", "1.0")),
            max_tokens=int(os.environ.get("STATE_BENCH_VLLM_MAX_TOKENS", "2048")),
        )

    @property
    def provider_name(self) -> str:
        return "vllm"

    @property
    def model_name(self) -> str:
        return self.model

    def _acquire(self) -> tuple[int, OpenAI]:
        with self._pool_lock:
            min_load = min(self._in_flight)
            candidates = {i for i, load in enumerate(self._in_flight) if load == min_load}
            for offset in range(len(self.clients)):
                idx = (self._next_client_idx + offset) % len(self.clients)
                if idx in candidates:
                    self._next_client_idx = (idx + 1) % len(self.clients)
                    break
            self._in_flight[idx] += 1
            return idx, self.clients[idx]

    def _release(self, idx: int) -> None:
        with self._pool_lock:
            self._in_flight[idx] -= 1

    def complete(self, prompt: str) -> tuple[str, Any]:
        idx, client = self._acquire()
        try:
            response = client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].text or "", getattr(response, "usage", None)
        finally:
            self._release(idx)


class VLLMChatClient(VLLMCompletionClient):
    def complete(self, prompt: str) -> tuple[str, Any]:
        idx, client = self._acquire()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or "", getattr(response, "usage", None)
        finally:
            self._release(idx)
