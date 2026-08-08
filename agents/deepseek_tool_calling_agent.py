import json
from typing import Any

from state_bench.agents.base import AgentToolCallRequest, AgentTurnResponse, BaseAgent


class DeepSeekToolCallingAgent(BaseAgent):
    """DeepSeek Chat Completions agent using native tool calls."""

    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **_kwargs):
        super().__init__(runtime_context=runtime_context)
        self.client = client
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self._pending_tool_call_ids: list[str] = []
        self._pending_tool_call_names: list[str] = []

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        _ = system_prompt
        self._append_latest_harness_item(conversation[-1])
        response = self.client.complete_with_tools(messages=self._messages, tools=tools)
        message = response.choices[0].message
        usage = getattr(response, "usage", None)
        self.add_token_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            cached_input_tokens=getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None),
        )

        tool_calls = list(getattr(message, "tool_calls", None) or [])
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ]
        self._messages.append(assistant_message)
        self._pending_tool_call_ids = [tc.id for tc in tool_calls]
        self._pending_tool_call_names = [tc.function.name for tc in tool_calls]

        return AgentTurnResponse(
            text=message.content or "",
            tool_calls=[
                AgentToolCallRequest(name=tc.function.name, arguments=_parse_arguments(tc.function.arguments))
                for tc in tool_calls
            ],
        )

    def _append_latest_harness_item(self, item: dict[str, Any]) -> None:
        role = item.get("role")
        if role == "tool":
            for idx, call in enumerate(item.get("content") or []):
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": self._pending_tool_call_ids[idx],
                        "name": self._pending_tool_call_names[idx],
                        "content": json.dumps(call.get("result"), ensure_ascii=False),
                    }
                )
            self._pending_tool_call_ids = []
            self._pending_tool_call_names = []
            return

        if role in {"user", "assistant", "system"}:
            self._messages.append({"role": role, "content": str(item.get("content") or "")})


def _parse_arguments(arguments: str | None) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
