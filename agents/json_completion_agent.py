import json
from typing import Any

from state_bench.agents.base import AgentToolCallRequest, AgentTurnResponse, BaseAgent


class JsonCompletionAgent(BaseAgent):
    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **_kwargs):
        super().__init__(runtime_context=runtime_context)
        self.client = client

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        prompt = self._build_prompt(system_prompt, conversation, tools)
        text, usage = self.client.complete(prompt)
        self._record_usage(usage)
        payload = self._parse_json(text)
        tool_calls = [
            AgentToolCallRequest(name=str(item.get("name", "")), arguments=item.get("arguments") or {})
            for item in payload.get("tool_calls", [])
            if isinstance(item, dict)
        ]
        return AgentTurnResponse(text=str(payload.get("response", "") or ""), tool_calls=tool_calls)

    def _record_usage(self, usage: Any) -> None:
        if not usage:
            return
        self.add_token_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )

    def _build_prompt(
        self,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> str:
        compact_tools = [
            {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
            }
            for tool in tools
        ]
        transcript = "\n".join(self._format_message(msg) for msg in conversation)
        return f"""{system_prompt}

You are operating inside STATE-Bench. You may either answer the user or request benchmark tools.
Return exactly one JSON object and no other text.

JSON schema:
{{
  "response": "assistant message to the user; use an empty string when only calling tools",
  "tool_calls": [
    {{"name": "tool_name", "arguments": {{"argument_name": "argument_value"}}}}
  ]
}}

Rules:
- Use only tool names listed below.
- If you need information or need to change the environment, return tool_calls.
- For write tools, first get the relevant policy, then preview with confirm=false or omitted, ask the user for confirmation, and only then call confirm=true.
- If no tool is needed, return "tool_calls": [].

Available tools:
{json.dumps(compact_tools, ensure_ascii=False)}

Conversation:
{transcript}

JSON:
"""

    def _format_message(self, msg: dict[str, Any]) -> str:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "assistant" and msg.get("tool_calls"):
            return (
                f"assistant: {content}\n"
                f"assistant_tool_calls_and_results: {json.dumps(msg['tool_calls'], ensure_ascii=False)}"
            )
        if role == "tool":
            return f"tool_results: {json.dumps(content, ensure_ascii=False)}"
        return f"{role}: {content}"

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = self._strip_code_fences(text.strip())
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            json_text = self._first_json_object(cleaned)
            if not json_text:
                return {"response": cleaned, "tool_calls": []}
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                return {"response": cleaned, "tool_calls": []}
        if not isinstance(data, dict):
            return {"response": str(data), "tool_calls": []}
        if not isinstance(data.get("tool_calls", []), list):
            data["tool_calls"] = []
        return data

    def _strip_code_fences(self, text: str) -> str:
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text

    def _first_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None
