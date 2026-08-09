"""verl AgentLoop adapter for STATE-Bench customer-support RL.

This module is imported only by verl training jobs. It keeps the online rollout
semantics aligned with the official STATE-Bench harness: task-local environment,
benchmark tools, V4 Flash user simulator, deterministic state reward, and
tool-quality penalties.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from uuid import uuid4

import httpx
from openai import OpenAI

from state_bench.domain import get_domain_config
from state_bench.env_loader import load_task_environment
from state_bench.rl.reward import compute_reward_v1
from state_bench.schemas import StateDiff, TaskDefinition
from state_bench.scoring import compute_efficiency, evaluate_state_requirements

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.workers.rollout.replica import TokenOutput


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


def _assistant_tool_message(content: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{idx}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                },
            }
            for idx, call in enumerate(tool_calls)
        ]
    return message


class _OpenAICompatibleSimulatorClient:
    def __init__(self) -> None:
        api_key = os.environ.get("STATE_BENCH_EVAL_API_KEY")
        base_url = os.environ.get("STATE_BENCH_EVAL_BASE_URL")
        model = os.environ.get("STATE_BENCH_EVAL_MODEL") or os.environ.get("STATE_BENCH_EVAL_DEPLOYMENTS")
        if not api_key or not base_url or not model:
            raise ValueError(
                "STATE-Bench verl simulator requires STATE_BENCH_EVAL_API_KEY, "
                "STATE_BENCH_EVAL_BASE_URL, and STATE_BENCH_EVAL_MODEL."
            )
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=httpx.Client(trust_env=False))

    def complete_chat(self, messages: list[dict[str, str]], temperature: float | None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(os.environ.get("STATE_BENCH_EVAL_MAX_TOKENS", "4096")),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        thinking = os.environ.get("STATE_BENCH_DEEPSEEK_THINKING") or os.environ.get("STATE_BENCH_EVAL_THINKING")
        if thinking:
            kwargs["extra_body"] = {"thinking": {"type": thinking}}
        return self.client.chat.completions.create(**kwargs).choices[0].message.content or ""


def _format_simulator_conversation(conversation: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in conversation:
        role = msg["role"].upper()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")
        if role == "ASSISTANT" and tool_calls:
            tc_summary = "\n".join(
                f"[Called {tc['name']}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)[:200]})]"
                for tc in tool_calls
            )
            content = f"{tc_summary}\n{content}" if content else tc_summary
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


@register("state_bench_customer_support")
class StateBenchCustomerSupportAgentLoop(AgentLoopBase):
    """Run one STATE-Bench customer-support task as a verl multi-turn rollout."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.domain = get_domain_config("customer_support")
        self.tool_schemas = [_to_chat_completion_tool(tool) for tool in self.domain.tool_schemas]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns or self.domain.max_agent_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        self.max_agent_turn_tokens = int(os.environ.get("STATE_BENCH_RL_AGENT_TURN_MAX_TOKENS", "1024"))
        self.simulator_client = _OpenAICompatibleSimulatorClient()

    async def _generate(self, request_id: str, runtime_ids: list[int], sampling_params: dict[str, Any]) -> TokenOutput:
        params = dict(sampling_params)
        if self.tool_parser.stop_token_ids:
            params["stop_token_ids"] = list(set((params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
        return await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=runtime_ids,
            sampling_params=params,
        )

    async def _tokenize_delta(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> list[int]:
        return await self.apply_chat_template(messages, tools=tools, remove_system_prompt=True)

    def _max_model_len(self) -> int:
        return int(
            self.rollout_config.max_model_len
            or (self.rollout_config.prompt_length + self.rollout_config.response_length)
        )

    def _remaining_generation_tokens(self, runtime_ids: list[int], initial_prompt_len: int) -> int:
        response_used = max(0, len(runtime_ids) - initial_prompt_len)
        response_left = int(self.rollout_config.response_length) - response_used
        context_left = self._max_model_len() - len(runtime_ids)
        return min(response_left, context_left)

    def _limit_sampling_params(
        self, sampling_params: dict[str, Any], runtime_ids: list[int], initial_prompt_len: int
    ) -> dict[str, Any]:
        params = dict(sampling_params)
        remaining = self._remaining_generation_tokens(runtime_ids, initial_prompt_len)
        current_limit = params.get("max_tokens", params.get("max_new_tokens", self.rollout_config.response_length))
        params["max_tokens"] = max(1, min(int(current_limit), remaining, self.max_agent_turn_tokens))
        params.pop("max_new_tokens", None)
        return params

    async def _simulator_respond(self, system_prompt: str, conversation: list[dict[str, Any]]) -> str:
        conversation_text = _format_simulator_conversation(conversation)
        instruction = (
            f"CONVERSATION SO FAR:\n{conversation_text}\n\n"
            "Respond as the customer based on the conversation and your rules above.\n"
            "YOUR RESPONSE:"
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}]
        temperature = os.environ.get("STATE_BENCH_SIMULATOR_TEMPERATURE")
        return (
            await self.loop.run_in_executor(
                None,
                lambda: self.simulator_client.complete_chat(
                    messages=messages,
                    temperature=float(temperature) if temperature is not None else None,
                ),
            )
        ).strip()

    def _truncate_tool_result(self, text: str) -> str:
        if len(text) <= self.max_tool_response_length:
            return text
        if self.tool_response_truncate_side == "left":
            return "(truncated)..." + text[-self.max_tool_response_length :]
        if self.tool_response_truncate_side == "right":
            return text[: self.max_tool_response_length] + "...(truncated)"
        length = self.max_tool_response_length // 2
        return text[:length] + "...(truncated)..." + text[-length:]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        t0 = time.monotonic()
        extra_info = kwargs.get("extra_info", {}) or {}
        task_path = extra_info.get("task_path")
        if not task_path:
            raise ValueError("STATE-Bench verl rollout requires extra_info.task_path")
        global_steps = kwargs.get("global_steps", 0)
        task = TaskDefinition.load(task_path)
        env_data, _ = load_task_environment(self.domain, task)
        env_copy = env_data.deep_copy()
        env = self.domain.environment_class(env_copy, now=task.now)
        db_before = env.get_full_snapshot()
        handlers = env.tool_handlers
        allowed_names = set(handlers)

        system_prompt = self.domain.agent_system_prompt.format(now=task.now, user_id=task.user_id)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.opening_message},
        ]
        conversation: list[dict[str, Any]] = [{"role": "user", "content": task.opening_message}]
        simulator_prompt = self.domain.build_simulator_prompt(task, env_data, task.user_id)

        request_id = uuid4().hex
        runtime_ids = await self.apply_chat_template(messages, tools=self.tool_schemas)
        initial_prompt_len = len(runtime_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        all_tool_calls: list[dict[str, Any]] = []
        metrics = AgentLoopMetrics()

        for turn_idx in range(self.max_assistant_turns):
            stop_reason = "max_turns"
            if self._remaining_generation_tokens(runtime_ids, initial_prompt_len) < 1:
                stop_reason = "token_budget_exhausted"
                break

            output = await self._generate(
                request_id,
                runtime_ids,
                self._limit_sampling_params(sampling_params, runtime_ids, initial_prompt_len),
            )
            assistant_ids = output.token_ids
            runtime_ids += assistant_ids
            response_mask += [1] * len(assistant_ids)
            if output.log_probs:
                response_logprobs += output.log_probs

            assistant_content, parsed_calls = await self.tool_parser.extract_tool_calls(assistant_ids, self.tool_schemas)
            requested_calls: list[dict[str, Any]] = []
            executed_calls: list[dict[str, Any]] = []
            for parsed in parsed_calls[: self.max_parallel_calls]:
                try:
                    arguments = json.loads(parsed.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                except Exception as exc:
                    arguments = {}
                    result = {"error": f"invalid tool arguments: {exc}"}
                else:
                    if parsed.name not in allowed_names:
                        result = {"error": f"disallowed tool: {parsed.name}"}
                    else:
                        result = handlers[parsed.name](arguments)
                requested_calls.append({"name": parsed.name, "arguments": arguments})
                record = {"name": parsed.name, "arguments": arguments, "result": result}
                executed_calls.append(record)
                all_tool_calls.append(record)

            conversation.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": executed_calls if executed_calls else None,
                }
            )

            if executed_calls:
                messages.append(_assistant_tool_message(assistant_content, requested_calls))
                tool_messages = []
                for call_idx, record in enumerate(executed_calls):
                    tool_content = json.dumps(record["result"], ensure_ascii=False)
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"call_{call_idx}",
                            "content": self._truncate_tool_result(tool_content),
                        }
                    )
                messages.extend(tool_messages)
                delta_ids = self.turn_separator + await self._tokenize_delta(tool_messages)
                runtime_ids += delta_ids
                response_mask += [0] * len(delta_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(delta_ids)
                continue

            if turn_idx >= self.max_assistant_turns - 1:
                stop_reason = "max_turns"
                break
            user_response = await self._simulator_respond(simulator_prompt, conversation)
            conversation.append({"role": "user", "content": user_response})
            if self.domain.check_termination and self.domain.check_termination(user_response):
                stop_reason = "simulator_termination"
                break
            user_message = {"role": "user", "content": user_response}
            messages.append(user_message)
            delta_ids = self.turn_separator + await self._tokenize_delta([user_message])
            runtime_ids += delta_ids
            response_mask += [0] * len(delta_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(delta_ids)

        db_after = env.get_full_snapshot()
        state_diff = StateDiff.compute(db_before, db_after)
        efficiency = compute_efficiency(conversation, all_tool_calls)
        state_score = evaluate_state_requirements(task, state_diff)
        reward_parts = compute_reward_v1(
            state_requirements_met=state_score.score if state_score else None,
            tool_errors=efficiency.tool_errors,
            redundant_calls=efficiency.redundant_calls,
        )

        full_response_ids = runtime_ids[initial_prompt_len:]
        response_ids = full_response_ids[: self.rollout_config.response_length]
        response_mask = response_mask[: self.rollout_config.response_length]
        if response_logprobs:
            response_logprobs = response_logprobs[: self.rollout_config.response_length]

        metrics.generate_sequences = round(time.monotonic() - t0, 4)
        metrics.tool_calls = float(efficiency.tool_calls)
        return AgentLoopOutput(
            prompt_ids=runtime_ids[:initial_prompt_len],
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
            reward_score=reward_parts["reward_v1"],
            num_turns=efficiency.turns,
            metrics=metrics,
            extra_fields={
                "task_id": task.task_id,
                "state_requirements_met": state_score.score if state_score else None,
                "state_requirements_reasoning": state_score.reasoning if state_score else None,
                "state_diff": state_diff.to_dict(),
                "tool_calls": efficiency.tool_calls,
                "tool_errors": efficiency.tool_errors,
                "redundant_calls": efficiency.redundant_calls,
                "reward_parts": reward_parts,
                "stop_reason": stop_reason,
                "runtime_token_count": len(runtime_ids),
                "response_token_count": len(full_response_ids),
                "global_steps": global_steps,
                "min_global_steps": global_steps,
                "max_global_steps": global_steps,
            },
        )
