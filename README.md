# agent_rl_state_bench

<p align="center">
  <img src="assets/state-bench-banner.svg" alt="STATE-Bench: Benchmark For Enterprise Workflows" width="100%" />
</p>

<p align="center">
  <a href="https://microsoft.github.io/STATE-Bench/leaderboard/"><img src="assets/leaderboard-live-badge.svg" alt="Leaderboard Live" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
  <a href="https://opensource.microsoft.com/blog/2026/05/19/introducing-state-bench-a-benchmark-for-ai-agent-memory/"><img src="https://img.shields.io/badge/Blog-Read-blue" alt="Blog" /></a>
</p>

<p align="center">
  <a href="docs/RUN_BENCHMARK.md">Main Track</a> &nbsp;·&nbsp; <a href="docs/AGENT_LEARNING_TRACK.md">Agent Learning Track</a>
</p>

STATE-Bench evaluates AI agents on realistic, multi-step enterprise workflows across three domains: **travel**, **customer support**, and **shopping assistant**.

Each task gives the agent a task-local sandbox database, domain-specific tools, and a simulated user. To pass a task, the agent must do multi-step reasoning by gathering the right information with domain tools, applying the correct policy, taking actions to update the database to the right final state when needed, and following the required procedure in conversation.

## Overview

STATE-Bench includes 450 challenging enterprise tasks across three domains.

| Domain | Tasks | Description |
| --- | ---: | --- |
| **Travel** | 150 | Flight, hotel, and car rental bookings; cancellations, updates, fee and policy reasoning, cross-product trip planning |
| **Customer Support** | 150 | Returns, refunds, exchanges, warranty claims, cancellations, shipping issues, and order changes |
| **Shopping Assistant** | 150 | Product search, cart updates, applying promos, loyalty redemption, shipping options, and compatibility checks |

## Choose Your Benchmark Track

Start with the track that matches what you want to evaluate. Each track guide links to the setup and reference docs only when you need them.

| Goal | Start here |
| --- | --- |
| Evaluate an agent or model directly on the provided enterprise benchmark tasks | **[Main Track](docs/RUN_BENCHMARK.md)** |
| Evaluate agentic memory, skills, or prompt optimization | **[Agent Learning Track](docs/AGENT_LEARNING_TRACK.md)** |

The **Main Track** is the default benchmark path. The **Agent Learning Track** uses the same simulator, domain tools, judges, and metrics, but adds train trajectories and a retrieval hook for reusable learnings such as memories, skills, or prompt optimizations.

For run commands and flag selection, start with the relevant track guide. The shared [`run_batch` reference](docs/eval/run-batch.md) lists which flags are required for the built-in agent, custom clients, and learning-track runs.

<br/>

<p align="center">
  <img src="assets/chat_bubble_2.svg" alt="Sample task trajectory from the Travel domain" width="55%" />
  <br/>
  <em>Sample task trajectory from the Travel domain.</em>
</p>

## Metrics

STATE-Bench reports four headline metrics:

| Metric | What it measures |
| --- | --- |
| **Task Completion pass@1** | Average task completion rate across five runs per task. |
| **Task Completion pass^5** | Percentage of tasks completed successfully on all five runs. |
| **UX Score** | LLM-judged conversation quality on a 1-5 scale. |
| **Cost Per Task** | Average agent cost from user-reported dollars per task. |

## 继续实验迁移说明

这个 fork 记录了 customer-support 方向的 SFT/RL 冷启动实验。换机器继续跑时，GitHub 仓库已经包含代码、脚本、合成任务、SFT/RL parquet 数据和轻量级实验报告；模型权重、checkpoint、日志和原始 rollout 输出不放进 git，需要单独准备。

### 仓库内已包含

- 官方 STATE-Bench 代码和 customer-support 合成数据：`synthetic_data/customer_support/`
- train task trajectories：`datasets/train_task_trajectories/`
- SFT 数据：`data/sft/`
- RL 数据：`data/rl/customer_support_stage1_synthetic32/train.parquet`
- 本地 Qwen official-harness 评测脚本：`scripts/run_qwen_native_customer_support.sh`
- GRPO stage1 启动脚本：`scripts/run_qwen3_4b_customer_support_grpo_stage1.sh`
- verl AgentLoop 适配：`state_bench/rl/verl_agent_loop.py`
- reward helper：`state_bench/rl/reward.py`
- RL 数据构建和 rollout 汇总脚本：`state_bench/scripts/build_rl_task_dataset.py`、`state_bench/scripts/summarize_rl_rollouts.py`
- 实验记录：`docs/customer_support_progress_20260809.md`、`docs/rl_stage1_20step_summary_20260809.md`

### 新机器需要单独准备

- Qwen 模型权重，例如 `Qwen3-4B-Instruct-2507`
- SFT checkpoint，例如 `qwen3_4b_sft_lora_r16_a32_round123_toolsplit_clean_main8192_lr1e4_b20/global_step_100_hf_lora_merged`
- 如果继续已有 RL 实验，需要单独拷贝 RL checkpoint，例如 `global_step_20_hf_merged`
- verl 代码仓库和可用 conda 环境
- DeepSeek API 配置，不要提交到 git；可以使用本地 JSON 或环境变量

推荐的本地 API 配置文件格式：

```json
{
  "base_url": "https://your-api-endpoint/v1",
  "api_key": "your-api-key"
}
```

默认脚本会尝试读取 `/home/hj/shixi/deepseek.json`。新机器路径不同的话，可以用环境变量覆盖：

```bash
export STATE_BENCH_EVAL_CONFIG_JSON=/path/to/deepseek.json
export STATE_BENCH_EVAL_MODEL=deepseek-v4-flash
export STATE_BENCH_DEEPSEEK_THINKING=disabled
```

### 路径覆盖

训练脚本默认路径是当前机器的路径。换机器后建议运行时显式覆盖：

```bash
export VERL_ROOT=/path/to/verl
export PYTHON_BIN=/path/to/conda/env/bin/python
export MODEL_PATH=/path/to/qwen_or_sft_checkpoint
export TRAIN_FILE=$PWD/data/rl/customer_support_stage1_synthetic32/train.parquet
```

然后启动小规模 GRPO：

```bash
TOTAL_TRAINING_STEPS=20 \
SAVE_FREQ=10 \
TEST_FREQ=-1 \
TRAIN_BATCH_SIZE=4 \
PPO_MINI_BATCH_SIZE=4 \
ROLLOUT_N=2 \
bash scripts/run_qwen3_4b_customer_support_grpo_stage1.sh
```

本地 vLLM official-harness 评测时，注意 Qwen tool-calling adapter 的单轮输出预算由 `OPENAI_COMPAT_AGENT_MAX_TOKENS` 控制，例如：

```bash
OPENAI_COMPAT_AGENT_BASE_URLS="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1" \
OPENAI_COMPAT_AGENT_MODEL="Qwen3-4B-Instruct-2507" \
OPENAI_COMPAT_AGENT_MAX_TOKENS=1024 \
STATE_BENCH_EVAL_RPM_LIMIT=0 \
SPLIT=test \
NUM_RUNS=1 \
NUM_WORKERS=2 \
bash scripts/run_qwen_native_customer_support.sh
```

### 不进 Git 的内容

- `checkpoints/`：模型 checkpoint 和 merged HF 权重
- `outputs*/`：完整轨迹、per-task metrics、rollout 输出
- `logs/`：训练和 serving 日志
- API key、本地 provider 配置、任何 `.env`

这些文件如果需要复现具体实验，应通过 `rsync`、`scp` 或共享存储单独迁移。

## License

STATE-Bench is released under the MIT License. See [LICENSE](LICENSE).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## Disclosures

Datasets provided in this benchmark were synthetically generated using large language models. The benchmark is intended for research purposes and users should exercise caution and consider the limitations of synthetic data when interpreting results.
