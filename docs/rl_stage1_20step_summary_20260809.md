# RL Stage1 20-Step 小规模验证报告

日期：2026-08-10

## 训练配置

- 初始模型：`Qwen3-4B-Instruct-2507` 的 SFT LoRA step100 merged checkpoint
- 训练框架：verl GRPO
- 任务集：`data/rl/customer_support_stage1_synthetic32/train.parquet`
- GPU：`CUDA_VISIBLE_DEVICES=0,1,2,3`
- `TRAIN_BATCH_SIZE=4`
- `ROLLOUT_N=2`
- 每 step 轨迹数：8
- 总 step：20
- 总轨迹数：160
- `MAX_RESPONSE_LENGTH=8192`
- 单轮 assistant 输出预算：`STATE_BENCH_RL_AGENT_TURN_MAX_TOKENS=1024`
- 最大 assistant turn：10
- 采样参数：`temperature=0.7`，`top_p=0.8`，`top_k=20`
- checkpoint 频率：10 step

## 训练产物

- 训练日志：`logs/rl/qwen3_4b_sft_step100_lora_grpo_stage1_20step_budget_turn10_20260809_230314.log`
- Rollout：`outputs/rl_train_rollouts/qwen3_4b_sft_step100_lora_grpo_stage1_20step_budget_turn10_20260809_230314/`
- FSDP checkpoint：
  `checkpoints/customer_support/rl/qwen3_4b_sft_step100_lora_grpo_stage1_20step_budget_turn10_20260809_230314/`
- 已保存 checkpoint：`global_step_10`、`global_step_20`
- HF merged checkpoint：
  `/mnt/hj/shixi/state-bench/checkpoints/customer_support/rl/qwen3_4b_sft_step100_lora_grpo_stage1_20step_budget_turn10_20260809_230314/global_step_20_hf_merged`

## 训练过程指标

- 训练总耗时：约 1.68 小时
- 20 step 平均 rollout reward：0.3047
- 前 10 step 平均 reward：0.3150
- 后 10 step 平均 reward：0.2944
- 平均 response length：2397.8 tokens
- 平均 turns：6.17
- 平均 step time：303.2 秒
- 最好 batch：step 17，reward 0.7500，平均 response length 1220，平均 turns 3.875
- 最差末端 batch：step 20，reward -0.1562，平均 response length 4030，最大 response length 8192，平均 turns 7.5

## Test Split 评测

评测使用官方 STATE-Bench harness：

- Agent：本地 vLLM OpenAI-compatible server，4 个 server 分别部署在 GPU 0-3
- Agent 模型：`global_step_20_hf_merged`
- Agent 单次输出预算：`OPENAI_COMPAT_AGENT_MAX_TOKENS=1024`
- 主 Agent 不显式指定温度，使用模型/vLLM 默认采样参数
- Simulator/Judge：DeepSeek V4 Flash
- Simulator 温度：0.2
- DeepSeek thinking：disabled
- Split：test，50 条，pass@1
- 输出目录：`outputs/customer_support_qwen3_4b_sft_step100_rl_step20_mtok1024_native_test/`

评测结果：

| 模型 | Runs | Task Completion pass@1 | State pass@1 | Task Requirements pass@1 | Mean UX | Mean Turns | Mean Tool Calls | Mean Output Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 原始 Instruct | 4 | 31.0% | 46.0% | 48.0% | 0.0 | 6.0 | 6.5 | 1173.0 |
| SFT LoRA step100 | 4 | 41.5% | 45.0% | 61.5% | 0.0 | 5.3 | 10.4 | 1425.8 |
| RL step20 | 1 | 38.0% | 44.0% | 60.0% | 2.95 | 5.0 | 9.8 | 1092.7 |

说明：

- RL step20 只跑了 pass@1，因此和 SFT 的 4-run 均值不是完全等价对比。
- RL step20 的 completion 低于 SFT step100 的 4-run 均值，但高于原始 Instruct。
- State pass 和 Task Requirements pass 与 SFT step100 接近，没有看到明显提升。
- Mean turns 和 output tokens 比 SFT 略低，但仍存在少数 15-turn 长尾。

## 问题观察

- 训练端在 budget/turn10 配置下不再普遍打满 8k，输出长度比旧配置明显正常。
- test 评测中仍有多条样本进入长对话或反复确认，典型 15-turn 样本包括：
  `7-return_restocking_waived`、`39-hard_damaged_scale_wrong_shirt_choice_goodwill`、`63-challenge_bulk_clawback`、`109-hard_exchange_noncanonical_size`、`147-spare_compound_warranty_then_price_match`。
- 工具调用仍偏多，最高样本达到 27 次 tool calls。
- 失败集中在复杂价格/退货/多动作组合任务，说明当前 reward 对“正确完成流程”和“少走弯路”的约束还不够强。

## 初步结论

这次 20-step GRPO 证明链路可以跑通，但没有给出足够强的性能收益信号。相对 SFT step100，RL step20 的 pass@1 没有提升，且仍有工具调用过多和长尾对话问题。

下一步不建议直接扩大到长时间 RL。更稳妥的做法是先做一个更小的 reward/debug 迭代：

- 保留前三类 reward：task/state、tool format、process/tool behavior。
- 暂时不加 efficiency reward，但需要统计工具调用冗余和 15-turn 长尾作为诊断指标。
- 针对失败样本抽样看轨迹，确认是 reward 没覆盖、环境反馈诱导、还是 SFT 底座已有错误。
- 修正后再跑 20-50 step 小规模 GRPO，并优先看 test pass@1 和长尾样本是否改善。
