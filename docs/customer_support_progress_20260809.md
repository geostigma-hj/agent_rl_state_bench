# Customer Support Agent RL 进展记录

日期：2026-08-09

## 基线决策

- 本项目用于本地训练和评测的主 Agent 模型：`Qwen3-4B-Instruct-2507`。
- 合成数据和 SFT 轨迹采样的主要 Teacher/Simulator：DeepSeek V4 Flash。
- Simulator 温度：`0.2`。
- 主 Agent 的 baseline 解码参数默认遵循模型/服务端默认设置，除非明确测试 thinking 变体。

## 官方 Train Split 基线结果

- 当前选定本地模型为不开 thinking 的 `Qwen3-4B-Instruct-2507`；thinking 虽然可能改变表现，但延迟过高，不适合作为当前阶段默认配置。
- DeepSeek V4 Flash 和 Pro 均已使用官方 STATE-Bench harness 评测。
- 最终模型选择基于延迟和质量权衡：可训练目标使用 `Qwen3-4B-Instruct-2507`，不把 DeepSeek Pro/Flash 作为训练目标模型。

详细原始输出保留在 `outputs/`。

## 合成数据状态

- 合成 customer_support 任务数：590。
- 任务类型分布：
  - `cancel_order`: 60
  - `compound`: 60
  - `edge_case`: 60
  - `exchange_item`: 80
  - `price_match_refund`: 90
  - `return_item`: 65
  - `shipping_claim`: 110
  - `warranty_claim`: 65
- 全量人工审查：590/590 PASS。
- V4 Flash judge：590/590 ACCEPT。
- 确定性 checker：590/590 通过，0 失败，120 warnings。
- 模板残留检查：0 个剩余问题。
- 修复日志：854 处字段/文本级修复。

重要审查产物：

- `synthetic_data/customer_support/full_manual_audit/summary.json`
- `synthetic_data/customer_support/full_manual_audit/full_audit_report.md`
- `synthetic_data/customer_support/full_manual_audit/repair_log.json`
- `synthetic_data/customer_support/full_manual_audit/reviews/`

## 指标字段完备性

构造数据保留了官方 STATE-Bench 评测所需字段：

- 状态指标：`tasks/*.json::state_requirements` 加 rollout 里的 `state_diff`。
- 流程/轨迹指标：`tasks/*.json::task_requirements` 加 rollout 里的 `conversation` 和 `tool_calls`。
- 工具/写操作指标：rollout 里的 `conversation.tool_calls`；customer_support 的写工具在 domain config 中声明。

有 120 条模板派生的 compound/edge 任务中，`metadata.gold_tool_plan` 和 `metadata.expected_state_diff` 为 null。这不影响官方评测，因为官方 scoring 使用的是 task requirements 和 rollout diff，而不是这些 metadata 辅助字段。

## Teacher Rollout 与 SFT 数据

已用官方 harness 采集 V4 Flash teacher rollout：

- Round 1：441/590 accepted。
- Round 2：438/590 accepted。
- Round 3：对前两轮均失败的任务做 targeted retry，新增 21 条 accepted。
- 合并后 SFT 轨迹数：900 条 accepted trajectories。

第一版 SFT 转换已废弃，原因是 assistant 最终回复文本和 tool call 被打包进同一条 assistant message。清洗后的转换现在会把 assistant tool-call message 和最终自然语言回复分开保存，并在 parquet 读回后丢弃 null-expanded 的 tool arguments。

可复现性说明：当前本地 verl checkout `/home/hj/shixi/verl` 额外 patch 了 `verl/utils/dataset/multiturn_sft_dataset.py`，用于读取 parquet 后清理 null-expanded 的嵌套 tool-call arguments。该 checkout 不在本仓库内，后续复现需要保留同样本地 patch，或使用等价的上游/data-loader 修复。

清洗后的 SFT 数据：

- 冷启动主 split：`data/sft/customer_support_v4_flash_teacher_round123_targeted_toolsplit_clean_main8192`
- Train/val 行数：849/45。
- 清洗后超过 8192 tokens 的长尾：train 6 条，val 0 条。

## 冷启动 SFT

先做过一次 full-parameter SFT，基座为 `/mnt/models/Qwen3-4B-Instruct-2507`，训练框架为 verl FSDP，不是 LoRA：

- GPU：6 x RTX 4090。
- 最大序列长度：8192。
- Train batch size：24。
- 学习率：`1e-5`。
- 总 step：100。
- 保存/验证间隔：25 step。
- Attention：`flash_attention_2`。
- checkpoint 保存内容：model 和 extra state。

验证 loss：

- Step 25：0.4191。
- Step 50：0.3944。
- Step 75：0.3990。
- Step 100：0.4163。

Step 50 的验证 loss 最低，因此用于评测。合并后的 HF checkpoint：

`checkpoints/customer_support/qwen3_4b_sft_cold_start_round123_toolsplit_clean_main8192_full_flash_lr1e5_sp2_b24/global_step_50_hf_merged`

训练产物：

- Log：`logs/sft/qwen3_4b_instruct_2507_v4_flash_teacher_round123_toolsplit_clean_main8192_full_flash_lr1e5_sp2_b24_20260809_062924.log`
- Metrics CSV：`logs/sft/qwen3_4b_instruct_2507_v4_flash_teacher_round123_toolsplit_clean_main8192_full_flash_lr1e5_sp2_b24_20260809_062924_metrics.csv`

## Full SFT 测试集评测

Step 50 checkpoint 已用官方 STATE-Bench customer_support test split 和官方 harness 评测：

- Agent：`OpenAICompatibleToolCallingAgent`
- Agent model label：`Qwen3-4B-SFT-clean-step50`
- 本地服务：vLLM，GPU 0-4 上 5 个 endpoint。
- vLLM context：12288 tokens。
- 主跑 Agent max completion cap：1024 tokens。
- Overflow 补跑：任务 `72-challenge_bulk_plus_shipping_clawback` 使用 max completion cap 256，以适配本地 vLLM 的 12288-token context。
- Simulator/Judge：通过 `STATE_BENCH_EVAL_PROVIDER=deepseek` 使用 DeepSeek V4 Flash。
- Simulator 温度：`0.2`。
- 主 Agent 温度：本地服务/模型默认行为，未显式覆盖。
- UX scoring：跳过。

50/50 test 任务结果：

- `task_completion_pass@1`：24%。
- 通过任务：12/50。
- 失败任务：38/50。
- 总工具调用：503。
- Tool errors：112。

结果路径：

`outputs/customer_support_qwen3_4b_sft_clean_step50_main8192_lr1e5_12k_mtok1024_native_test_20260809_1239`

按任务名粗略估算的类别通过率：

- `cancel_order`：1/2。
- `edge_case`：3/3。
- `exchange_item`：1/7。
- `price_match_refund`：0/6。
- `return_item`：2/8。
- `shipping_claim`：3/14。
- `warranty_claim`：1/3。
- 其他 challenge 任务：1/7。

直接观察：full SFT 能产生可工作的 tool calls，但测试集质量较低。最明显的问题是工具调用过多或非法，50 条任务里有 112 个 tool errors。Price match、exchange、compound、多动作 challenge 是后续数据/RL 最需要针对的弱项。

## LoRA SFT 消融

由于 full-parameter SFT 在官方 test split 上低于 base，因此追加了 LoRA SFT 实验。

训练配置：

- Base model：`/mnt/models/Qwen3-4B-Instruct-2507`。
- 数据：`data/sft/customer_support_v4_flash_teacher_round123_targeted_toolsplit_clean_main8192`。
- Train/val 行数：849/45。
- GPU：5 x RTX 4090，verl FSDP 训练时每个 rank 一个模型 shard。
- LoRA rank/alpha：`r=16`，`alpha=32`。
- 最大序列长度：8192。
- Train batch size：20。
- 学习率：`1e-4`。
- 总 step：100，约等于在 849 条训练样本上跑 2.3 个 epoch。
- 保存/验证间隔：25 step。

验证 loss：

- Step 25：0.5781。
- Step 50：0.5065。
- Step 75：0.4687。
- Step 100：0.4501。

Step 100 时验证 loss 仍在下降，因此 LoRA 训练从 loss 看还没有明确收敛。为了快速获得下游信号，先评测 step100。

LoRA 产物说明：

- verl 的 FSDP merger 会产出完整 HF model 加单独的 `lora_adapter` 目录。
- 为避免 vLLM 评测时误 serve 未加载 adapter 的 base model，评测前使用 PEFT `merge_and_unload` 将 adapter 显式合入主权重。
- 合并后的推理 checkpoint：`checkpoints/customer_support/qwen3_4b_sft_lora_r16_a32_round123_toolsplit_clean_main8192_lr1e4_b20/global_step_100_hf_lora_merged`。

官方 harness test50 结果：

- 输出：`outputs/customer_support_qwen3_4b_sft_lora_r16_step100_main8192_lr1e4_12k_mtok1024_native_test_20260809_1343`。
- Agent model label：`Qwen3-4B-SFT-LoRA-r16-step100`。
- 本地服务：vLLM，GPU 0-4 上 5 个 endpoint。
- vLLM context：12288 tokens。
- 主跑 Agent max completion cap：1024 tokens。
- Overflow 补跑：任务 `41-hard_two_lost_shipments_threshold_split` 和 `71-hard_exchange_late_compensation_destination_choice` 使用 max completion cap 256，以适配本地 vLLM 的 12288-token context。
- Simulator/Judge：DeepSeek V4 Flash。
- Simulator 温度：`0.2`。
- UX scoring：跳过。

50/50 test 任务结果：

- `task_completion_pass@1`：46%。
- 通过任务：23/50。
- 失败任务：27/50。
- 总工具调用：547。
- Tool errors：151。
- 平均 turns：5.5。

按任务名粗略估算的类别通过率：

- `cancel_order`：2/4。
- `edge_case`：3/3。
- `exchange_item`：3/7。
- `price_match_refund`：1/5。
- `return_item`：5/8。
- `shipping_claim`：3/12。
- `warranty_claim`：2/3。
- 其他 challenge 任务：4/8。

当前官方 customer_support test50 对比：

- Base Qwen3-4B-Instruct-2507：30%。
- Full-parameter SFT step50：24%。
- LoRA SFT step100：46%。

直接观察：在当前小数据 regime 下，LoRA 明显优于 base 和 full-parameter SFT，即使验证 loss 还没有收敛。主要残留问题仍是工具调用精度和流程合规：LoRA 通过任务更多，但在 hard multi-step cases 上仍产生大量 tool errors。

## 4 次重复稳定性评测

为了降低单次采样的偶然性，对官方 test50 复用已有 run1，并追加 run2-run4，统计 4 次重复结果。这里的 `pass^4` 采用官方 harness 口径，即同一 task 在 4 次 run 中全部成功才记为通过。

评测设置：

- Split：官方 `test`，50 条任务。
- Simulator/Judge：DeepSeek V4 Flash。
- Simulator 温度：`0.2`。
- Agent：本地 vLLM 原生 tool-calling。
- Agent sampling：使用模型默认 generation config。
- 主跑 Agent max completion cap：1024 tokens；极少数上下文超限样本用 256 tokens 单独补跑。

结果：

| 模型 | Run1 | Run2 | Run3 | Run4 | 4-run 平均 Pass@1 | Pass^4 | Pass-any^4 | 输出目录 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base Qwen3-4B-Instruct-2507 | 15/50 | 17/50 | 16/50 | 14/50 | 31.0% | 8% | 64% | `outputs/customer_support_qwen3_4b_instruct_2507_native_tools_test50_deepseek_v4_flash` |
| LoRA step100 | 23/50 | 23/50 | 16/50 | 21/50 | 41.5% | 22% | 62% | `outputs/customer_support_qwen3_4b_sft_lora_r16_step100_main8192_lr1e4_12k_mtok1024_native_test_20260809_1343` |

落盘文件：

- Base 官方指标：`outputs/customer_support_qwen3_4b_instruct_2507_native_tools_test50_deepseek_v4_flash/metrics_pass4.json`。
- Base 手工汇总：`outputs/customer_support_qwen3_4b_instruct_2507_native_tools_test50_deepseek_v4_flash/manual_pass4_summary.json`。
- LoRA 官方指标：`outputs/customer_support_qwen3_4b_sft_lora_r16_step100_main8192_lr1e4_12k_mtok1024_native_test_20260809_1343/metrics_pass4.json`。
- LoRA 手工汇总：`outputs/customer_support_qwen3_4b_sft_lora_r16_step100_main8192_lr1e4_12k_mtok1024_native_test_20260809_1343/manual_pass4_summary.json`。

结论：重复评测后，LoRA step100 仍显著优于未 SFT 的 base。单轮 LoRA run3 掉到 32%，说明 test50 单次 run 的方差确实不小；但 4-run 平均仍从 base 的 31.0% 提升到 41.5%，严格 `pass^4` 从 8% 提升到 22%。这比单次 46% vs 30% 更稳健，支持继续把 LoRA step100 作为 cold-start policy。

## LoRA 继续训练到 200 Step

为了验证“验证 loss 继续下降”是否能转化为官方 harness 性能提升，将 LoRA 从 step100 继续训练到 step200。

实现说明：SFT 脚本原本依赖 verl 默认 `trainer.total_epochs=4`，导致 `TOTAL_TRAINING_STEPS=200` 的续训会在 epoch 4 结束时提前停止，无法到达 step200。脚本现在暴露 `TOTAL_EPOCHS`；续训使用 `TOTAL_EPOCHS=6`，并从 `global_step_150` resume 后跑到 step200。

续训验证 loss：

- Step 100：0.4501。
- Step 125：0.4383。
- Step 150：0.4316。
- Step 175：0.4353。
- Step 200：0.4318。

官方 harness test50 结果：

| Checkpoint | Val loss | Pass@1 | 通过数 | 工具调用 | Tool errors | 平均 turns | 输出目录 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LoRA step100 | 0.4501 | 46% | 23/50 | 547 | 151 | 5.5 | `outputs/customer_support_qwen3_4b_sft_lora_r16_step100_main8192_lr1e4_12k_mtok1024_native_test_20260809_1343` |
| LoRA step150 | 0.4316 | 38% | 19/50 | 501 | 112 | 5.1 | `outputs/customer_support_qwen3_4b_sft_lora_r16_step150_main8192_lr1e4_12k_mtok1024_native_test_20260809_1437` |
| LoRA step200 | 0.4318 | 36% | 18/50 | 442 | 89 | 5.2 | `outputs/customer_support_qwen3_4b_sft_lora_r16_step200_main8192_lr1e4_12k_mtok1024_native_test_20260809_1449` |

结论：step100 之后，验证 loss 对该 benchmark 的指示性变差。继续 SFT 会减少 tool errors，但任务完成率同步下降。可能的失败模式是行为过拟合或 policy drift，模型变得更保守/更偏离有用的探索式工具调用。当前最佳 cold-start checkpoint 仍是 LoRA `global_step_100`，不是 step150 或 step200。

## 下一步

1. 使用 LoRA `global_step_100` 作为当前 cold-start policy。
2. 不再只按验证 loss 选择 SFT checkpoint；需要结合官方 harness 任务成功率，或建立 held-out task validation set。
3. 分析 LoRA step100 的失败轨迹，按 tool error 类型和任务类别定位问题，重点看 shipping、price match、exchange、compound。
4. RL 从 LoRA step100 初始化，先跑小规模 sanity run，再启动全数据训练。

## RL Rollout 配置修正

首次 20-step GRPO sanity 发现 step1 开始 response length 即异常偏长，后续 step20 全部打满 8192。排查后确认主要不是 8K 总预算本身，而是 RL agent loop 在追加 simulator user message 时重复传入 `tools=self.tool_schemas`，触发 Qwen chat template 每轮重新注入完整工具 schema，导致 rollout response side 被多次工具说明污染。

已修正：

- 工具 schema 只在初始 prompt 注入，后续 user delta 不再重复传 tools。
- Rollout sampling 显式对齐 Qwen generation config：temperature 0.7、top_p 0.8、top_k 20。
- 增加单 assistant turn cap：`STATE_BENCH_RL_AGENT_TURN_MAX_TOKENS=1024`，整条 rollout 仍可保留 `MAX_RESPONSE_LENGTH=8192`。
- 暂时关闭 redundant-call penalty，避免当前阶段混入 efficiency reward；下一轮 RL reward 先保留 final state / process / tool 三类。

1-step sanity 对比：

| 配置 | response length mean | max | clip ratio | output char mean | 工具 schema 重复 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 loop + 对齐 sampling/turn cap，但仍重复 tools | 6308.9 | 8192 | 0.25 | 25195 | 3-9 次/轨迹 |
| 修复 tools 重复后 | 1643.3 | 2130 | 0.125 | 6008.6 | 0 |

结论：旧 20-step GRPO 结果不能作为策略质量判断依据，主要用于证明训练链路可运行。后续 RL 需要基于修复后的 rollout loop 重新从 LoRA step100 做 sanity run。
