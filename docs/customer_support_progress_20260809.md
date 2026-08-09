# Customer Support Agent RL Progress

Date: 2026-08-09

## Baseline Decision

- Final project agent model for local training/eval: `Qwen3-4B-Instruct-2507`.
- Primary teacher/simulator provider for synthetic data and SFT trajectory generation: DeepSeek V4 Flash.
- Simulator temperature: `0.2`.
- Main agent decoding for baselines follows provider/model defaults unless explicitly testing thinking variants.

## Baseline Results on Official Train Split

- `Qwen3-4B-Instruct-2507` without thinking is the selected local model; thinking was slower and not worth the latency for this project stage.
- DeepSeek V4 Flash and Pro were evaluated with the official STATE-Bench harness.
- The final model choice is based on the observed latency/quality tradeoff: use Qwen3-4B-Instruct-2507 for the trainable agent, not DeepSeek Pro/Flash as the trainable target.

Detailed raw outputs remain under `outputs/`.

## Synthetic Data Status

- Synthetic customer_support tasks: 590.
- Task type distribution:
  - `cancel_order`: 60
  - `compound`: 60
  - `edge_case`: 60
  - `exchange_item`: 80
  - `price_match_refund`: 90
  - `return_item`: 65
  - `shipping_claim`: 110
  - `warranty_claim`: 65
- Full manual audit: 590/590 PASS.
- V4 Flash judge: 590/590 ACCEPT.
- Deterministic checker: 590/590 passed, 0 failed, 120 warnings.
- Pattern residue check: 0 remaining issues.
- Repair log: 854 field/text-level repairs.

Important audit artifacts:

- `synthetic_data/customer_support/full_manual_audit/summary.json`
- `synthetic_data/customer_support/full_manual_audit/full_audit_report.md`
- `synthetic_data/customer_support/full_manual_audit/repair_log.json`
- `synthetic_data/customer_support/full_manual_audit/reviews/`

## Metric Readiness

The constructed data keeps the fields needed by official STATE-Bench evaluation:

- State metrics: `tasks/*.json::state_requirements` plus rollout `state_diff`.
- Process/trajectory metrics: `tasks/*.json::task_requirements` plus rollout `conversation` and `tool_calls`.
- Tool/write metrics: rollout `conversation.tool_calls`; customer_support write tools are declared in domain config.

For 120 template-derived compound/edge tasks, `metadata.gold_tool_plan` and `metadata.expected_state_diff` are null. This does not affect official evaluation because scoring uses task requirements and rollout diffs, not those metadata helper fields.

## Teacher Rollouts And SFT Data

V4 Flash teacher rollouts were collected with the official harness:

- Round 1: 441/590 accepted.
- Round 2: 438/590 accepted.
- Round 3 targeted retry on failed-both tasks: 21 accepted.
- SFT trajectories after merge: 900 accepted trajectories.

The first SFT conversion was discarded because assistant final text and tool calls were packed into the same assistant message. The clean conversion now stores assistant tool-call messages separately from final natural-language responses and drops null-expanded tool arguments from parquet readback.

Reproducibility note: the active local verl checkout at `/home/hj/shixi/verl` was also patched in `verl/utils/dataset/multiturn_sft_dataset.py` to drop null-expanded nested tool-call arguments after reading parquet. That checkout is outside this repository, so future runs need either the same local patch or an equivalent upstream/data-loader fix.

Clean SFT data:

- Main split used for cold start: `data/sft/customer_support_v4_flash_teacher_round123_targeted_toolsplit_clean_main8192`
- Train/val rows: 849/45.
- Long-tail over 8192 tokens after cleaning: 6 train rows, 0 val rows.

## Cold-Start SFT

Full-parameter SFT was run on `/mnt/models/Qwen3-4B-Instruct-2507` with verl FSDP, not LoRA:

- GPUs: 6 x RTX 4090.
- Max sequence length: 8192.
- Train batch size: 24.
- Learning rate: `1e-5`.
- Total steps: 100.
- Save/eval interval: 25 steps.
- Attention: `flash_attention_2`.
- Checkpoint save contents: model and extra state only.

Validation loss:

- Step 25: 0.4191.
- Step 50: 0.3944.
- Step 75: 0.3990.
- Step 100: 0.4163.

Step 50 was selected for evaluation because it had the lowest validation loss. The merged HF checkpoint is:

`checkpoints/customer_support/qwen3_4b_sft_cold_start_round123_toolsplit_clean_main8192_full_flash_lr1e5_sp2_b24/global_step_50_hf_merged`

Training artifacts:

- Log: `logs/sft/qwen3_4b_instruct_2507_v4_flash_teacher_round123_toolsplit_clean_main8192_full_flash_lr1e5_sp2_b24_20260809_062924.log`
- Metrics CSV: `logs/sft/qwen3_4b_instruct_2507_v4_flash_teacher_round123_toolsplit_clean_main8192_full_flash_lr1e5_sp2_b24_20260809_062924_metrics.csv`

## SFT Test Evaluation

The step 50 checkpoint was evaluated on the official STATE-Bench customer_support test split with the official harness:

- Agent: `OpenAICompatibleToolCallingAgent`
- Agent model label: `Qwen3-4B-SFT-clean-step50`
- Local serving: vLLM, 5 endpoints on GPUs 0-4.
- vLLM context: 12288 tokens.
- Agent max completion cap: 1024 tokens for the main run.
- Overflow rerun: task `72-challenge_bulk_plus_shipping_clawback` used max completion cap 256 to fit the 12288-token local vLLM context.
- Simulator/Judge: DeepSeek V4 Flash through `STATE_BENCH_EVAL_PROVIDER=deepseek`.
- Simulator temperature: `0.2`.
- Main agent temperature: provider/default local serving behavior; no explicit temperature override.
- UX scoring: skipped.

Result on 50/50 test tasks:

- `task_completion_pass@1`: 24%.
- Passed tasks: 12/50.
- Failed tasks: 38/50.
- Total tool calls: 503.
- Tool errors: 112.

Result path:

`outputs/customer_support_qwen3_4b_sft_clean_step50_main8192_lr1e5_12k_mtok1024_native_test_20260809_1239`

Category pass-rate estimate from task names:

- `cancel_order`: 1/2.
- `edge_case`: 3/3.
- `exchange_item`: 1/7.
- `price_match_refund`: 0/6.
- `return_item`: 2/8.
- `shipping_claim`: 3/14.
- `warranty_claim`: 1/3.
- Other challenge tasks: 1/7.

Immediate observation: SFT produced working tool calls, but quality is still low on test. The biggest visible issue is excessive or invalid tool use, with 112 tool errors across 50 tasks. Price-match, exchange, compound, and multi-action challenge cases are the most obvious weak areas for targeted data/RL.

## LoRA SFT Ablation

A LoRA SFT run was added because the full-parameter SFT result was worse than the base model on the official test split.

Training configuration:

- Base model: `/mnt/models/Qwen3-4B-Instruct-2507`.
- Data: `data/sft/customer_support_v4_flash_teacher_round123_targeted_toolsplit_clean_main8192`.
- Train/val rows: 849/45.
- GPUs: 5 x RTX 4090, one model shard per rank during verl FSDP training.
- LoRA rank/alpha: `r=16`, `alpha=32`.
- Max sequence length: 8192.
- Train batch size: 20.
- Learning rate: `1e-4`.
- Total steps: 100, approximately 2.3 epochs over the 849-row training split.
- Save/eval interval: 25 steps.

Validation loss:

- Step 25: 0.5781.
- Step 50: 0.5065.
- Step 75: 0.4687.
- Step 100: 0.4501.

The validation curve was still decreasing at step 100, so the LoRA run had not clearly converged. Step 100 was evaluated first to get a fast downstream signal.

LoRA artifact notes:

- verl's FSDP merger produced a full HF model plus a separate `lora_adapter` directory.
- For vLLM evaluation, the adapter was explicitly merged into the base weights with PEFT `merge_and_unload` to avoid accidentally serving the unadapted base model.
- Merged inference checkpoint: `checkpoints/customer_support/qwen3_4b_sft_lora_r16_a32_round123_toolsplit_clean_main8192_lr1e4_b20/global_step_100_hf_lora_merged`.

Official harness test50 result:

- Output: `outputs/customer_support_qwen3_4b_sft_lora_r16_step100_main8192_lr1e4_12k_mtok1024_native_test_20260809_1343`.
- Agent model label: `Qwen3-4B-SFT-LoRA-r16-step100`.
- Local serving: vLLM, 5 endpoints on GPUs 0-4.
- vLLM context: 12288 tokens.
- Agent max completion cap: 1024 tokens for the main run.
- Overflow rerun: tasks `41-hard_two_lost_shipments_threshold_split` and `71-hard_exchange_late_compensation_destination_choice` used max completion cap 256 to fit the 12288-token local vLLM context.
- Simulator/Judge: DeepSeek V4 Flash.
- Simulator temperature: `0.2`.
- UX scoring: skipped.

Result on 50/50 test tasks:

- `task_completion_pass@1`: 46%.
- Passed tasks: 23/50.
- Failed tasks: 27/50.
- Total tool calls: 547.
- Tool errors: 151.
- Mean turns: 5.5.

Category pass-rate estimate from task names:

- `cancel_order`: 2/4.
- `edge_case`: 3/3.
- `exchange_item`: 3/7.
- `price_match_refund`: 1/5.
- `return_item`: 5/8.
- `shipping_claim`: 3/12.
- `warranty_claim`: 2/3.
- Other challenge tasks: 4/8.

Current comparison on official customer_support test50:

- Base Qwen3-4B-Instruct-2507: 30%.
- Full-parameter SFT step50: 24%.
- LoRA SFT step100: 46%.

Immediate observation: LoRA is clearly better than both base and full-parameter SFT in this low-data regime, even though the validation loss had not converged. The main residual issue is still tool-call precision and process compliance: the LoRA model passes more tasks but also produces many tool errors on hard multi-step cases.

## LoRA Continuation To 200 Steps

The LoRA run was continued beyond step 100 to test whether the still-decreasing validation loss translated into better official harness performance.

Implementation note: the SFT script originally relied on verl's default `trainer.total_epochs=4`, which stopped a `TOTAL_TRAINING_STEPS=200` continuation at epoch 4 before reaching step 200. The script now exposes `TOTAL_EPOCHS`; the continuation was rerun with `TOTAL_EPOCHS=6` and resumed from `global_step_150`.

Continuation validation loss:

- Step 100: 0.4501.
- Step 125: 0.4383.
- Step 150: 0.4316.
- Step 175: 0.4353.
- Step 200: 0.4318.

Official harness test50 results:

| Checkpoint | Val loss | Pass@1 | Passed | Tool calls | Tool errors | Mean turns | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LoRA step100 | 0.4501 | 46% | 23/50 | 547 | 151 | 5.5 | `outputs/customer_support_qwen3_4b_sft_lora_r16_step100_main8192_lr1e4_12k_mtok1024_native_test_20260809_1343` |
| LoRA step150 | 0.4316 | 38% | 19/50 | 501 | 112 | 5.1 | `outputs/customer_support_qwen3_4b_sft_lora_r16_step150_main8192_lr1e4_12k_mtok1024_native_test_20260809_1437` |
| LoRA step200 | 0.4318 | 36% | 18/50 | 442 | 89 | 5.2 | `outputs/customer_support_qwen3_4b_sft_lora_r16_step200_main8192_lr1e4_12k_mtok1024_native_test_20260809_1449` |

Conclusion: validation loss is misleading for this benchmark after step 100. Continued SFT reduces tool errors, but it also reduces task success. The likely failure mode is behavior overfitting / policy drift away from useful exploratory tool use. The best current cold-start checkpoint remains LoRA `global_step_100`, not step150 or step200.

## Next Steps

1. Use LoRA `global_step_100` as the current cold-start policy.
2. Do not select SFT checkpoints by validation loss alone; use official-harness task success or a held-out task validation set.
3. Inspect failed LoRA step100 trajectories by tool error type and task category, especially shipping, price match, exchange, and compound tasks.
4. For RL, initialize from LoRA step100 and use short RL sanity runs before launching full-data training.
