# Synthetic Customer Support Full Manual Audit

- Date: 2026-08-09
- Total tasks: 590
- Human audit shards: 20 shards completed
- Final verdict: {'PASS': 590}
- Deterministic checker: 590/590 passed, 0 failed, 120 warnings
- Pattern issue candidates after repair: 0
- Repair operations logged: 854
- Pytest: `tests/test_synthetic_customer_support.py` 3 passed

## Task Type Counts

- cancel_order: 60
- compound: 60
- edge_case: 60
- exchange_item: 80
- price_match_refund: 90
- return_item: 65
- shipping_claim: 110
- warranty_claim: 65

## Repair Categories

- compound_bulk_shipping_summary_math_clarification: 5
- edge_no_write_simulator_preview_residue_cleanup: 15
- edge_vague_wrong_item_env_alignment: 5
- edge_vague_wrong_item_task_text_alignment: 5
- edge_wrong_item_env_modeling_clarification: 1
- exchange_price_difference_requirement: 41
- exchange_target_product_profile_completion: 116
- exchange_target_product_realism: 36
- exchange_target_product_realism_task_text: 36
- exchange_target_text_alignment: 27
- exchange_task_text_normalization: 58
- old_expiry_warranty_choice_clarification: 11
- price_match_product_profile: 90
- price_opening_order_token_cleanup: 2
- return_mislabel_summary_policy_clarification: 1
- return_not_as_described_language: 4
- shipping_low_value_free_shipping_explained: 53
- shipping_product_metadata: 6
- shipping_product_profile_completion: 9
- template_challenge_meta_cleanup: 85
- template_residue_cleanup: 120
- warranty_extended_coverage_explained: 44
- warranty_product_profile: 55
- warranty_product_profile_completion: 29

## Final Assessment

All 590 synthetic tasks are marked eligible for SFT after full sharded manual review, deterministic validation, pattern checks, and targeted repairs. The last unresolved human-review findings were exchange replacement-product profile residue and missing large price-difference disclosure requirements; both were fixed and revalidated.

Residual traceability note: some template-derived compound/edge tasks keep null `metadata.gold_tool_plan` or `metadata.expected_state_diff`. This is not a content contradiction in the official harness path, because those tasks are evaluated through `state_requirements` and `task_requirements`. For SFT sampling, do not rely on metadata gold plans for those tasks; use sampled trajectories or official harness rollouts.

## Files

- Per-task review records: `synthetic_data/customer_support/full_manual_audit/reviews`
- Summary JSON: `synthetic_data/customer_support/full_manual_audit/summary.json`
- Repair log: `synthetic_data/customer_support/full_manual_audit/repair_log.json`
- Deterministic results: `synthetic_data/customer_support/full_manual_audit/deterministic_check_all.json`
