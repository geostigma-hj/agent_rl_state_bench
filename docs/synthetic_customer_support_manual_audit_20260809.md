# Synthetic Customer Support Manual Audit - 2026-08-09

## Purpose

Before sampling SFT cold-start trajectories, we manually audited a stratified 10% sample of the newly generated customer-support synthetic tasks to estimate data construction quality and identify tasks that should be filtered or repaired.

## Scope

- Dataset: `synthetic_data/customer_support`
- Population: 590 synthetic tasks
- Sample size: 59 tasks
- Sampling seed: `20260809`
- Sampling method: stratified by `task_type`
- Review artifacts:
  - `synthetic_data/customer_support/manual_reviews/sample_manifest.json`
  - `synthetic_data/customer_support/manual_reviews/reviews/*.json`
  - `synthetic_data/customer_support/manual_reviews/summary.json`
  - `synthetic_data/customer_support/manual_reviews/manual_review_report.md`

## Review Criteria

Each sampled task was checked against:

- Task intent clarity
- Consistency between `task_summary`, `opening_message`, `known_info`, and `task_rules`
- Reachability and correctness of `state_requirements`
- Whether `task_requirements` are reasonable and non-contradictory
- Whether the automatic judge decision is credible
- Whether privacy, authority, refund, amount, or policy constraints are violated

## Result

| Verdict | Count | Rate |
| --- | ---: | ---: |
| PASS | 53 | 89.8% |
| REVIEW | 5 | 8.5% |
| FAIL | 1 | 1.7% |

Strict pass rate is `53 / 59 = 89.8%`.

Non-fail rate is `58 / 59 = 98.3%`.

For SFT cold start, the recommended usable rate after excluding critical/problematic tasks is approximately `57 / 59 = 96.6%` on the audited sample.

## SFT Filtering Recommendation

Exclude these tasks before SFT:

```text
syn-return-046-auto_031_gold_prime_not_as_described
syn-warranty-011-long_expiry_paid_options
```

Keep but optionally clean later:

```text
syn-price-032-auto_017_standard_debit_card_slatetab_pro_11_inch_5_drop
syn-price-054-auto_039_silver_gift_card_probook_laptop_40_drop
syn-price-068-auto_053_gold_paypal_flexcharge_travel_dock_25_drop
syn-shipping-068-auto_058_gold_nonfragile_smart_fitness_band
```

The keep-but-review group has correct main task logic and reachable state targets, but contains product metadata noise such as mismatched `subcategory` fields.

## Critical Failure

`syn-return-046-auto_031_gold_prime_not_as_described`

The opening message says the customer opened the item but no longer wants it, which is a `changed_mind` return. The task and state targets force `not_as_described`, with no supporting user fact. This makes the task unsuitable for SFT until repaired.

## Main Review Issues

- Several price-match or shipping tasks have product metadata residue, such as a laptop/tablet/dock/fitness-band product carrying a headphone-like subcategory.
- One warranty task has an ambiguous paid-repair versus discounted-replacement resolution, while the state target forces a replacement item.

## Project Decision

Proceed with SFT cold-start sampling after applying the exclusion list. The synthetic data quality is good enough for the next stage, but the generator should later be cleaned so product attributes stay synchronized when product names are changed.
