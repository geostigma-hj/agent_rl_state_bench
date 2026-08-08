# Synthetic Customer Support Manual Audit

- population: 590
- sample: 59 (10.0%)
- seed: 20260809
- verdicts: {'PASS': 53, 'REVIEW': 5, 'FAIL': 1}

## Review / Fail Items
- REVIEW syn-price-032-auto_017_standard_debit_card_slatetab_pro_11_inch_5_drop (price_match_refund): usable_if_metadata_noise_accepted
  - 主 price-match 路径正确：原价 599、current_price 594、正确退款 5，用户误报 20 可由 agent 查价纠正。
  - 质量疑点：产品名 SlateTab Pro 11-inch 但 subcategory 为 headphones_budget，属于商品属性模板残留。
- REVIEW syn-price-054-auto_039_silver_gift_card_probook_laptop_40_drop (price_match_refund): usable_if_metadata_noise_accepted
  - 主 price-match 路径正确：ProBook Laptop 原价 799、current_price 759、退款 40；用户误报 35 可被纠正。
  - 质量疑点：产品名是 laptop，但 subcategory/weight 像 headphones 模板残留，现实感偏弱。
- REVIEW syn-price-068-auto_053_gold_paypal_flexcharge_travel_dock_25_drop (price_match_refund): usable_if_metadata_noise_accepted
  - 主 price-match 路径正确：原价 69、current_price 44、退款 25；用户误报 40 可由 agent 查价纠正。
  - 质量疑点：FlexCharge Travel Dock 的 subcategory 为 headphones_budget，商品属性不自然。
- FAIL syn-return-046-auto_031_gold_prime_not_as_described (return_item): exclude_from_sft_until_fixed
  - 核心矛盾：opening_message 是“I opened it, but I do not want it anymore”，自然语义是 changed_mind，不是 not_as_described。
  - user_simulator/context 和 task_requirements 强制 not_as_described，但 known_info 没给任何“描述不符”的事实，agent 无法从对话事实合理推出该原因。
  - state_requirements/gold_tool_plan 要求 refund_amount=241、restocking_fee=0；这与用户开场的 changed_mind 事实冲突，judge auto-accept 未捕捉该问题。
- REVIEW syn-shipping-068-auto_058_gold_nonfragile_smart_fitness_band (shipping_claim): usable_if_metadata_noise_accepted
  - 运输损坏主路径正确：full refund 89、return_label_issued=true、restocking_fee=0、非 fragile 不给 goodwill。
  - 质量疑点：Smart Fitness Band 的 subcategory 写成 headphones，商品属性有轻微不一致。
- REVIEW syn-warranty-011-long_expiry_paid_options (warranty_claim): needs_policy_clarification_before_sft_prefer_exclude
  - 保修已过期超过 30 天，paid option 路径可执行，claim_count/status/resolution 可由工具达成。
  - 质量疑点：任务语义是 full-price repair or 25% off replacement 的选项，但 state_requirements 强制创建 replacement_item_id，用户选择不够明确。

## Counts By Type
- cancel_order: {'PASS': 6}
- compound: {'PASS': 6}
- edge_case: {'PASS': 6}
- exchange_item: {'PASS': 8}
- price_match_refund: {'PASS': 6, 'REVIEW': 3}
- return_item: {'PASS': 6, 'FAIL': 1}
- shipping_claim: {'PASS': 10, 'REVIEW': 1}
- warranty_claim: {'REVIEW': 1, 'PASS': 5}
