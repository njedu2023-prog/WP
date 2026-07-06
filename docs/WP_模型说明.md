# WP 模型说明

第一阶段模型使用规则分：

```text
wp_score =
  0.30 * sector_strength_score
+ 0.25 * stock_strength_score
+ 0.20 * acceptance_score
+ 0.10 * momentum_score
+ 0.10 * capital_score
+ 0.05 * pattern_score
- 0.25 * risk_penalty_score
```

`p_limitup_t1` 使用 sigmoid 映射到 0-100%。
