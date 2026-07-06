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

当前已落地的判断模块：

- 今日涨幅强度：`pct_chg > 6` 是硬过滤，并进入个股强度分。
- 成交额放大：使用 `amount_ratio_5d`、`amount_ratio_20d` 判断放大是否充分或过度。
- 收盘靠近高点：使用 `close_position` 判断资金承接质量。
- 资金承接：综合收盘位置、换手率、成交额放大、量价同步。
- 尾盘/盘中回落风险：使用 `intraday_pullback_pct` 处罚冲高回落。
- 高开低走风险：使用 `gap_open_pct` 与 `open_to_close_pct` 识别。
- 量价结构：`volume_price_sync_flag` 标记成交放大、收盘强、走势未失控的结构。
- 动量：使用真实 `ret_5d`、`ret_20d`，并识别 `high_20d_break`、`platform_break_20d`。
- 龙虎榜：`dragon_tiger_flag` 和 `dragon_tiger_net_rate` 进入资金分和解释。

尚未接入复杂机器学习模型；当前仍是规则模型 + 历史验证指标。
