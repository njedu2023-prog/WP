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

V2.1 排序分为：

```text
ranking_score = 0.70 * capital_score + 0.30 * wp_score
```

其中 `capital_score` 综合绝对成交活跃度、5/20 日成交额放大、量比质量、板块成交与可用资金线索，并不是按绝对成交额直接排名。`p_limitup_t1` 对 `ranking_score` 做单调概率映射，基础先验为 1%；字段覆盖不足或实时行情 fallback 时，概率向基础先验收缩或降级。

当前已落地的判断模块：

- 今日涨幅强度：`pct_chg > 8` 是硬过滤，并进入个股强度分。
- 成交额放大：使用 `amount_ratio_5d`、`amount_ratio_20d` 判断放大是否充分或过度。
- 收盘靠近高点：使用 `close_position` 判断资金承接质量。
- 资金承接：综合收盘位置、换手率、成交额放大、量价同步。
- 尾盘/盘中回落风险：使用 `intraday_pullback_pct`、`late_pullback_pct`、`late_volume_ratio`、`tail_lift_flag` 识别冲高回落和尾盘偷袭。
- 日内均衡价格：使用 `intraday_vwap_position` 判断收盘是否强于日内均衡价格。
- 高开低走风险：使用 `gap_open_pct` 与 `open_to_close_pct` 识别。
- 量价结构：`volume_price_sync_flag` 标记成交放大、收盘强、走势未失控的结构。
- 动量：使用真实 `ret_3d`、`ret_5d`、`ret_10d`、`ret_20d`，结合 `ma5_position`、`ma10_position`、`ma20_position`，并识别 `high_20d_break`、`platform_break_20d`。
- 龙虎榜：`dragon_tiger_flag` 和 `dragon_tiger_net_rate` 进入资金分和解释。
- 热门题材与公告：`hot_topic_flag`、`announcement_flag` 已接入口径；没有上游数据时按 0 处理。
- 统计校准：只读取同模型版本、早于当前交易日、`backtest_data_mode=intraday_1420` 且允许校准的样本；先按交易日和代码去重，再用保序 Logit 截距校准。收盘日线代理回测不会进入实时校准。

## 尾盘收益排序

`tail_profit_v1` 在完整的涨幅超过 8% 候选池内计算横截面排名：

```text
tail_profit_score = 100 * (
  0.50 * (1 - pct_chg_rank)
+ 0.25 * capital_score_rank
+ 0.20 * sector_strength_score_rank
+ 0.05 * (1 - risk_penalty_score_rank)
)
```

排序前执行硬约束：

- `8% < pct_chg <= 12%`；
- `risk_penalty_score <= 45`；
- `close_position >= 50`；
- `0 < amount_ratio_5d <= 2.5`；
- 前一日未涨停、当日未涨停，且基础数据完整。

Top50 与尾盘观察使用同一排序。并列时依次按涨幅较低、资金分较高、板块分较高、风险分较低排序，避免浮点尾数决定结果。14:35 后最多输出 1 支主票；没有合格候选时输出空仓。14:50 前仍需人工确认涨幅守住 8%、承接未破坏，系统不做自动下单。

当前模型为固定、可解释的规则模型。历史结果不能保证未来收益；真实效果只按新模型版本的最终尾盘快照和下一交易日市场真值累计验证。
