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

`p_limitup_t1` 先由规则分做初始映射；当历史样本足够时，再按历史分层命中率做主导校准，避免把规则高分误读为真实高概率。

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
- 统计校准：当 `outputs/backtests/*/trades.csv` 历史样本足够时，会按概率分层命中率对 `p_limitup_t1` 做主导校准，并输出 `p_limitup_t1_raw`、`calibration_sample_count`、`self_learning_adjustment`。

## 买入观察决策

买入观察计划在 Top 排序之后执行，不改变原始 Top50 排名。系统会综合 `p_limitup_t1`、`wp_score`、`acceptance_score`、`sector_strength_score`、`momentum_score`、`model_confidence`、`capital_score`，并扣减 `risk_penalty_score` 形成 `decision_score`。

组合约束：

- 最多输出 5 支；
- 单板块最多 2 支，避免同质化过高；
- 风险分超过配置阈值的股票不进入买入观察；
- 承接、置信度、收盘位置、日内回撤必须达到配置门槛；
- 输出 14:50 前人工确认条件、放弃条件和买入理由。

尚未接入 LogisticRegression、RandomForest、LightGBM/XGBoost 等复杂机器学习模型；当前是规则模型 + 历史统计校准 + 历史验证指标。
