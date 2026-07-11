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

## 买入观察决策

买入观察计划严格在 Top50 内执行，不改变原始 Top50 排名。系统会综合 `p_limitup_t1`、`wp_score`、`acceptance_score`、`sector_strength_score`、`momentum_score`、`model_confidence`、`capital_score`，并扣减 `risk_penalty_score` 形成 `decision_score`。

组合约束：

- 最多输出 5 支；
- 单板块最多 2 支，避免同质化过高；
- 风险分超过配置阈值的股票不进入买入观察；
- 承接、置信度、收盘位置、日内回撤必须达到配置门槛；
- 输出 14:50 前人工确认条件、放弃条件和买入理由。

尚未接入 LogisticRegression、RandomForest、LightGBM/XGBoost 等监督学习模型。当前是可解释规则模型、保序统计校准和时间分段历史验证；收盘代理回测不能证明 14:20 实盘收益。
