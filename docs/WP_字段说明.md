# WP 字段说明

- `p_limitup_t1`: 下一交易日涨停概率。
- `wp_score`: 综合排序分。
- `sector_strength_score`: 板块强度分。
- `stock_strength_score`: 个股强度分。
- `acceptance_score`: 资金承接质量分。
- `risk_penalty_score`: 风险惩罚分。
- `core_reason`: 入选核心理由。
- `risk_reason`: 风险提示。
- `amount_ratio_5d`: 当日成交额相对近 5 个交易日均额放大倍数。
- `amount_ratio_20d`: 当日成交额相对近 20 个交易日均额放大倍数。
- `ret_3d` / `ret_5d` / `ret_10d` / `ret_20d`: 近 3/5/10/20 个交易日涨跌幅。
- `ma5_position` / `ma10_position` / `ma20_position`: 收盘价相对 5/10/20 日均线的位置。
- `close_position`: 收盘价在日内高低区间中的位置，越接近 100 越靠近全天高点。
- `intraday_pullback_pct`: 从全天高点回落到收盘的幅度。
- `intraday_vwap_position`: 收盘相对日内均衡价格的强弱位置；无分钟 VWAP 时使用典型价近似。
- `late_pullback_pct`: 尾盘/盘中回落风险幅度。
- `late_price_change_pct`: 尾盘价格变化强弱。
- `late_volume_ratio`: 尾盘量能相对强度。
- `tail_lift_flag`: 是否存在尾盘放量拉升结构。
- `open_to_close_pct`: 开盘到收盘的涨跌幅，用于识别高开低走。
- `gap_open_pct`: 开盘相对昨收的跳空幅度。
- `amplitude`: 当日振幅。
- `high_20d_break`: 是否突破近 20 日高点。
- `platform_break_20d`: 是否突破近 20 日收盘平台。
- `dragon_tiger_flag`: 是否有龙虎榜资金线索。
- `hot_topic_flag`: 是否属于当日热门板块/题材口径。
- `announcement_flag`: 是否存在公告或异动信息扰动；无上游数据时为 0。
- `stock_age_days`: 上市天数，用于新股过滤。
- `suspended_flag`: 停牌或无成交状态标记。
- `delist_flag`: 退市整理或名称含退市风险标记。
- `data_quality_flag`: 关键字段缺失或价格/成交额异常标记。
- `high_open_low_walk_flag`: 是否出现高开低走弱结构。
- `volume_price_sync_flag`: 是否量价结构同步。
- `p_limitup_t1_raw`: 统计校准前的规则模型概率。
- `calibration_sample_count`: 当前概率区间用于统计校准的历史样本数。
- `self_learning_adjustment`: 历史统计校准对概率的修正幅度。

## 买入观察计划字段

- `buy_rank`: 买入观察优先级，最多 5 支。
- `portfolio_group`: 组合层级，核心或标准。
- `decision_score`: 买入决策分，综合次日概率、WP 评分、承接、板块、动量、置信度和风险惩罚。
- `confirm_before_buy`: 14:50 前需要人工确认仍然成立的条件。
- `reject_if`: 触发后应放弃买入的条件。
- `buy_reason`: 进入买入观察计划的核心原因。
- `skip_reason`: 未进入买入观察计划的原因，输出在 `wp_buy_decision.csv`。
