# WP V7 字段合同

字段按用途分层。`v3` 仅是既有文件路径和 JSON schema 的兼容名称，不代表仍在运行旧模型。

## 身份与时间

- `trade_date`：信号交易日。
- `target_trade_date`：固定退出日，即下一个 A 股交易日。
- `signal_slot`：`14:20` 至 `14:50` 的七个合法五分钟时点之一。
- `ts_code` / `name`：股票代码和名称。
- `market_data_time`：行情数据实际覆盖时间。
- `capture_started_at` / `capture_completed_at`：实时采集开始和结束时间。
- `slot_bar_time` / `slot_bar_lag_minutes`：用于该信号的分钟线时间及其相对目标时点的滞后。
- `data_age_seconds`：推理时行情年龄。

## 成交合同

- `signal_price`：合法时点可见的信号价。
- `first_signal_price`：同一股票当日第一次通过门槛时锁定的不可变信号价。
- `entry_benchmark_slot`：首次信号后 5 分钟的固定结算时点。
- `entry_benchmark_price`：结算时点真实 5 分钟线的收盘价。
- `entry_benchmark_status`：`PENDING`、`SETTLED` 或 `NON_FILL`。
- `entry_price`：`entry_benchmark_price` 加声明的入场滑点；不得从信号价回退。
- `entry_slippage_bps`：入场滑点，当前合同为 10 bps。
- `round_trip_cost_bps`：除入场滑点外的往返成本，当前合同为 25 bps。
- `baseline_all_in_cost_bps`：基准总摩擦，当前为 35 bps。
- `slot_amount`：信号分钟成交额。
- `execution_eligible`：价格、上市天数、流动性、容量、涨跌停距离和数据完整性全部可执行。
- `entry_fillable` / `exit_fillable`：历史标签中的入场和 T+1 收盘退出可成交判断。

## 模型输出

- `p_entry_fill`：信号后下一根 5 分钟线按容量与涨停约束可成交买入的概率。
- `p_exit_fill_given_entry`：已买入后，T+1 收盘合同可执行卖出的条件概率。
- `p_round_trip_fill` / `p_round_trip_fill_lower`：买卖完整成交概率及集成保守值。
- `p_conditional_net_positive`：买卖均成交时，扣成本后净收益大于零的条件概率。
- `p_net_positive`：买入成交、T+1 可卖且扣成本后净收益大于零的校准联合概率。
- `p_net_positive_lower`：联合概率的集成保守值。
- `p_cross_section_top`：完整成交条件下，该票进入同槽收益前 20% 的概率。
- `p_severe_loss`：完整成交后净收益不高于 -2% 的条件概率。
- `conditional_expected_net_return_pct`：完整成交条件下的净收益均值估计。
- `expected_utility_pct`：把入场未成交记 0、退出失败记固定惩罚后的综合净效用。
- `downside_q10_pct`：完整成交条件下净收益第 10 分位估计。
- `selection_score` / `selection_rank_pct`：LambdaRank 横截面排序分和同槽百分位。
- `probability_model_spread` / `fill_probability_model_spread` /
  `selection_rank_spread`：长短训练窗成员间不稳定度。
- `model_version` / `model_fingerprint` / `policy_fingerprint`：模型、完整模型指纹和候选政策指纹。

## 固定门槛证据

- `passes_entry_fill_probability`：买入成交概率通过。
- `passes_exit_fill_probability`：T+1 可卖概率通过。
- `passes_round_trip_fill_probability`：完整成交保守概率通过。
- `passes_probability`：主概率通过。
- `passes_probability_lower`：保守概率下界通过。
- `passes_conditional_probability`：完整成交后的盈利概率通过。
- `passes_severe_loss`：严重损失概率通过。
- `passes_expected_utility`：综合净效用通过。
- `passes_downside`：下行分位通过。
- `passes_selection_rank`：横截面排名通过。
- `passes_prior_oos_evidence`：当前冻结政策已经在此前设计期和一次性确认期
  通过样本数、独立交易日、胜率下界、平均净收益和 Profit Factor 门禁。
- `passes_stability`：长短训练窗分歧通过。
- `passes_freshness`：实时行情新鲜度通过。
- `passes_policy`：全部固定门槛同时通过。

## 候选账本

- `status`：模型候选状态；`QUALIFIED` 或 `SHADOW_QUALIFIED`，不表示用户成交。
- `first_signal_time`：首次通过门槛的合法时点，不可修改。
- `last_signal_time`：当日最后一次再次通过门槛的时点。
- `appearance_count`：当日通过门槛的时点数。
- `first_signal_features`：首次信号时冻结的因果特征。
- `qualification_evidence`：首次通过每项门槛的证据。
- `covered_slots` / `missing_slots`：当日七个信号时点的覆盖和缺失集合。
- `integrity_status`：`COLLECTING`、`COMPLETE`、`INCOMPLETE` 或
  `INCOMPLETE_ENTRY`。
- `NO_SIGNAL`：七个时点没有任何合格票的正式日摘要，不参与收益统计。

## T+1 真值

- `truth_status`：`pending` 或 `verified`。
- `gross_return_pct`：不可变入场价到复权 T+1 收盘价的毛收益。
- `net_return_pct`：毛收益扣除合同往返成本后的净收益。
- `net_positive`：`net_return_pct > 0`。
- `t1_open` / `t1_high` / `t1_low` / `t1_close`：T+1 日真实四价，仅用于事后验证。
- `truth_contract`：固定为下一根 5 分钟线参考成交价到 T+1 收盘、扣声明成本。

模型候选统计和用户真实成交记录必须分离；人工是否买入、实际成交价和仓位不属于这些字段。
