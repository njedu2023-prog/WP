# WP 合格候选合同

## 唯一目标

```text
P(T+1_close_net_return > 0 | T日首次合格时可观测信息)
```

其中：

```text
gross_return = T+1_close / first_qualified_signal_price - 1
net_return   = gross_return - round_trip_cost
```

## 时间与状态

| 北京时间 | 状态 | 允许行为 |
| --- | --- | --- |
| 14:20-14:50 | `ACTIVE` | 逐票判定，可新增多支 `QUALIFIED` |
| 14:50 后至 15:00 | `FROZEN` | 禁止新增候选，只保留已锁定台账 |
| 15:00 以后 | `CLOSED` | 清除可买展示，保留历史候选和待验证状态 |

当天没有股票通过全部门槛时记录 `NO_SIGNAL` 日摘要。`NO_SIGNAL` 不参与收益计算。

## 不可变候选键

正式候选唯一键是：

```text
(strategy_version, plan_trade_date, ts_code)
```

首次写入后，下列字段不可修改：

- `first_signal_time`
- `plan_price`
- `strategy_version`
- `target_trade_date`
- `exit_contract`

后续再次合格只能更新 `last_signal_time`、`last_published_at` 和 `appearance_count`。

## 固定门槛

每支股票独立检查：

- 市场状态允许；
- 行情年龄不超过 120 秒；
- 至少 30 个真实因果样本和 30 个独立交易日；
- 有效加权样本不少于 15；
- 成本后盈利概率不少于 58%；
- 单侧 95% Wilson 下界不少于 50%；
- 成本后期望净收益不少于 0.30%；
- 下行 10% 分位不低于 -4.50%；
- 连续基础资格次数达到配置要求。

任一门槛不通过即保持 `WATCH`。不得为了增加名单数量放宽门槛。

## 数据与统计

基础资格快照用于训练；合格候选台账用于正式统计。收盘日线代理权重为 0，不能授权候选。

正式页面报告：

- 合格信号数、候选日和无合格票日；
- 逐票净盈利率、平均和中位净收益；
- 同日全部候选的等权日收益，仅作为横截面汇总；
- 待验证和已验证候选数。

等权日收益不是用户真实组合收益。用户实际买入结果必须由独立成交记录统计。
