# WP - T+1 净盈利概率决策系统

WP 只服务一个目标：

> 最大化 A 股交易日 14:20-15:00 可成交买入后，按预先定义的 T+1 退出合同卖出并扣除成本后，净收益大于 0 的概率。

系统允许 `NO_TRADE`。没有足够的实时因果证据时，空仓是正式决策，不以凑名单为目标。

## 生产合同

- `14:20-14:44`：只采集真实尾盘快照，不形成正式买入决策。
- `14:45-14:54`：每个交易日最多锁定一次 `BUY`，否则继续观察。
- `14:55`：尚未锁定 `BUY` 时，正式记录 `NO_TRADE`。
- `15:00` 以后：禁止生成或修改当日买入名单。
- 入场参考：正式决策时的实时价格，人工确认可成交。
- 退出合同：T+1 收盘卖出。
- 验证口径：`T+1收盘价 / 锁定买价 - 1 - 往返成本`。

当日正式决策写入不可变策略账本。后续刷新不能换股票、换买价或把盘后名单补成盘中信号。

## 模型约束

- 只有真实 `14:20-14:55` 因果样本可以授权交易。
- 收盘日线代理仅用于研究，权重固定为 `0`，不能提高生产置信度。
- 生产门槛同时检查样本量、盈利概率、Wilson 概率下界、成本后期望、下行分位数、连续合格和连续领先。
- 任一门槛不满足即 `WATCH` 或 `NO_TRADE`。
- 当前系统不承诺盈利；策略账本中的样本外净收益是唯一晋级依据。

详细合同见 [`docs/STRATEGY_CONTRACT.md`](docs/STRATEGY_CONTRACT.md)。

## 运行

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m wp.main
python -m pytest -q
```

生产工作流在北京时间 14:16 启动一段连续会话，14:20 至 15:00 每 5 分钟刷新一次。这样复用同一 runner 和依赖环境，避免关键尾盘时段反复冷启动。

## 关键输出

```text
outputs/csv/wp_strategy_ledger.csv
outputs/json/wp_strategy_ledger.json
outputs/csv/wp_legacy_history_audit.csv
outputs/json/wp_legacy_history_audit.json
outputs/csv/wp_t1_forecast.csv
outputs/csv/wp_decision_support.csv
outputs/csv/wp_buy_plan_validation.csv
outputs/json/wp_manifest.json
outputs/html_reports/latest.html
```

- `wp_strategy_ledger`：唯一正式策略绩效口径，一天最多一条 `BUY/NO_TRADE`。
- `wp_buy_plan_validation`：研究样本账本，不计入正式策略收益。
- `wp_t1_forecast`：概率、置信下界、成本后期望及样本充分性。

GitHub Pages：

```text
https://njedu2023-prog.github.io/WP/outputs/html_reports/latest.html
```

系统只提供人工决策支持，不连接券商、不自动下单，也不保证收益。
