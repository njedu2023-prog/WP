# WP - 尾盘合格票 T+1 净收益系统

WP 只服务一个目标：

> 在 A 股交易日 14:20-14:50 发布全部通过固定门槛的可成交候选，并提高这些候选按 T+1 收盘卖出、扣除往返成本后净收益大于 0 的概率。

系统不替用户选择唯一股票。一个时点可以有多支合格票，由人工决定是否买入以及买哪一支。没有股票通过门槛时，正式记录 `NO_SIGNAL`，绝不为凑名单降低门槛。

## 生产合同

- `14:20-14:50`：每 5 分钟逐票判定；任何股票第一次通过全部门槛时立即写入 `QUALIFIED`。
- 首次合格时间、首次合格价、模型版本和 T+1 目标日写入后不可修改。
- 同一股票当天再次合格只增加出现次数和最后出现时间，不产生独立交易样本。
- `14:50` 后禁止新增合格票；`15:00` 后页面不展示可继续买入的名单。
- 每支合格票都按 `T+1收盘价 / 首次合格价 - 1 - 往返成本` 验证。
- 模型候选统计与用户实际成交记录严格分离。

详细合同见 [`docs/STRATEGY_CONTRACT.md`](docs/STRATEGY_CONTRACT.md)。

## 两层账本

- `wp_buy_plan_validation`：基础资格快照研究账本，用于积累真实因果样本和训练校准，不代表正式合格票。
- `wp_strategy_ledger`：合格候选不可变账本，是页面逐票净胜率和净收益统计的正式口径。

分层可以避免冷启动自锁：研究样本可以持续积累，但只有通过概率、样本、风险、稳定性和数据新鲜度全部门槛的股票才进入正式候选统计。

## 运行

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m wp.main
python -m pytest -q
```

生产工作流在北京时间 14:16 启动连续会话，覆盖 14:20、14:25、14:30、14:35、14:40、14:45、14:50、14:55 和 15:00。收盘真值工作流从 15:08 开始轮询。

## 关键输出

```text
outputs/csv/wp_strategy_ledger.csv
outputs/json/wp_strategy_ledger.json
outputs/csv/wp_buy_plan_validation.csv
outputs/json/wp_decision_support.json
outputs/json/wp_manifest.json
outputs/html_reports/latest.html
```

GitHub Pages：

```text
https://njedu2023-prog.github.io/WP/outputs/html_reports/latest.html
```

系统只提供人工决策支持，不连接券商、不自动下单，也不保证收益。
