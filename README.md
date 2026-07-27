# WP V3 - 尾盘 T+1 净盈利概率系统

WP V3 只优化一个可检验目标：

> 最大化 A 股主板股票在 T 日 14:20-14:50 可成交买入后，按固定 T+1 收盘卖出并扣除既定往返成本后，净收益大于 0 的概率。

系统发布全部通过固定门槛的候选，由人工决定是否买入以及买哪一支。零候选是正常结果，模型不得为了生成名单放宽门槛。

## V3 与旧系统的根本区别

- 不再把 `涨幅 8%-12%` 当作生产入口；模型从全市场主板可成交样本学习。
- 历史训练与实时推理使用同一套 5 分钟因果特征。
- 标签是实际交易合同的净收益正负，不是涨停、排名或未来最高价代理。
- 采用时间窗口集成、概率校准、期望收益回归和下行分位回归共同决策。
- 每支股票第一次通过即锁定信号时间、信号价、政策/模型指纹和 T+1 目标日。
- 所有候选逐票验证；模型候选与用户实际成交严格分离。
- 三年滚动样本外回测通过后，仍必须完成至少 150 个交易日的前瞻影子运行。

## 固定交易合同

| 项目 | 合同 |
| --- | --- |
| 信号时点 | 14:20、14:25、14:30、14:35、14:40、14:45、14:50 |
| 股票范围 | A 股主板，其他板块只能作为独立 challenger |
| 参考订单 | 100,000 元，且不超过当前 5 分钟成交额的 1% |
| 入场价 | 首次信号价加 10bp 滑点 |
| 往返成本 | 25bp |
| 退出 | 下一 A 股交易日收盘 |
| 无法退出 | 停牌或 T+1 收盘钉在跌停价时计为失败，缺价采用预先声明的 -10% 惩罚 |
| 截止 | 14:50 后禁止新增，14:55 冻结，15:00 清除可买展示 |

完整定义见 [STRATEGY_CONTRACT.md](docs/STRATEGY_CONTRACT.md)。

## 模型与研究

V3 分类器是三个时间窗口上的正则逻辑回归与直方图梯度提升集成。分类概率通过独立的最近 21 个交易日做 Platt 校准；另有均值回归和 10% 分位回归预测净收益及下行风险。

逐票资格同时要求：

- 可成交、流动性、上市时间、价格和涨跌停距离合格；
- 校准盈利概率、模型间保守下界、经验 Wilson 下界和按交易日聚类下界合格；
- 所在校准概率区间同时有足够股票数和独立交易日；
- 期望净收益和下行分位合格；
- 模型间分歧不超过上限；
- 市场数据年龄不超过 7 分钟。

算法、特征和限制见 [WP_V3_MODEL_CARD.md](docs/WP_V3_MODEL_CARD.md)。三年回测协议见 [WP_V3_BACKTEST_PROTOCOL.md](docs/WP_V3_BACKTEST_PROTOCOL.md)。

## 三条生产链路

1. `Run WP V3 Tail Session`：14:16 启动一个持续进程，锚定七个信号槽、14:55 冻结和 15:00 清场。
2. `Validate WP Close`：15:08-16:10 轮询 T+1 收盘真值，逐票结算并更新模型登记册。
3. `Research WP V3 Three-Year Model`：构建三年 point-in-time 数据，执行时间滚动样本外回测，登记影子模型。

上游仓库不再阻塞实时推理。实时和历史数据均由 WP 自己使用 Tushare 构建并执行完整性检查。

## 本地验证

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src:. .venv/bin/pytest -q
```

三年研究需要 `TUSHARE_TOKEN`：

```bash
PYTHONPATH=src .venv/bin/python scripts/build_wp_v3_history.py
PYTHONPATH=src .venv/bin/python scripts/run_wp_v3_research.py
```

## 关键输出

```text
outputs/html_reports/latest.html
outputs/json/wp_manifest.json
outputs/json/wp_model_registry_v3.json
outputs/json/wp_v3_candidate_ledger.json
outputs/json/wp_v3_historical_replay.json
outputs/csv/wp_v3_live_predictions.csv
outputs/csv/wp_buy_plan_validation.csv
artifacts/wp_v3_research/wp_v3_backtest.json
artifacts/wp_v3_research/models/<fingerprint>.joblib
```

GitHub Pages：

```text
https://njedu2023-prog.github.io/WP/outputs/html_reports/latest.html
```

系统只提供人工决策支持，不连接券商、不自动下单，也不保证收益。
