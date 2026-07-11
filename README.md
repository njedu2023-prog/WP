# WP - 次日涨停概率排序系统

WP 从上游 `njedu2023-prog/a-share-top3-data` 的 WP latest 数据读取候选输入，筛选“今日涨幅超过 8%、前一日未涨停、今日未涨停”的 A 股股票，并按次日涨停概率输出 Top50 报告，同时生成 14:20 尾盘买入观察计划，最多 5 支，并给出确认条件和放弃条件。

当前模型版本为 `wp_rule_v2_1`。系统先计算多因子 `wp_score`，再以 `70% capital_score + 30% wp_score` 形成 `ranking_score`；`p_limitup_t1` 是 `ranking_score` 的单调概率映射，不再维护会打乱排序的第二套隐含权重。行情使用 fallback 时，概率与置信度同步降级。

## 快速运行

```bash
pip install -r requirements.txt
python -m wp.main
```

## 历史区间测试

可手动指定历史时间段，用同一套筛选和排序逻辑跑回测：

```bash
WP_MODE=backtest WP_BACKTEST_START=20260701 WP_BACKTEST_END=20260703 python -m wp.main
```

GitHub Actions 页面点 `Run workflow` 时也可以选择：

```text
mode = backtest
start_date = 20260701
end_date = 20260703
```

回测输出：

```text
outputs/backtests/<start>_<end>/trades.csv
outputs/backtests/<start>_<end>/buy_trades.csv
outputs/backtests/<start>_<end>/daily_summary.csv
outputs/backtests/<start>_<end>/summary.json
outputs/json/wp_backtest_latest.json
outputs/html_reports/backtest_latest.html
```

普通历史回测使用收盘日线代理 14:20 状态，只做方向审计，不进入实时概率校准。只有同模型版本、早于当前日期、标记为 `intraday_1420` 且允许校准的真实快照样本，才会用于保序的 Logit 截距校准。

默认读取：

```text
https://raw.githubusercontent.com/njedu2023-prog/a-share-top3-data/main/data/wp/latest/wp_latest_rank_input.csv
```

输出：

```text
outputs/csv/wp_top50.csv
outputs/csv/wp_full_rank.csv
outputs/csv/wp_model_debug.csv
outputs/json/latest.json
outputs/json/wp_manifest.json
outputs/json/wp_data_healthcheck.json
outputs/html_reports/latest.html
```

GitHub Pages 默认入口：

```text
https://njedu2023-prog.github.io/WP/outputs/html_reports/latest.html
```

本系统只做辅助决策，不做自动交易。
