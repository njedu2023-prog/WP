# WP - 次日涨停概率排序系统

WP 从上游 `njedu2023-prog/a-share-top3-data` 的 WP latest 数据读取候选输入，筛选“今日涨幅超过 6%、前一日未涨停、今日未涨停”的 A 股股票，并按次日涨停概率输出 Top50 报告。

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
outputs/backtests/<start>_<end>/daily_summary.csv
outputs/backtests/<start>_<end>/summary.json
outputs/json/wp_backtest_latest.json
outputs/html_reports/backtest_latest.html
```

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
