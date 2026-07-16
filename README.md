# WP - 尾盘收益排序系统

WP 在 GitHub Actions 中直连 Tushare 获取当前交易日数据，并复用 `njedu2023-prog/a-share-top3-data` 的同一套数据处理器生成标准输入。系统先从全市场筛选“今日涨幅超过 8%、前一日未涨停、今日未涨停”的 A 股股票，再用尾盘收益模型排序并截取 Top50。14:35 后最多输出 1 支主观察票；没有合格标的时保持空仓。上游仓库的 latest CSV 仅作为直连失败时的显式故障回退，报告会标记 `direct_fallback_used=true`，不会伪装成正常直连结果。

基础特征模型仍为 `wp_rule_v2_1`，尾盘收益排序模型为 `tail_profit_v1`。尾盘模型优先选择涨幅不过热、资金和板块较强、风险较低的股票，并硬性排除涨幅超过 12%、风险分超过 45、收盘位置低于 50、5 日成交额放大超过 2.5 倍的候选。

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

普通历史回测使用收盘日线代理 14:35 状态，只做方向审计，不进入实时概率校准。真实尾盘效果由 14:35 至 14:50 的最终观察名单按下一交易日市场真值累计验证。

实时主输入：

```text
data/direct/latest/wp_latest_rank_input.csv
```

直连阶段要求关键字段完整、历史特征覆盖率达到质量门槛，并记录行情日期、计划时间槽、处理器版本和是否回退。故障回退地址为：

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

本系统只做辅助决策，不做自动交易，也不保证收益。
