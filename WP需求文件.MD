# WP 项目开发需求文档：次日涨停概率排序系统

## 0. 项目目标

开发一个独立项目 **WP**，主程序放在：

```text
https://github.com/njedu2023-prog/WP
```

数据基础设施复用并扩展：

```text
https://github.com/njedu2023-prog/a-share-top3-data
```

WP 的核心目标是：

> 从 A 股当日盘中/收盘候选股中，筛选出“涨幅已经超过 6%、前一日没有涨停”的股票，并按照“第二天涨停概率”从高到低排序，输出前 50 名 HTML 报告页面。

系统应具备：

1. 自动拉取 a-share-top3-data 的数据；
2. 自动识别符合条件的候选股票；
3. 计算第二天涨停概率；
4. 输出前 50 名排序；
5. 每 10 分钟刷新一次数据链路；
6. 每 10 分钟更新一次 HTML 报告页面；
7. 具备历史验证、自学习、日志、异常兜底能力。

---

# 一、项目定位

## 1.1 项目名称

```text
WP - 次日涨停概率排序系统
```

## 1.2 项目用途

用于盘中或收盘后辅助人工判断：

```text
哪些股票第二天最可能涨停？
```

不是普通涨幅榜，也不是简单强势股榜，而是专门识别：

```text
今日已经明显走强，但还没有涨停，且昨日不是涨停板的股票。
```

这类股票通常属于：

1. 板块启动中的前排票；
2. 强资金开始抢筹的票；
3. 可能从趋势强票转化为次日涨停票；
4. 具备“今日未涨停，明日冲板/涨停”潜力的票。

---

# 二、核心筛选条件

WP 候选池必须同时满足以下条件。

## 2.1 今日涨幅条件

```text
今日涨幅 >= 6%
```

字段来源优先级：

1. `pct_chg`
2. `change_pct`
3. `涨跌幅`
4. 根据 `close / pre_close - 1` 计算

要求：

```text
pct_chg >= 6.00
```

如果字段缺失，需要在程序中自动计算。

---

## 2.2 前一日没有涨停

候选股前一个交易日不得为涨停。

判断方式：

```text
前一交易日 close < 前一交易日 limit_up_price
```

或者：

```text
前一交易日 is_limit_up != 1
```

要求：

1. 如果已有 `is_limit_up` 字段，优先使用；
2. 如果没有，需要根据涨停价计算；
3. ST、创业板、科创板、北交所涨停规则需要区分；
4. 前一日一字板、T字板、实体涨停都视为“前一日涨停”，必须排除。

---

## 2.3 今日不能已经涨停

WP 不是涨停接力系统，不是找“今天已经涨停的票”，而是找：

```text
今日涨幅超过 6%，但今日尚未涨停，第二天有机会涨停的票。
```

因此建议增加硬过滤：

```text
今日未涨停
```

判断条件：

```text
today_close < today_limit_up_price
```

或者：

```text
is_limit_up_today != 1
```

如果用户后续希望保留“今日已涨停票”，可以另外扩展为 `WP-LimitUp` 分支；但当前 WP 主线建议只做：

```text
6% 以上强势未涨停票 → 预测次日涨停概率
```

---

## 2.4 股票池范围

默认覆盖：

```text
沪深 A 股主板、创业板、科创板、中小板历史代码、北交所可选
```

建议默认排除：

1. ST / *ST；
2. 退市整理；
3. 停牌；
4. 新股上市未满 10 个交易日；
5. 当日成交额过低的极端流动性票；
6. 无法计算涨停价的异常票；
7. 前复权数据缺失票；
8. 数据字段严重缺失票。

建议基础流动性门槛：

```text
amount >= 1亿元
```

如果盘中运行，可使用实时成交额；如果收盘后运行，用日成交额。

---

# 三、数据来源设计

## 3.1 上游仓库

```text
a-share-top3-data
```

WP 不单独重新造数据底座，而是复用 a-share-top3-data 的基础设施。

需要在 a-share-top3-data 中新增目录：

```text
data/wp/
```

建议目录结构如下：

```text
a-share-top3-data/
└── data/
    └── wp/
        ├── raw/
        │   └── YYYY/
        │       └── YYYYMMDD/
        │           ├── wp_snapshot_0930.csv
        │           ├── wp_snapshot_0940.csv
        │           ├── wp_snapshot_0950.csv
        │           ├── ...
        │           └── wp_snapshot_1500.csv
        ├── features/
        │   └── YYYY/
        │       └── YYYYMMDD/
        │           └── wp_features.csv
        ├── candidates/
        │   └── YYYY/
        │       └── YYYYMMDD/
        │           └── wp_candidates.csv
        ├── labels/
        │   └── YYYY/
        │       └── YYYYMMDD/
        │           └── wp_labels.csv
        ├── reports_input/
        │   └── YYYY/
        │       └── YYYYMMDD/
        │           └── wp_rank_input.csv
        └── latest/
            ├── wp_latest_snapshot.csv
            ├── wp_latest_features.csv
            ├── wp_latest_candidates.csv
            ├── wp_latest_rank_input.csv
            └── wp_manifest.json
```

---

## 3.2 a-share-top3-data 需要新增的数据任务

在 a-share-top3-data 中新增 WP 数据生产脚本：

```text
scripts/wp/
├── build_wp_snapshot.py
├── build_wp_features.py
├── build_wp_candidates.py
├── build_wp_labels.py
├── build_wp_rank_input.py
└── wp_data_healthcheck.py
```

---

## 3.3 GitHub Actions 新增数据流

在 a-share-top3-data 中新增 workflow：

```text
.github/workflows/run_wp_data_10min.yml
```

运行频率：

```yaml
schedule:
  - cron: "*/10 1-7 * * 1-5"
```

注意：GitHub Actions 使用 UTC 时间。A 股交易时间为北京时间：

```text
09:30 - 11:30
13:00 - 15:00
```

北京时间 UTC+8，因此 UTC 对应：

```text
01:30 - 03:30
05:00 - 07:00
```

建议 workflow 内部增加交易日历判断，不是交易日直接退出。

---

# 四、WP 主程序仓库设计

## 4.1 WP 仓库目录结构

```text
WP/
├── .github/
│   └── workflows/
│       └── run_wp_10min.yml
├── config/
│   ├── wp_config.yml
│   ├── model_weights.yml
│   └── risk_rules.yml
├── data/
│   ├── input/
│   ├── output/
│   └── cache/
├── docs/
│   ├── WP_系统说明.md
│   ├── WP_字段说明.md
│   └── WP_模型说明.md
├── outputs/
│   ├── csv/
│   │   ├── wp_top50.csv
│   │   ├── wp_full_rank.csv
│   │   └── wp_model_debug.csv
│   ├── html_reports/
│   │   ├── latest.html
│   │   └── archive/
│   └── json/
│       ├── latest.json
│       └── wp_manifest.json
├── src/
│   └── wp/
│       ├── __init__.py
│       ├── main.py
│       ├── data_loader.py
│       ├── calendar.py
│       ├── candidate_filter.py
│       ├── feature_engineering.py
│       ├── scoring_model.py
│       ├── limitup_probability.py
│       ├── risk_penalty.py
│       ├── ranking.py
│       ├── report_html.py
│       ├── report_md.py
│       ├── validation.py
│       ├── backtest.py
│       └── utils.py
├── tests/
│   ├── test_candidate_filter.py
│   ├── test_limitup_probability.py
│   ├── test_ranking.py
│   └── test_report_html.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# 五、WP 每 10 分钟自动刷新机制

## 5.1 WP 主程序 workflow

新增：

```text
.github/workflows/run_wp_10min.yml
```

运行频率：

```yaml
schedule:
  - cron: "*/10 1-7 * * 1-5"
```

要求：

1. 每 10 分钟运行一次；
2. 自动读取 a-share-top3-data 最新 WP 数据；
3. 生成最新排序；
4. 输出 HTML 报告；
5. 自动提交到 WP 仓库；
6. GitHub Pages 自动展示最新页面。

---

## 5.2 运行逻辑

每次运行流程：

```text
1. 判断是否为 A 股交易日
2. 判断当前时间是否在有效运行时间窗口
3. 拉取 a-share-top3-data 最新数据
4. 读取 wp_latest_rank_input.csv
5. 执行候选池过滤
6. 执行特征工程
7. 执行涨停概率模型
8. 执行风险惩罚
9. 输出 full_rank
10. 截取 top50
11. 生成 CSV / JSON / HTML
12. 写入 outputs/
13. 自动 commit
14. GitHub Pages 展示 latest.html
```

---

# 六、模型设计

WP 模型目标：

```text
预测股票下一个交易日涨停概率 P_limitup_T1
```

最终排序字段：

```text
wp_score
```

排序逻辑：

```text
wp_score = P_limitup_T1 * 100 + 强资金加分 + 强板块加分 + 强个股加分 - 风险惩罚
```

---

# 七、核心因子体系

## 7.1 强资金板块因子

这是 WP 的第一核心。

字段建议：

```text
sector_name
sector_pct_chg
sector_rank
sector_limitup_count
sector_gt6_count
sector_amount
sector_amount_ratio
sector_net_inflow
sector_turnover
sector_hot_score
```

计算逻辑：

```text
强资金板块分 = 板块涨幅强度 + 板块涨停家数 + 板块6%以上家数 + 板块成交额放大 + 板块资金净流入 + 板块排名稳定性
```

建议评分：

```text
sector_strength_score: 0 - 100
```

强板块定义建议：

```text
sector_rank <= 10
且 sector_gt6_count >= 3
且 sector_amount_ratio >= 1.2
```

---

## 7.2 强个股因子

字段建议：

```text
ts_code
name
trade_date
price
pct_chg
amount
volume
turnover_rate
volume_ratio
amplitude
high
low
open
close
pre_close
intraday_high_pct
close_position
```

重点判断：

1. 今日涨幅是否强；
2. 成交额是否明显放大；
3. 收盘是否靠近全天高点；
4. 是否有明显资金承接；
5. 是否不是尾盘硬拉；
6. 是否不是高开低走；
7. 是否量价结构健康。

建议评分：

```text
stock_strength_score: 0 - 100
```

---

## 7.3 资金承接因子

字段建议：

```text
amount_ratio_5d
amount_ratio_20d
turnover_rate
turnover_rate_5d_avg
close_near_high
late_pullback_pct
intraday_vwap_position
```

重点逻辑：

```text
上涨不是问题，关键是有没有人接。
```

高分结构：

1. 成交额放大但不过度爆量；
2. 收盘价靠近高点；
3. 回落小；
4. 换手充分但不失控；
5. 量价同步。

低分结构：

1. 尾盘偷袭拉升；
2. 高开低走；
3. 放巨量滞涨；
4. 冲高回落；
5. 板块强但个股掉队。

建议评分：

```text
acceptance_score: 0 - 100
```

---

## 7.4 次日涨停概率因子

核心输出字段：

```text
p_limitup_t1
```

基础模型可以先用规则模型，后续升级机器学习模型。

### 第一阶段：规则评分模型

```text
p_limitup_t1 = sigmoid(
    w1 * sector_strength_score
  + w2 * stock_strength_score
  + w3 * acceptance_score
  + w4 * momentum_score
  + w5 * capital_score
  - w6 * risk_penalty
)
```

输出范围：

```text
0 - 100%
```

---

### 第二阶段：历史统计模型

用历史数据训练：

```text
今日满足：
涨幅 >= 6%
前一日未涨停
今日未涨停

观察：
下一交易日是否涨停
```

标签：

```text
label_t1_limitup = 1 / 0
```

训练维度：

1. 板块强度；
2. 个股涨幅；
3. 成交额放大；
4. 换手率；
5. 收盘位置；
6. 前 5 日涨跌；
7. 前 20 日涨跌；
8. 是否突破平台；
9. 是否创阶段新高；
10. 是否有龙虎榜；
11. 是否有异动公告；
12. 是否属于当日热门题材；
13. 是否前一日涨停；
14. 是否今日涨停；
15. 是否尾盘拉升。

---

## 7.5 动量因子

字段建议：

```text
ret_3d
ret_5d
ret_10d
high_20d_break
ma5_position
ma10_position
ma20_position
```

高分：

1. 短线趋势向上；
2. 今日突破平台；
3. 不是连续暴涨后的末端；
4. 5日线、10日线结构向上。

低分：

1. 已连续大涨多日；
2. 高位放量滞涨；
3. 距离均线过远；
4. 明显加速末端。

---

## 7.6 风险惩罚因子

必须加入风险惩罚，否则 WP 会把很多“尾盘硬拉、明天接不住”的票排到前面。

风险项包括：

```text
risk_penalty = 
  高位加速风险
+ 尾盘偷袭风险
+ 爆量滞涨风险
+ 冲高回落风险
+ 板块后排风险
+ 前期套牢压力风险
+ 异常公告风险
+ ST/退市/停牌风险
+ 流动性不足风险
```

输出：

```text
risk_penalty_score: 0 - 100
```

---

# 八、最终评分模型

建议最终字段：

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

然后映射为：

```text
p_limitup_t1
```

排序优先级：

```text
1. p_limitup_t1 降序
2. wp_score 降序
3. sector_strength_score 降序
4. acceptance_score 降序
5. amount 降序
```

---

# 九、输出字段要求

## 9.1 wp_top50.csv 字段

```text
rank
ts_code
name
trade_date
update_time
price
pct_chg
pre_day_limitup
today_limitup
sector_name
sector_strength_score
stock_strength_score
acceptance_score
momentum_score
capital_score
risk_penalty_score
p_limitup_t1
wp_score
model_confidence
signal_level
core_reason
risk_reason
```

---

## 9.2 HTML 报告字段

HTML 页面标题：

```text
WP 次日涨停概率 Top50
```

页面说明：

```text
筛选条件：今日涨幅 >= 6%，前一日未涨停，今日未涨停。
目标：预测下一交易日涨停概率。
刷新频率：每 10 分钟。
```

主表字段：

```text
排名
代码
名称
当前价/收盘价
今日涨幅
所属板块
板块强度
个股强度
承接分
次日涨停概率
WP评分
模型置信度
信号等级
核心理由
风险提示
更新时间
```

---

## 9.3 信号等级

```text
S级：p_limitup_t1 >= 45%，且风险分 <= 30
A级：p_limitup_t1 >= 35%，且风险分 <= 40
B级：p_limitup_t1 >= 25%，且风险分 <= 50
C级：p_limitup_t1 >= 15%
D级：低参考价值
```

注意：

```text
S级不等于一定涨停，只代表模型排序中的最高优先级。
```

---

# 十、HTML 页面要求

## 10.1 页面风格

要求：

1. 中文界面；
2. 表格清晰；
3. 支持横向滚动；
4. 内容不要强制换行；
5. 适配手机端；
6. 前 10 名高亮；
7. 风险高的票用明显风险标识；
8. 页面顶部显示更新时间；
9. 页面顶部显示数据状态；
10. 页面底部显示模型说明。

---

## 10.2 页面模块

HTML 报告至少包含：

```text
1. 今日运行状态
2. 数据更新时间
3. 候选池数量
4. 入选 Top50 数量
5. 板块热度 Top10
6. WP 次日涨停概率 Top50
7. 模型解释
8. 风险提示
9. 数据健康检查
```

---

# 十一、数据健康检查

必须输出：

```text
wp_data_healthcheck.json
```

检查内容：

1. 是否交易日(严格A股交易日)；
2. 当前数据时间；
3. 股票数量是否正常；
4. 涨幅字段是否存在；
5. 昨日涨停字段是否存在；
6. 今日涨停字段是否存在；
7. 板块字段是否存在；
8. 成交额字段是否存在；
9. 候选池数量是否异常；
10. Top50 是否为空；
11. 是否使用 fallback 逻辑。

如果异常，HTML 页面必须显示：

```text
数据异常 / 数据不完整 / 使用降级逻辑
```

不能静默失败。

---

# 十二、历史验证与自学习

## 12.1 标签定义

每日生成标签：

```text
label_t1_limitup
```

含义：

```text
今日 WP 候选股，下一交易日是否涨停。
```

字段：

```text
trade_date
ts_code
name
p_limitup_t1
wp_score
next_trade_date
next_day_high
next_day_close
next_day_limitup_price
label_t1_limitup
next_day_max_pct
next_day_close_pct
```

---

## 12.2 验证指标

每个交易日收盘后或下一交易日收盘后验证：

```text
Top10 命中率
Top20 命中率
Top50 命中率
AUC
Brier
Precision@10
Precision@20
Precision@50
平均次日最高涨幅
平均次日收盘涨幅
最大回撤风险
```

---

## 12.3 自学习机制

初始阶段使用规则模型。

当历史样本足够后，升级为：

```text
规则模型 + 统计校准 + 机器学习模型
```

建议机器学习模型：

```text
LogisticRegression
RandomForest
LightGBM / XGBoost 可选
```

初始版本不强制依赖复杂模型，必须保证系统先跑通。

---

# 十三、WP 与 a-share-top3-data 的边界

## 13.1 a-share-top3-data 负责

```text
1. 获取基础行情数据
2. 获取板块数据
3. 获取涨停价/昨日涨停状态
4. 获取成交额、换手率、涨幅等基础字段
5. 生成 WP 原始数据
6. 生成 WP 特征输入
7. 生成 latest 数据文件
```

---

## 13.2 WP 负责

```text
1. 读取 a-share-top3-data 的 latest 数据
2. 二次过滤候选池
3. 计算次日涨停概率
4. 风险惩罚
5. 排序
6. 输出 Top50
7. 生成 HTML 报告
8. 生成历史归档
9. 生成验证报告
```

---

# 十四、必须生成的输出文件

WP 每次运行后必须生成：

```text
outputs/csv/wp_top50.csv
outputs/csv/wp_full_rank.csv
outputs/csv/wp_model_debug.csv
outputs/json/latest.json
outputs/json/wp_manifest.json
outputs/html_reports/latest.html
```

HTML 归档：

```text
outputs/html_reports/archive/YYYYMMDD/HHMM.html
```

---

# 十五、GitHub Pages

WP 仓库需要支持 GitHub Pages。

默认页面：

```text
https://njedu2023-prog.github.io/WP/outputs/html_reports/latest.html
```

也可以额外生成：

```text
https://njedu2023-prog.github.io/WP/
```

首页自动跳转到 latest.html。

---

# 十六、容错要求

程序不得因为单个字段缺失直接崩溃。

必须具备 fallback：

1. 板块数据缺失：板块分降级为中性分；
2. 昨日涨停字段缺失：尝试用涨停价计算；
3. 涨停价缺失：尝试按市场规则估算；
4. 成交额缺失：该票风险升高；
5. 候选池为空：HTML 正常生成，但显示“无符合条件股票”；
6. 数据源无法读取：保留上一版 latest.html，并生成错误日志。

---

# 十七、日志要求

每次运行输出日志：

```text
logs/wp_YYYYMMDD_HHMM.log
```

日志内容：

```text
运行时间
是否交易日
数据源路径
读取股票数量
候选池数量
Top50数量
异常字段
fallback使用情况
最终输出路径
```

---

# 十八、模型解释字段

每只股票必须生成：

```text
core_reason
risk_reason
```

示例：

```text
core_reason:
所属板块排名靠前，板块内多只股票涨幅超过6%，个股成交额明显放大，收盘接近全天高点，资金承接较强。

risk_reason:
短线涨幅偏高，若次日板块分化，存在冲高回落风险。
```

---

# 十九、第一阶段开发目标

第一阶段不追求复杂机器学习，先完成稳定可运行版本。

## 19.1 第一阶段必须完成

```text
1. a-share-top3-data 新增 data/wp/ 数据目录
2. a-share-top3-data 新增 WP 数据生产脚本
3. WP 仓库建立完整主程序
4. 支持每 10 分钟自动运行
5. 支持候选池过滤
6. 支持规则评分模型
7. 支持 Top50 输出
8. 支持 HTML 报告
9. 支持 GitHub Pages 展示
10. 支持数据健康检查
11. 支持日志
```

---

## 19.2 第一阶段可以暂缓

```text
1. 复杂机器学习模型
2. 深度回测看板
3. 多模型融合
4. 实盘交易接口
5. 自动下单
```

注意：

```text
WP 当前只做决策辅助，不做自动交易。
```

---

# 二十、验收标准

## 20.1 数据验收

必须看到：

```text
a-share-top3-data/data/wp/latest/wp_latest_rank_input.csv
```

且字段完整。

---

## 20.2 程序验收

WP 成功生成：

```text
outputs/csv/wp_top50.csv
outputs/html_reports/latest.html
```

---

## 20.3 页面验收

页面必须显示：

```text
WP 次日涨停概率 Top50
```

并且至少包含：

```text
排名
代码
名称
今日涨幅
所属板块
次日涨停概率
WP评分
信号等级
核心理由
风险提示
更新时间
```

---

## 20.4 逻辑验收

Top50 中所有股票必须满足：

```text
今日涨幅 >= 6%
前一日没有涨停
今日没有涨停
```

任何不满足条件的股票不得进入 Top50。

---

## 20.5 自动化验收

GitHub Actions 能够：

```text
每 10 分钟自动运行
自动更新 latest.html
自动提交 outputs
```

---

# 二十一、给 Codex 的开发执行要求

请 Codex 按以下顺序开发：

```text
第一步：在 a-share-top3-data 新增 WP 数据目录和数据生产脚本
第二步：在 WP 仓库建立主程序目录结构
第三步：实现数据读取和候选池过滤
第四步：实现规则评分模型
第五步：实现 Top50 排序输出
第六步：实现 HTML 报告
第七步：实现 GitHub Actions 每 10 分钟运行
第八步：实现数据健康检查和日志
第九步：自检 Top50 是否严格满足条件
第十步：提交完整代码并说明修改文件
```

---

# 二十二、特别注意事项

1. 不允许把“前一日涨停票”放进候选池；
2. 不允许把“今日已经涨停票”放进当前 WP 主线；
3. 不允许只按涨幅排序；
4. 不允许只按成交额排序；
5. 必须体现“强资金板块 + 强个股 + 承接质量 + 风险惩罚”；
6. 每 10 分钟刷新一次；
7. HTML 报告必须同步更新；
8. 任何数据异常必须在页面显示；
9. 模型输出必须有解释，不要只给分数；
10. 当前系统只做辅助决策，不做自动交易。

---

# 二十三、最终一句话目标

WP 要做成：

```text
A 股盘中/收盘后，专门寻找“今日涨幅超过 6%、昨日未涨停、今日未涨停、明日最可能涨停”的 Top50 排序系统。
```

核心不是找已经涨停的票，而是提前识别：

```text
明天可能冲板、封板、成为强势股的潜在标的。
```
