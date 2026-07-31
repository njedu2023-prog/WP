# WP V25 Positioning Rank Research Result

## Decision

V25 is rejected. It is not authorized for the required 150-trading-day future
shadow run or for production.

The prior-day holder-cost, financing-position, and abnormal-trading features
contained a weak same-day same-slot ranking signal, but the signal was below
the preregistered economic floor and did not produce a profitable executable
release policy. This is a valid negative result, not a data, coverage, or
workflow failure.

## Immutable Evidence

- Repository: `njedu2023-prog/WP`
- Branch: `main`
- Preregistered model commit:
  `5c1b93e82db9286064d97c0cc2035fd751fe6523`
- Signal-price repair and data-build commit:
  `3259e4abbefd7640219e1fec0d8b62931f5b166e`
- Research-trigger commit:
  `abaa8ead788a63b8a65b1766ae6a7f5a3e5713a6`
- Immutable V9 source run: `30600193544`
- V24 point-in-time data run: `30635569735`
- V25 positioning data run: `30651844515`
- V25 positioning data artifact: `8801757186`
- V25 positioning data artifact name:
  `wp-v25-positioning-data-30651844515`
- V25 positioning data artifact digest:
  `sha256:7eb73901c9ad1d40fd8adb4619db953dc3e839272a69ad61d45f4e10b76b1557`
- V25 research run: `30652177604`
- V25 research job: `91227843609`
- Research artifact: `8801943931`
- Research artifact name:
  `wp-v25-positioning-rank-30652177604`
- Research artifact digest:
  `sha256:52226a062ebcd3b2bc1eb116f7e8f0d1e5cc74be579329da207b26f9a44554e7`
- Research artifact size: `15,129,968` bytes
- Evaluation trading days: `725`
- Model-covered trading days: `725`
- Outcome-blind source rows: `31,955`
- Joined point-in-time rows: `31,955`
- Verified selected outcomes: `72 / 72`
- Missing selected outcomes: `0`
- Inconsistent selected outcomes: `0`

## Data Contract Result

The full positioning dataset passed its frozen outcome-blind coverage
contract:

- candidate rows: `31,955`
- source trading days: `913`
- holder-cost coverage: `99.4962%`
- financing-position coverage: `93.9790%`
- prior-day abnormal-trading event rate: `3.0981%`
- holder-cost query failures: `0`
- financing-position query failures: `0`
- abnormal-trading query failures: `0`

The result therefore cannot be attributed to missing positioning data.

## Nested Out-Of-Sample Result

| Metric | Frozen requirement | Result | Pass |
| --- | ---: | ---: | :---: |
| Candidates | at least 120 | 72 | no |
| Candidate days | at least 80 | 67 | no |
| Candidate-day rate | 12%-35% | 9.2414% | no |
| Win rate | at least 55% | 41.6667% | no |
| Wilson win lower bound | at least 50% | 30.9852% | no |
| Clustered win lower bound | at least 48% | 28.9474% | no |
| Return above +0.50% | at least 40% | 29.1667% | no |
| Return at or below -2.00% | at most 15% | 20.8333% | no |
| Mean net return | at least +0.20% | -0.4311% | no |
| Clustered mean lower bound | above 0% | -0.8908% | no |
| Profit factor | at least 1.20 | 0.5829 | no |
| Additional 50bp stress mean | at least 0% | -0.9311% | no |
| Return 10th percentile | at least -3.00% | -3.3966% | no |
| Active calendar years | at least 3 | 4 | yes |
| Candidates per active year | at least 20 | 2 minimum | no |
| Positive calendar years | at least 3 | 1 | no |
| Worst calendar-year mean | at least -0.10% | -0.6433% | no |

Median net return was `-0.4409%`. The one-sided mean-return test produced
`p=0.9141`, so there is no statistical evidence that the true executable mean
is positive.

## Same-Slot Ranking Result

| Metric | Frozen requirement | Result | Pass |
| --- | ---: | ---: | :---: |
| Evaluable date-slot groups | at least 1,000 | 5,075 | yes |
| Mean same-slot Spearman IC | at least +0.05 | +0.0354 | no |
| Highest-minus-lowest return | at least +0.20% | +0.1666% | no |
| Clustered spread lower bound | above 0% | +0.0410% | yes |
| Positive spread years | at least 3 | 4 | yes |

The positioning family is not pure noise: the top-ranked stock beat the
bottom-ranked stock by `+0.1666%` on average. However, that is below the
precommitted `+0.20%` economic floor, the IC is below `+0.05`, and the
selective release policy remained deeply negative after costs. The weak rank
information is insufficient to authorize threshold tuning, subgroup mining,
or deployment.

## Calendar-Year Stability

| Year | Candidates | Candidate days | Win rate | Mean net return | Profit factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | 2 | 1 | 100.00% | +0.2707% | N/A |
| 2024 | 42 | 41 | 50.00% | -0.4627% | 0.5532 |
| 2025 | 5 | 5 | 0.00% | -0.6433% | 0.0000 |
| 2026 | 23 | 20 | 30.43% | -0.3883% | 0.6776 |

The only positive year contained two stocks on one trading day. Every year
with a meaningful number of candidates was negative.

## What V25 Established

- Prior-day positioning data can be acquired with adequate historical coverage
  and point-in-time timing.
- These features contain weak relative information inside the same live
  date-slot opportunity set.
- The weak relative signal does not translate into a profitable release
  policy under immutable first-signal entry, fixed T+1 close exit, and costs.
- Candidate scarcity is not the main defect: the released candidates had
  negative mean, low win rate, poor profit factor, and excessive tail losses.
- The final 42-day calibration window contained no fully eligible day, so the
  frozen current policy had no valid release threshold.

## Closed Decisions

- Do not deploy or shadow V25.
- Do not relax V25 gates.
- Do not tune V25 thresholds, weights, subgroups, years, or feature signs after
  reading this result.
- Do not combine V25 with V24 after observing both frozen test outcomes and
  present the combination as confirmatory evidence.
- Do not describe the positive same-slot spread as a profitable strategy.

Any continuation must preregister a genuinely independent point-in-time
information family and produce a new immutable result.
