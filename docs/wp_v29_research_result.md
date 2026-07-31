# WP V29 Hierarchical Peer Research Result

## Decision

V29 is rejected. It is not authorized for the required 150-trading-day future
shadow run or for production.

The selected candidates had a positive arithmetic mean return, but V29 failed
the system's primary probability objective, confidence requirements, cost
stress test, and same-day same-slot stock-selection mechanism. The positive
mean was concentrated in a right tail of winners while the median candidate
lost money. This is a valid negative model result, not a data, coverage, or
workflow failure.

## Immutable Evidence

- Repository: `njedu2023-prog/WP`
- Branch: `main`
- Initial V29 model commit:
  `28247b85110b29e30cfece8a9cf6974c24da3d7a`
- Immutable data-wiring repair commit:
  `0441c416cac5b7f6bb55c036e44ee9b9c3434cab`
- Immutable V24 point-in-time source run: `30635569735`
- V29 research run: `30660952694`
- V29 research job: `91256785568`
- Research artifact: `8805289801`
- Research artifact name:
  `wp-v29-peer-shrinkage-30660952694`
- Research artifact digest:
  `sha256:9147295ab777ff84d50e9bc20b72a8ababebe142484e8bcaa1c92d0725649951`
- Evaluation trading days: `725`
- Model-covered trading days: `725`
- Outcome-blind source rows: `31,955`
- Selected candidates: `350`
- Candidate days: `245`
- Missing selected outcomes: `0`

The first workflow attempt failed before producing a strategy result because
the research workflow omitted the immutable V24 point-in-time quality
artifact. The repair only restored the preregistered source-quality join and
fixed the empty-fold schema. It did not change features, labels, weights,
thresholds, gates, or the exit contract.

## Data Contract Result

The V29 outcome-blind hierarchical peer dataset passed its frozen contract:

- complete rows: `31,955 / 31,955`
- feature coverage: `100%`
- duplicate identities: `0`
- T+1 outcomes read during data construction: `false`

The result therefore cannot be attributed to missing peer features or an
incomplete candidate universe.

## Nested Out-Of-Sample Result

| Metric | Frozen requirement | Result | Pass |
| --- | ---: | ---: | :---: |
| Candidates | at least 120 | 350 | yes |
| Candidate days | at least 80 | 245 | yes |
| Candidate-day rate | 12%-35% | 33.7931% | yes |
| Win rate | at least 55% | 45.4286% | no |
| Wilson win lower bound | at least 50% | 40.2901% | no |
| Clustered win lower bound | at least 48% | 39.2040% | no |
| Return above +0.50% | at least 40% | 34.5714% | no |
| Return at or below -2.00% | at most 15% | 14.8571% | yes |
| Mean net return | at least +0.20% | +0.3230% | yes |
| Clustered mean lower bound | above 0% | -0.1241% | no |
| Profit factor | at least 1.20 | 1.3496 | yes |
| Additional 50bp stress mean | at least 0% | -0.1770% | no |
| Return 10th percentile | at least -3.00% | -2.9006% | yes |

Median net return was `-0.1443%`. The one-sided mean-return test produced
`p=0.1010`; the trade-date-clustered 95% interval was
`[-0.1241%, +0.6347%]`. The evidence does not establish that the true
executable mean is positive.

## Same-Slot Ranking Result

| Metric | Frozen requirement | Result | Pass |
| --- | ---: | ---: | :---: |
| Evaluable date-slot groups | at least 1,000 | 5,075 | yes |
| Mean same-slot Spearman IC | at least +0.05 | -0.0044 | no |
| Highest-minus-lowest return | at least +0.20% | -0.0090% | no |
| Clustered spread lower bound | above 0% | -0.1158% | no |
| Positive spread years | at least 3 | 2 | no |

The clustered spread interval was `[-0.1158%, +0.1056%]`. The yearly
highest-minus-lowest spreads were:

| Year | Same-slot spread |
| --- | ---: |
| 2023 | -0.1963% |
| 2024 | +0.0428% |
| 2025 | -0.0222% |
| 2026 | +0.0694% |

Hierarchical peer confirmation therefore supplied no stable cross-sectional
stock-selection information inside the live opportunity set.

## Calendar-Year Stability

| Year | Candidates | Candidate days | Win rate | Mean net return | Profit factor | 50bp stress mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 30 | 27 | 43.33% | +0.1236% | 1.0931 | -0.3764% |
| 2024 | 105 | 75 | 40.00% | -0.0447% | 0.9598 | -0.5447% |
| 2025 | 143 | 97 | 48.25% | +0.3715% | 1.7068 | -0.1285% |
| 2026 | 72 | 46 | 48.61% | +0.8461% | 1.6646 | +0.3461% |

The later-year improvement does not authorize a recent-period subgroup. The
model was preregistered for the full nested out-of-sample period, 2024 was
negative, and the same-slot mechanism was not positive or stable.

## Failed Frozen Gates

V29 failed these frozen gates:

1. minimum win rate;
2. minimum Wilson win-rate lower bound;
3. minimum trade-date-clustered win-rate lower bound;
4. minimum +0.50% margin hit rate;
5. positive clustered mean-return lower bound;
6. nonnegative additional 50bp stress mean;
7. minimum same-slot rank IC;
8. minimum same-slot highest-minus-lowest spread;
9. positive clustered same-slot spread lower bound;
10. positive same-slot spread in at least three calendar years.

Any one failure rejects V29. The ten failures leave no basis for a shadow or
production exception.

## What V29 Established

- Hierarchical L3-to-L2 peer features can be built point-in-time with complete
  historical coverage.
- The release frequency was practical, so candidate scarcity was not the
  failure.
- Positive arithmetic mean alone is insufficient for the stated objective.
  Fewer than half of the candidates made money and the median candidate lost.
- Industry peer confirmation did not distinguish the better stock from other
  stocks available at the same signal time.
- V29's right-tail winners are not evidence that threshold tuning can recover
  a probability-maximizing strategy.

## Closed Decisions

- Do not deploy or shadow V29.
- Do not relax V29 gates.
- Do not tune V29 weights, thresholds, feature signs, shrinkage strength,
  candidate caps, slots, years, or subgroups after reading this result.
- Do not combine V29 with previously rejected models and present the
  combination as confirmatory evidence.
- Do not describe the positive mean as a profitable or validated strategy.

Any continuation must preregister a genuinely independent point-in-time
information family, prove causal historical availability before reading
outcomes, retain the fixed executable entry and T+1 close exit contracts, and
keep same-day same-slot ranking as an explicit acceptance gate.
