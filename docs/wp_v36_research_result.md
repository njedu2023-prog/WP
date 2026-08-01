# WP V36 Post-Alert Confirmation Research Result

## Decision

V36 is permanently rejected.

It must not enter shadow or production. Its confirmation features, model,
policy, thresholds, basket, or favorable calendar subsets must not be retuned
against this result.

## Immutable Evidence

- protocol: `WP_V36_POST_ALERT_CONFIRMATION_PROTOCOL`
- V9 source run: `30600193544`
- V24 point-in-time data run: `30635569735`
- V34 one-minute data run: `30677075531`
- completed V36 research run: `30684193574`
- research job: `91326791988`
- evidence artifact: `8813382855`
- artifact digest:
  `sha256:240dd6b4bc610b73f6aa4d6173fa06166096b177b1e0e07d51d9bf574877a370`
- evaluation days: 725
- source candidate rows: 31,955
- legal 14:20-14:45 base rows: 27,390
- model-covered outer-test days: 703

All 27,390 legal base rows received complete causal confirmation features.
Every confirmation used only minutes `t+1` through `t+4`, while the immutable
entry remained the existing `t+5` close plus 10 bp slippage. Entry-price parity
and selected-outcome verification were both 100%.

## Nested Out-of-Sample Result

| Metric | Result | Required |
| --- | ---: | ---: |
| Candidates | 19 | at least 150 |
| Candidate days | 13 | at least 100 |
| Candidate-day rate | 1.79% | 15% to 40% |
| Win rate | 47.37% | at least 55% |
| Wilson lower win bound | 27.33% | at least 50% |
| Day-clustered lower win bound | 11.76% | at least 48% |
| Mean net return | +0.1737% | at least +0.20% |
| Median net return | -0.1718% | diagnostic |
| Day-clustered lower mean | -0.5216% | above 0% |
| Profit factor | 1.3387 | at least 1.20 |
| Mean after additional 50 bp stress | -0.3263% | at least 0% |
| Return 10th percentile | -1.1846% | at least -3.00% |
| Margin-hit rate | 42.11% | at least 35% |
| Tail-loss rate | 5.26% | at most 20% |

The strategy failed 12 historical gates. Profit factor, margin-hit rate,
tail-loss rate, return p10, temporal integrity, source integrity, and data
integrity passed. Sample size, opportunity frequency, win-rate confidence,
mean-return confidence, cost stress, and calendar stability did not.

## Calendar Stability

| Year | Candidates | Win rate | Mean net return |
| --- | ---: | ---: | ---: |
| 2023 | 0 | n/a | n/a |
| 2024 | 2 | 50.00% | -0.0214% |
| 2025 | 6 | 0.00% | -0.5365% |
| 2026 | 11 | 72.73% | +0.5966% |

The apparent aggregate gain came entirely from 2026. The confirmation rule
selected no 2023 candidates and lost in both 2024 and 2025.

## Ranking Diagnostics

- mean within-slot rank IC: `-0.03353`
- mean top-minus-bottom return: `-0.07157%`
- day-clustered lower spread bound: `-0.18513%`

The broader cross-sectional ranking direction was negative. The 19 selected
rows therefore do not reveal a stable positive ordering signal hidden behind
an overly strict threshold.

## Interpretation

Waiting four minutes after an existing alert and entering on the fifth minute
does not create a reliable, reusable edge under the fixed T+1 close contract.
The small positive raw mean is economically fragile, statistically
unresolved, and calendar-concentrated.

V36 is terminal. Further work must use a genuinely independent causal
information family or a different preregistered executable contract. Loosening
this policy, selecting only 2026, or mining these results for favorable
subgroups is prohibited.
