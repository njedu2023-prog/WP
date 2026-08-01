# WP V35 Full-Session Regime Research Result

## Decision

V35 is permanently rejected.

It must not enter shadow or production. Its features, score, threshold, basket,
or subgroups must not be retuned against this result.

## Immutable Evidence

- protocol: `WP_V35_FULL_SESSION_REGIME_PROTOCOL`
- V9 source run: `30600193544`
- V24 point-in-time data run: `30635569735`
- V34 full-session path data run: `30677075531`
- completed V35 research run: `30683037166`
- research job: `91323597809`
- evidence artifact: `8813016157`
- evaluation days: 725
- source and joined candidate rows: 31,955
- causal slot rows: 3,186
- model-covered outer-test days: 683

All selected rows had verified outcomes. Temporal, source, feature, data,
signal-time, and signal-price integrity checks passed.

## Nested Out-of-Sample Result

| Metric | Result | Required |
| --- | ---: | ---: |
| Candidates | 264 | at least 180 |
| Candidate days | 106 | at least 100 |
| Candidate-day rate | 14.62% | 12% to 28% |
| Win rate | 42.42% | at least 55% |
| Wilson lower win bound | 36.61% | at least 50% |
| Day-clustered lower win bound | 34.72% | at least 48% |
| Mean net return | -0.2867% | at least +0.20% |
| Day-clustered lower mean | -0.6398% | above 0% |
| Profit factor | 0.6965 | at least 1.20 |
| Mean after additional 50 bp stress | -0.7867% | at least 0% |
| Return 10th percentile | -2.7018% | at least -3.00% |
| Margin-hit rate | 28.41% | at least 35% |
| Tail-loss rate | 16.29% | at most 20% |

The strategy failed 11 historical gates. Only candidate count, opportunity
frequency, return p10, tail-loss rate, calendar coverage, and integrity gates
passed.

## Calendar Stability

| Year | Candidates | Win rate | Mean net return |
| --- | ---: | ---: | ---: |
| 2023 | 28 | 32.14% | -0.7298% |
| 2024 | 90 | 47.78% | -0.2021% |
| 2025 | 90 | 46.67% | +0.1164% |
| 2026 | 56 | 32.14% | -0.8489% |

Only one calendar year had a positive mean, and even that year's additional
50 bp stress result was negative.

## Engineering Incident

The first run, `30682671156`, exposed an implementation-only availability
error before a complete result could be produced:

- undocumented row floors of 1,000 training slots and 200 calibration slots
  overrode the frozen 252-day and 42-day windows;
- the script attempted to fit a final bundle before deciding whether the
  historical gates passed.

The correction derived row floors directly from the frozen day windows
(`252` and `42`) and generated a final bundle only after all historical gates
passed. No feature, label, model, policy, threshold, candidate, return, cost,
or acceptance rule changed. The corrected code passed 32 relevant tests.

## Interpretation

Cross-stock agreement of the existing full-session path features does not
license profitable tail slots. The signal is directionally negative after
costs and unstable across years.

V35 is terminal. Further work must introduce a genuinely new causal
information family or a different executable trade contract. Reusing these
features with looser thresholds, favorable dates, selected years, or
post-result subgroups is prohibited.
