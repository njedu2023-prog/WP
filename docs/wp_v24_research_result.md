# WP V24 Cross-Section Microstructure Research Result

## Decision

V24 is rejected. It is not authorized for the required 150-day future shadow
run or for production.

The wider top-five opportunity set solved the historical opportunity-frequency
problem, but the frozen cross-section microstructure model did not produce
positive executable economics under the fixed first-signal entry and T+1 close
exit contract. The result is a substantive negative finding, not a data or
workflow failure.

## Immutable Evidence

- Preregistered protocol commit:
  `78aa6419b378eef7d9528db5ba24b0304e4497ab`
- Data-evidence and research-trigger commit:
  `9b2a595b66d56fb2dd34be7f4506658b9a88e653`
- Immutable V9 source run: `30600193544`
- V24 point-in-time data run: `30635569735`
- V24 research run: `30643568519`
- V24 research job: `91199251576`
- Artifact: `8798903875`
- Artifact name: `wp-v24-cross-section-30643568519`
- Artifact digest:
  `sha256:54fdd2caf23b0f3a72ab51d52cbc9f429e9e4d50462f3fca4d916a07c9f0ac2a`
- Artifact size: `16,190,936` bytes
- Frozen contract tests: `50 passed`
- Evaluation days: `725`
- Model-covered days: `725`
- Outcome-blind source rows: `31,955`
- Joined point-in-time rows: `31,955`
- Verified selected outcomes: `488 / 488`
- Missing selected outcomes: `0`
- Inconsistent selected outcomes: `0`

## Nested Out-of-Sample Result

| Metric | Frozen requirement | Result | Pass |
| --- | ---: | ---: | :---: |
| Candidates | at least 120 | 488 | yes |
| Candidate days | at least 80 | 225 | yes |
| Candidate-day rate | 12%-35% | 31.0345% | yes |
| Win rate | at least 55% | 44.4672% | no |
| Wilson win lower bound | at least 50% | 40.1185% | no |
| Clustered win lower bound | at least 48% | 39.7659% | no |
| Return above +0.50% | at least 40% | 32.1721% | no |
| Return at or below -2.00% | at most 15% | 19.0574% | no |
| Mean net return | at least +0.20% | -0.1629% | no |
| Clustered mean lower bound | above 0% | -0.5735% | no |
| Profit factor | at least 1.20 | 0.8668 | no |
| Additional 50bp stress mean | at least 0% | -0.6629% | no |
| Return 10th percentile | at least -3.00% | -3.7593% | no |
| Active calendar years | at least 3 | 4 | yes |
| Candidates per active year | at least 20 | 49 minimum | yes |
| Positive calendar years | at least 3 | 1 | no |
| Worst calendar-year mean | at least -0.10% | -0.2820% | no |

Median net return was `-0.1801%`. The one-sided mean-return test produced
`p=0.8788`, so there is no statistical evidence that the true executable mean
is positive.

## Calendar-Year Stability

| Year | Candidates | Candidate days | Win rate | Mean net return | Profit factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | 49 | 27 | 32.65% | -0.1612% | 0.8619 |
| 2024 | 129 | 61 | 44.19% | +0.0415% | 1.0354 |
| 2025 | 144 | 69 | 47.22% | -0.2092% | 0.7503 |
| 2026 | 166 | 68 | 45.78% | -0.2820% | 0.8253 |

Only 2024 was positive, and its `+0.0415%` mean was not economically or
statistically sufficient. The most recent two years were both negative.

## What V24 Established

- Expanding the outcome-blind source set from one stock to five per slot can
  produce a useful number of candidates without outcome leakage.
- Candidate scarcity was not the principal reason V23 failed.
- The frozen V24 causal minute, auction, previous-day money-flow, and
  cross-sectional features did not separate profitable T+1-close outcomes.
- The calibration score threshold was unstable across folds, including two
  folds with no released candidates, while 2026 candidate frequency rose to
  `50.75%`. This indicates score-distribution and regime drift.
- Loss frequency and lower-tail severity remained too high. More candidates
  amplified noise rather than improving economic selection.

## Closed Decisions

- Do not deploy or shadow V24.
- Do not relax V24 gates.
- Do not search V24 score weights or thresholds after this result.
- Do not present V24 candidates as a profitable model.

Any continuation must be a separately preregistered hypothesis with a new
model family and a new immutable result. It may use this rejection only to
formulate that hypothesis, not to rewrite V24.
