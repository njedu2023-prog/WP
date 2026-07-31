# WP V23 Microstructure Research Result

## Decision

**Rejected.** V23 does not authorize production use or the 150-trading-day
future shadow stage.

The result is a valid negative result. Temporal integrity, immutable source
integrity, point-in-time data integrity, and selected-outcome completeness all
passed. The frozen policy itself produced too few candidates and negative
economics.

## Immutable Evidence

- Repository: `njedu2023-prog/WP`
- Branch: `main`
- Workflow run: `30633566537`
- Job: `91165485226`
- Source commit: `cae01ade00619f1f3d41f14dd3d601cc24c22d18`
- Immutable V9 source run: `30600193544`
- Point-in-time data run: `30630271759`
- Research artifact: `8794526887`
- Artifact name: `wp-v23-microstructure-30633566537`
- Artifact digest:
  `sha256:e280760c6e3c16075a05d586bc1551e32da8f35a6dad4d820525a25c3d7d3621`
- Completed at: `2026-07-31T13:20:45Z`
- Contract tests: `44 passed`

## Frozen Evaluation

- Evaluation trading days: `725`
- Model-covered trading days: `661`
- Outcome-blind source leader rows: `6,338`
- Joined point-in-time rows: `6,338`
- Selected candidates: `3`
- Candidate days: `2`
- Candidate-day rate: `0.2759%`
- Positive candidates: `1`
- Win rate: `33.3333%`
- Wilson 95% lower bound: `6.1492%`
- Day-clustered win-rate lower bound: `33.3333%`
- Mean net return: `-1.2593%`
- Median net return: `-0.7702%`
- Profit factor: `0.3149`
- Mean net return under an additional 50 bp stress: `-1.7593%`
- Net-return 10th percentile: `-3.9494%`
- Return above `+0.50%`: `33.3333%`
- Return at or below `-2.00%`: `33.3333%`
- Day-clustered mean-return lower bound: `-2.1305%`

The net-return field already applies the frozen T-day entry and fixed T+1
close-exit contract after the established round-trip costs.

## Outcome Audit

- Selected rows: `3`
- Selected days: `2`
- Verified selected outcomes: `3`
- Missing selected outcomes: `0`
- Inconsistent selected outcomes: `0`
- All selected outcomes verified: `true`

No selected candidate was silently removed from the economic metrics.

## Calendar-Year Distribution

| Year | Candidates | Candidate days | Win rate | Mean net return |
|---|---:|---:|---:|---:|
| 2023 | 0 | 0 | 0.0000% | N/A |
| 2024 | 3 | 2 | 33.3333% | -1.2593% |
| 2025 | 0 | 0 | 0.0000% | N/A |
| 2026 | 0 | 0 | 0.0000% | N/A |

The first two outer folds were left unscored because they contained only
`847` and `1,141` complete prior OOS training rows, below the preregistered
minimum of `1,200`. The floor was not relaxed after observing the run.

## Gate Result

Passed integrity gates:

- Temporal integrity
- Immutable source and feature-contract integrity
- Point-in-time data integrity

Failed economic and coverage gates:

- Minimum nested OOS candidates
- Minimum nested OOS candidate days
- Practical candidate-day rate
- Minimum win rate
- Minimum Wilson lower bound
- Minimum clustered win-rate lower bound
- Minimum margin hit rate
- Maximum tail-loss rate
- Minimum mean net return
- Positive clustered mean-return lower bound
- Minimum profit factor
- Nonnegative additional-50-bp stress return
- 10th-percentile loss floor
- Minimum three active calendar years
- Minimum candidates per active year
- Minimum three positive calendar years
- Worst-calendar-year floor

## Interpretation

V23 asked a narrow question: can a conservative microstructure ensemble rescue
the single outcome-blind V9 leader chosen at each slot? The answer is no.

Most calibration windows had no row that simultaneously passed all fixed
probability lower bounds, severe-loss upper bounds, model-disagreement limits,
execution limits, and source-risk limits. Consequently, the frozen economic
threshold was infinite in nearly every scored fold. This is not evidence that
no profitable tail-market opportunity exists. It is evidence that the V23
combination of a one-leader opportunity set and stacked conservative gates is
not a usable strategy.

V23 is closed. Its thresholds, metrics, and conclusion must not be revised
after this result.
