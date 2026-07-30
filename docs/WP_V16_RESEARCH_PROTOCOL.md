# WP V16 Research Protocol

## Objective

V16 has one objective: maximize the probability that a candidate first
published between 14:20 and 14:50 can be bought at the recorded executable
signal price and produces a positive net return when sold at the fixed T+1
close contract.

The return includes the configured entry slippage, round-trip costs, and
failed-exit penalty. A day with no qualified candidate is a valid
`NO_SIGNAL` day.

## Frozen Boundary

- V15 remains unchanged and frozen.
- V16 is a separate research challenger.
- Historical success cannot authorize production.
- V16 requires at least 150 future A-share trading days of immutable shadow
  operation after all historical gates pass.

## Causal Input

- Source universe: immutable V11 frontier derived from V9 walk-forward,
  out-of-sample predictions.
- Feature set: T-day information observable no later than each signal slot.
- Signal slots: 14:20, 14:25, 14:30, 14:35, 14:40, 14:45, and 14:50.
- A symbol is recorded at its first qualifying signal. Later appearances
  cannot change the first signal time or price.
- T+1 data is label-only and must never enter a T-day feature.

## Specialist Models

Six overlapping specialists evaluate different market structures:

1. Early tail structure.
2. Late confirmation.
3. Market and industry leadership.
4. Trend persistence.
5. Pullback recovery.
6. Liquidity-confirmed breakout.

Each specialist independently estimates positive-return probability, severe
loss probability, and expected net return. Probabilities are calibrated on a
separate chronological calibration segment.

## Nested Walk-Forward

For every outer test fold:

1. Use at most 504 prior trading days for model fitting, with at least 252.
2. Purge two trading days.
3. Use 42 prior trading days for probability and return calibration.
4. Purge two trading days before the outer test fold.
5. Select policy thresholds only from prior out-of-sample specialist scores.
6. Use 84 days for policy design, purge two days, then use 42 days once for
   confirmation, followed by another two-day purge before the outer test.
7. Freeze the confirmed policy for the untouched outer fold.

If any segment is unavailable or no policy passes, the fold emits
`NO_SIGNAL`.

## Policy Search And Multiple Testing

The policy family is declared in code before results are observed. It covers
probability, expected return, severe-loss probability, fill probability,
expert agreement, model disagreement, cross-sectional rank, daily candidate
count, and early/late slot groups.

- Design mean significance is tested on trade-day mean returns with a five-lag
  Newey-West/HAC variance estimate.
- Benjamini-Hochberg controls the policy-family false discovery rate at
  `q <= 0.10`.
- A five-day circular block bootstrap provides the clustered confidence
  interval.
- Confirmation is run once on a disjoint segment.

## Historical Readiness Gates

All gates must pass:

- At least 250 nested out-of-sample candidates.
- At least 50 candidate trading days.
- Win rate at least 55%.
- Wilson win-rate lower bound at least 52%.
- Five-day block-bootstrap win-rate lower bound at least 52%.
- Mean net return at least 0.20%.
- Trade-day clustered mean lower bound above zero.
- Profit Factor at least 1.20.
- Mean remains non-negative after an additional real 50bp cost per trade.
- Return 10th percentile no worse than -3%.
- Complete temporal-integrity audit.

Passing these gates only permits a future shadow candidate. It does not permit
production use.

## Full T+1 Path Data

V16 separately builds a versioned five-minute T+1 path dataset for every
immutable candidate and target date. Each pair requires the complete session
through 15:00 and a minimum bar-count quality gate. This dataset is for
predeclared exit-contract research only; any path-dependent exit must execute
on the next observable bar and must be revalidated through the same nested
protocol.

## Required Artifacts

- Source digests and data-quality manifest.
- Candidate rejection funnel.
- Per-fold model and policy audit.
- All out-of-sample specialist scores.
- Nested out-of-sample candidates.
- Frequency-profit Pareto frontier.
- Loadable shadow model bundle and digest.
- Human-readable research evidence report.

No artifact may claim profitability or production authorization unless the
historical gates and the subsequent 150-day immutable shadow both pass.
