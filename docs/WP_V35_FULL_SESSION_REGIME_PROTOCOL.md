# WP V35 Full-Session Regime License Protocol

## Status

This protocol is frozen before any V35 outcome is calculated. V35 is one
model family with one policy and one historical run. Post-result threshold
repair, feature substitution, subgroup selection, or another V35 run is
prohibited.

V35 is research only. Passing every historical gate can authorize only a new
150-trading-day untouched shadow period. It cannot authorize production.

## Objective

Maximize the probability that every executable candidate published between
14:20 and 14:50 earns a positive net return under the fixed T+1 close exit,
after all established costs and fill rules. A day with no licensed slot
produces `NO_SIGNAL`.

## Why This Is Distinct

V20 and V22 already rejected stock-level opportunity gates and ordinary market
state licensing. V34 added a genuinely new information family: causal
full-session minute paths available at each signal timestamp. V34 showed that
these paths did not rank individual stocks.

V35 tests one narrower mechanism:

> The cross-stock agreement of full-session paths inside the immutable V24
> top-five opportunity set may identify a small number of tail slots where a
> fixed source-ranked basket is favorable.

V35 does not reuse the failed V34 stock score, does not search individual
stocks with V34 outcomes, and does not add any post-result rule.

## Immutable Sources

- V9 nested out-of-sample source run: `30600193544`;
- V24 outcome-blind point-in-time data run: `30635569735`;
- V34 outcome-blind full-session path data run: `30677075531`;
- V34 probe run: `30676761165`;
- evaluation end: the immutable source configuration, no later than
  `20260724`.

The V24 identity set, source ranks, signal timestamps, signal prices, folds,
and model fingerprints are immutable.

## Candidate Basket

For each trade date and each legal slot from 14:20 through 14:50:

1. start from the immutable V24 top-five candidates;
2. require execution eligibility, complete V23 point-in-time data, complete
   V34 path data, lower round-trip fill probability at least `0.95`, source
   severe-loss probability at most `0.45`, source probability and expected
   return model spreads at most `0.30`, and data age at most 420 seconds when
   available;
3. sort by immutable `v20_stock_rank_in_slot`, then source selection score,
   source lower positive probability, and stock code;
4. retain the first three candidates;
5. require at least two eligible basket members.

These steps use no T+1 outcome. The basket definition cannot change after the
research result is observed.

## Regime Features

The license sees only slot time, basket member count, and preregistered
cross-stock aggregates of V34 path variables:

- medians of 23 economic path variables covering session/morning/afternoon
  return, volatility, drawdown/rebound/reversal, VWAP position, amount
  acceleration and imbalance, flow agreement, price-amount correlation,
  liquidity, and recent-versus-prior volatility;
- interquartile ranges of eight return, VWAP, flow, and volatility variables;
- seven breadth fractions for positive session/afternoon/post-14:00 return,
  above-VWAP behavior, positive VWAP gap, positive amount imbalance, and flow
  agreement.

Stock identity, stock outcome, future price, T+1 return, and all labels are
forbidden model features.

## Labels

Labels are created only inside each outer fold's prior train and calibration
dates, from the fixed basket:

- `good`: basket mean net return is positive and a strict majority of members
  have positive net return;
- `margin`: basket mean net return exceeds `+0.50%` and a strict majority of
  members have positive net return;
- `severe`: any basket member has net return at or below `-2.00%`;
- regression target: basket mean net return, clipped to `[-10%, +10%]` only
  for model fitting.

All final candidate metrics use the original, unclipped individual net
returns.

## Nested Out-of-Sample Design

For every outer V9 fold:

- use the previous 252 A-share trading days for training;
- purge two trading days;
- use the next 42 trading days for calibration;
- purge two more trading days before the outer test;
- give each trading day equal total fitting weight, with a fixed recency
  weight inside the training period;
- fit fixed regularized tree and linear models for `good`, `margin`, `severe`,
  and basket mean return;
- calibrate probabilities and expected return only on the prior calibration
  period.

An outer test date can be scored exactly once. No outcome from the outer test
fold can affect its model, calibrator, or threshold.

## Fixed Policy

- target candidate-day rate: `20%`;
- daily maximum: three candidates;
- only one slot can be licensed per day;
- the score threshold is the prior calibration period's unlabeled daily-max
  score quantile corresponding to the fixed 20% rate;
- model probability spreads must be at most `0.40`;
- model expected-return spread must be at most `5.00%`;
- when more than one slot crosses the frozen threshold, use the first
  chronological slot;
- publish the fixed source-ranked basket from that slot, up to three members;
- do not replace a candidate with a later signal;
- preserve immutable signal timestamp and signal price;
- allow `NO_SIGNAL`.

The threshold calculation cannot read calibration returns or labels.

## Historical Acceptance Gates

Every gate must pass:

- at least 180 candidates and 100 candidate days;
- candidate-day rate between 12% and 28%;
- win rate at least 55%;
- Wilson lower win bound at least 50%;
- day-clustered lower win bound at least 48%;
- mean net return at least `+0.20%`;
- day-clustered lower mean-return bound above zero;
- profit factor at least `1.20`;
- mean net return after an additional 50 bp stress nonnegative;
- 10th return percentile at least `-3.00%`;
- margin-hit rate at least 35%;
- tail-loss rate at most 20%;
- at least three active calendar years;
- at least 25 candidates in every active year;
- at least three positive calendar years;
- worst calendar-year mean at least `-0.10%`;
- all temporal, source, feature, data, signal-time, signal-price, and outcome
  integrity checks pass.

## Decision Rule

- If any historical gate fails, V35 is permanently rejected and must not enter
  shadow or production.
- If all historical gates pass, freeze the resulting bundle and run it without
  modification for at least 150 future A-share trading days. Production still
  remains unauthorized until that untouched shadow contract passes.
