# WP V22 Market-License Protocol

## Status

This protocol is frozen before any V22 historical result is produced.

V22 is research-only. A complete historical pass can authorize only an
immutable 150-trading-day future shadow run. It cannot authorize production.

## Objective

Maximize the probability that an executable candidate first published between
14:20 and 14:50 on T earns a positive net return under the fixed T+1 close exit
after all established round-trip costs.

The system may publish zero candidates. It must not lower thresholds to force a
daily list.

## Why V22 is a distinct mechanism

V21 run `30624949710`, artifact `8791116655`, digest
`sha256:028d4d67947f1e83e252092e1d88bfb9be89dbcd9589f3faf2119ebca8f17144`
failed 13 preregistered historical gates. Its stock-level economic-margin
classifier did not transfer out of sample.

V22 does not retune the V20 or V21 stock scores. It separates two decisions:

1. the immutable V9 selector chooses one executable stock leader at each
   five-minute slot;
2. a separate market-license model decides whether the market and
   opportunity-set state permits releasing that leader.

The license model cannot use the leader's code, identity, stock score, rank,
individual stock features, or future outcome as prediction inputs.

## Immutable execution contract

- Entry: the existing executable signal-price contract at the first qualifying
  14:20-14:50 observation.
- Exit: the existing fixed T+1 close contract.
- Return: net of the established baseline all-in round-trip costs.
- Stress: subtract a further 50bp from every executed candidate.
- Failed exits: retain the existing non-fill penalty contract.
- First signal: once a stock qualifies, its signal time and signal price are
  immutable.
- Daily output: at most two distinct candidates; the human decides whether and
  which one to buy.

## Source and chronology

- Source: immutable V9 outer-fold out-of-sample predictions restored from run
  `30600193544`.
- Evaluation window: the frozen V9 three-year window.
- Per slot: V9 chooses one leader after fixed source fill, loss-risk, model
  disagreement, and freshness filters.
- Gate training: 126 prior out-of-sample trading days.
- Train/calibration purge: two trading days.
- Probability and threshold calibration: 42 prior out-of-sample trading days.
- Calibration/test purge: two trading days.
- Test: the next untouched outer fold.
- Labels from a test fold must never affect its leader identity, features,
  market-license model, calibration, gates, or threshold.

## Frozen feature contract

The license model may use only:

- point-in-time market return, breadth, dispersion, gap, tail path, prior
  market trend and volatility, and up/down-limit counts already present in the
  immutable V9 feature rows;
- point-in-time aggregates of the eligible opportunity set at that slot:
  count, intraday-return distribution, breadth, V9 probability distribution,
  utility distribution, score distribution and top margin, median severe-loss
  probability, and median round-trip fill probability.

The following are forbidden:

- `ts_code`, stock identity, stock rank, and the selected stock's score;
- any selected-stock raw feature;
- `net_return_pct`, labels, T+1 prices, future fields, and truth fields.

At least 12 usable causal license features are required in every fitted fold.

## Frozen model

The policy family size is exactly one.

- Binary target: whether the independently selected V9 slot leader has
  `net_return_pct > 0`.
- Models: one regularized histogram gradient-boosted tree and one regularized
  logistic model.
- Blend: 70% tree, 30% linear.
- Calibration: isotonic calibration on the prior 42-day calibration segment.
- Conservative probability: calibrated probability minus half the model
  disagreement minus 0.02.

## Frozen policy

- Target candidate-day rate used only for unlabeled threshold calibration:
  12%.
- Maximum candidates per day: two.
- License-model spread: at most 0.25.
- Source round-trip fill probability lower bound: at least 0.95.
- Source severe-loss probability: at most 0.40.
- Source probability and expected-return model spread: at most 0.30.
- Active-session market-data age: at most 420 seconds when present.
- Threshold: the calibration-day maximum conservative license probability that
  corresponds to the fixed 12% target candidate-day rate.
- Within a day: retain the earliest qualifying signal for each distinct stock,
  then at most the first two distinct candidates.

No threshold, feature family, policy component, time slot, or calendar subset
may be changed after V22 results are visible.

## Historical acceptance gates

All gates must pass:

- at least 60 nested out-of-sample candidates;
- at least 45 candidate days;
- candidate-day rate between 8% and 22%;
- win rate at least 55%;
- Wilson win-rate lower bound at least 48%;
- day-clustered win-rate lower bound at least 48%;
- at least 45% of candidates earn more than +0.50% net;
- at most 18% of candidates lose 2.00% or more;
- mean net return at least +0.25%;
- day-clustered 95% lower bound for mean net return above zero;
- Profit Factor at least 1.25;
- mean return remains nonnegative after an additional 50bp cost;
- 10th-percentile return at least -3.00%;
- candidates in at least three calendar years;
- at least ten candidates in every active calendar year;
- at least three positive calendar years;
- worst calendar-year mean net return at least -0.10%;
- complete source, feature, and temporal integrity.

## Shadow gate

Passing history only authorizes a frozen research candidate for shadow mode.
Promotion remains forbidden until:

- at least 150 future A-share trading days have elapsed;
- at least 30 immutable candidates on at least 20 candidate days exist;
- the separate shadow promotion contract passes without refitting from shadow
  outcomes.

If any historical gate fails, V22 is rejected. The result may justify acquiring
a genuinely new point-in-time data family, but it may not be repaired by
post-result threshold or subgroup search.
