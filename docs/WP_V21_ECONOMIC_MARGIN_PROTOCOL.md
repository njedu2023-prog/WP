# WP V21 Economic-Margin Protocol

## Status

This protocol is frozen before any V21 historical result is produced.

V21 is research-only. Even if every historical gate passes, it cannot enter
production until it completes at least 150 future A-share trading days of
immutable shadow operation.

## Objective

Maximize the probability that an executable candidate first published between
14:20 and 14:50 on T earns a positive net return under the fixed T+1 close exit
after all established round-trip costs.

The system may publish zero candidates. It must not lower thresholds to force a
daily list.

## Why V21 is a new mechanism

V20 run `30623421294` produced 302 candidates but only a 46.03% win rate,
0.0291% mean net return, and -0.4709% mean return under an additional 50bp cost
stress.

The immutable V20 loss-attribution run is `30624371230`, artifact
`8790726669`, digest
`sha256:5cb3cf1c23b8cac8b8053dd64f7d01f2a3659a27650c313d34a2b0d1e40ccbc6`.
It showed that V20's composite score and expected-return estimates had almost
no monotonic relationship with realized return. Losses were also distributed:
the worst 20 events explained only 43.74% of total loss points.

V21 therefore does not tune V20's score or copy a favorable diagnostic
subgroup. It changes the training objective:

1. predict whether net return exceeds a +0.50% economic safety margin;
2. separately predict whether net return is at or below -2.00%;
3. rank only by the conservative lower bound of the safety-margin
   probability;
4. use tail probability, model disagreement, fill probability, return
   quantile, and data freshness as hard risk gates.

Diagnostic subgroups are explanatory evidence only and are not V21 policy
rules.

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
- Per slot: retain the top three V9 leaders using information available at that
  slot.
- Gate training: 126 prior out-of-sample trading days.
- Train/calibration purge: two trading days.
- Probability and threshold calibration: 42 prior out-of-sample trading days.
- Calibration/test purge: two trading days.
- Test: the next untouched outer fold.
- Labels from a test fold must never affect its features, models, probability
  calibration, hard gates, or threshold.

## Frozen model

The policy family size is exactly one.

### Safety-margin model

- Binary target: `net_return_pct > +0.50`.
- Models: one regularized histogram gradient-boosted tree and one regularized
  logistic model.
- Blend: 70% tree, 30% linear.
- Calibration: isotonic calibration on the prior 42-day calibration segment.
- Conservative probability:
  calibrated probability minus half the model spread minus 0.02.

### Tail-loss model

- Binary target: `net_return_pct <= -2.00`.
- Models and blend: the same fixed tree/linear architecture.
- Calibration: independent isotonic calibration.
- Conservative upper probability:
  calibrated probability plus half the model spread plus 0.02.

### Return-tail model

- Target: realized net return.
- Model: regularized 20th-quantile histogram gradient boosting.
- Calibration: add the weighted 20th percentile of prior calibration
  residuals.

## Frozen policy

- Target candidate-day rate used only for unlabeled threshold calibration:
  12%.
- Maximum candidates per day: two.
- Safety-margin model spread: at most 0.25.
- Tail model spread: at most 0.25.
- Tail-loss probability upper bound: at most 0.35.
- Predicted calibrated return 20th percentile: at least -3.00%.
- Round-trip fill probability lower bound: at least 0.95.
- Source severe-loss probability: at most 0.40.
- Active-session market-data age: at most 420 seconds when present.
- Threshold: the calibration-day maximum conservative margin probability that
  corresponds to the fixed 12% target candidate-day rate.
- Within a day: sort by earliest qualifying slot first, then conservative
  margin probability; retain at most two distinct stocks.

No threshold or policy component may be changed after V21 results are visible.

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
- complete source and temporal integrity.

## Shadow gate

Passing history only authorizes a frozen research candidate for shadow mode.
Promotion remains forbidden until:

- at least 150 future A-share trading days have elapsed;
- at least 30 immutable candidates on at least 20 candidate days exist;
- the separate shadow promotion contract passes without refitting from shadow
  outcomes.

If any historical gate fails, V21 is rejected. The result may guide a genuinely
new preregistered mechanism, but it may not be repaired by post-result
threshold search.
