# WP V25 Positioning Rank Protocol

## Status

This protocol is frozen before any V25 profitability outcome is read. The V25
data feasibility probe may inspect only source availability. The full V25 data
build is outcome blind and cannot authorize this model.

V24 is closed and rejected. Its global classification AUC did not translate
into useful same-day, same-slot stock discrimination. V25 therefore asks one
independent question: do prior-day holder-cost and positioning features rank
stocks inside the live top-five opportunity set well enough to produce
positive net economics?

There is one model family, one policy, and one set of acceptance gates. A
failed result closes V25. Features, weights, thresholds, subgroups, and exit
rules may not be changed after the result is read.

## Objective And Execution Contract

- Signal times: T-day 14:20, 14:25, 14:30, 14:35, 14:40, 14:45, and 14:50.
- Source set: immutable V9 top five per slot from the V24 data contract.
- Entry: established executable next-five-minute-close contract with the
  established slippage assumptions.
- Exit: established fixed T+1 close-auction contract.
- Outcome: net return after established round-trip costs and failed-exit
  treatment.
- Multiple qualified stocks may be published; the human user chooses whether
  and which one to buy.
- `NO_SIGNAL` is valid and thresholds cannot be relaxed to force a list.

For each trade date and stock, the first qualifying signal is immutable.
Later appearances cannot replace its time, price, model, or outcome.

## New Information Family

Only causal V25 positioning features may drive the new alpha models:

- previous A-share trading day's `cyq_perf` holder-cost distribution;
- previous and second-previous trading day's `margin_detail` balances and
  flows;
- previous trading day's `top_list` abnormal-trading disclosure.

The previous V9/V24 fill and severe-loss estimates remain hard execution-risk
controls. They are not V25 alpha inputs.

No T-day close, T+1 value, future bar, outcome, truth, return label, or
post-signal field may be a feature.

## Frozen Model

- Training window: 252 prior A-share trading days.
- Purge: 2 trading days.
- Calibration window: 42 prior A-share trading days.
- Final purge before every outer test fold: 2 trading days.
- Minimum complete training rows: 4,000.
- Minimum complete calibration rows: 800.
- Minimum pairwise training rows: 12,000.
- Minimum pairwise calibration rows: 2,000.
- Outer tests: pre-existing immutable V9 folds.

Three regularized logistic models are fitted:

1. absolute probability that net return is positive;
2. absolute probability that net return is at or below -2.00%;
3. pairwise probability that stock A beats stock B inside the same date and
   five-minute slot.

The pairwise sample contains both A-B and B-A differences. Each date-slot
group has equal total pair weight. Repeated five-minute appearances are
stock-day equalized in the absolute models. All probabilities are calibrated
on the prior calibration window only.

The pairwise score of a stock is its mean calibrated probability of beating
the other stocks in its live date-slot group.

## Frozen Release Policy

The score is:

```text
1.25 * (P(net positive) - 0.50)
+ 1.00 * (within-slot pairwise score - 0.50)
- 1.00 * P(net return <= -2.00%)
```

Hard gates:

- complete point-in-time and prior-positioning row;
- round-trip fill lower bound at least 0.95;
- frozen source severe-loss probability at most 0.45;
- V25 positive probability at least 0.50;
- V25 severe-loss probability at most 0.25;
- pairwise score at least 0.50;
- market-data age no more than 420 seconds when available;
- legal 14:20-14:50 slot.

Only the highest-score eligible stock in each date-slot can proceed. The score
threshold is the prior calibration-window quantile that targets a 25%
candidate-day rate; it is not selected for historical profit. Streaming order
is authoritative, and at most three first signals are released per trade day.

## Historical Acceptance Gates

The nested out-of-sample release must pass every V24 economic gate:

- at least 120 candidates and 80 candidate days;
- candidate-day rate between 12% and 35%;
- win rate at least 55%;
- Wilson win-rate lower bound at least 50%;
- trade-date-clustered win-rate lower bound at least 48%;
- +0.50% hit rate at least 40%;
- -2.00% tail-loss rate at most 15%;
- mean net return at least +0.20%;
- clustered mean-return lower bound above zero;
- profit factor at least 1.20;
- nonnegative mean after another 50bp cost;
- 10th-percentile return no worse than -3.00%;
- at least three active years, at least 20 candidates in every active year,
  at least three positive years, and worst active-year mean at least -0.10%;
- all temporal, identity, point-in-time, and outcome integrity checks.

V25 also must pass every stock-ranking gate over all nested OOS scored rows:

- at least 1,000 evaluable date-slot groups;
- mean same-slot Spearman IC at least +0.05;
- mean highest-minus-lowest ranked return at least +0.20%;
- trade-date-clustered 95% lower bound of that spread above zero;
- positive highest-minus-lowest spread in at least three calendar years.

## Decision Rule

- Any failed historical gate rejects V25.
- A historical pass does not authorize production.
- A passing bundle is frozen and must complete at least 150 untouched future
  A-share shadow trading days, with at least 60 candidates over at least 40
  candidate days, before a production decision.
- Shadow and production must use the same first-signal entry and fixed T+1
  close exit contracts.
