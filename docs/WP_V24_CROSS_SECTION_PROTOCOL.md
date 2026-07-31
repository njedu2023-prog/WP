# WP V24 Cross-Section Microstructure Protocol

## Frozen Research Question

V23 is closed and rejected. It selected only one outcome-blind source leader
per slot and its stacked conservative gates reduced 725 evaluation days to
three candidates with negative net economics.

V24 asks one new question: can a wider but still outcome-blind opportunity set,
combined with causal cross-sectional and microstructure information, produce a
useful-frequency candidate set with positive out-of-sample net economics?

This protocol is frozen before any V24 outcome result is read. There is one
model and one release policy. A failed result closes V24; its thresholds,
features, weights, and subgroups may not be tuned after seeing the result.

## Objective And Execution Contract

- Signal times: T-day 14:20, 14:25, 14:30, 14:35, 14:40, 14:45, and 14:50.
- Source set: fixed top five stocks per slot from the immutable V9 broad-recall
  frontier and selection score.
- Source identity selection is outcome blind.
- Entry: the established executable next-five-minute-close contract including
  the established slippage assumptions.
- Exit: the established fixed T+1 close-auction contract.
- Outcome: net return after the established round-trip costs and execution
  failure treatment.
- Multiple qualified stocks may be published; the human user decides whether
  and which stock to buy.
- `NO_SIGNAL` is valid. No threshold may be relaxed to force a list.

The first qualifying signal for each stock and trade date is immutable. A later
appearance may not replace its signal time, signal price, model version, or
economic outcome.

## Outcome-Blind Opportunity Set

For every immutable V9 outer fold:

1. project only source columns that exist before the T-day signal;
2. construct the V19 broad-recall frontier without loading labels or returns;
3. apply the frozen V20 source score;
4. retain ranks one through five in each slot;
5. only then attach causal point-in-time features;
6. attach outcomes only inside the research process after identities and all
   source-data digests have been verified.

The data manifest must prove:

- exactly one row per trade date, slot, and stock;
- no more than five source rows per slot;
- source rank between one and five;
- immutable source shard and dataset digests;
- no outcome field read during source selection;
- at least 98% complete point-in-time feature coverage;
- zero unresolved Tushare query failures.

## Point-In-Time Feature Contract

V24 may use only:

- one-minute bars from 13:55 through the signal timestamp;
- the completed same-day opening auction;
- L2 money flow from the immediately preceding A-share trading date;
- frozen V9 source priors;
- V19/V20 cross-sectional ranks, score gaps, slot breadth, dispersion,
  persistence, and score-path features computed at or before the signal.

No T+1 value, outcome, future bar, closing truth, return label, or post-signal
field may be a model feature.

## Frozen Model

- Training window: 252 prior A-share trading days.
- Purge: 2 trading days.
- Calibration window: 42 prior A-share trading days.
- Final purge before each outer test fold: 2 trading days.
- Minimum complete training rows: 4,000.
- Minimum complete calibration rows: 800.
- Outer tests: the pre-existing immutable V9 folds.

Repeated five-minute appearances of one stock do not create repeated economic
sample weight. Training and calibration weights are normalized so each
stock-day has the same total weight before temporal weighting.

The model has four heads:

- probability that net return is above zero;
- probability that net return exceeds +0.50%;
- probability that net return is at or below -2.00%;
- expected clipped net return.

Each probability head is a frozen blend of 70% shallow regularized histogram
tree and 30% regularized logistic model, calibrated only on prior data. The
return head is a frozen 70% shallow regularized tree and 30% ridge blend,
calibrated only on prior data.

## Frozen Release Policy

V24 uses one soft economic score:

```text
0.50 * expected net return
+ 1.25 * (P(net positive) - 0.50)
+ 0.75 * (P(net return > 0.50%) - 0.35)
- 1.00 * P(net return <= -2.00%)
- model disagreement penalties
```

Hard probability lower-bound gates from V23 are removed. The following
execution and integrity gates remain hard:

- round-trip fill lower bound at least 0.95;
- source severe-loss probability at most 0.45;
- each probability-model spread at most 0.40;
- expected-return model spread at most 5.00 percentage points;
- market-data age no more than 420 seconds when an age is available;
- complete point-in-time feature row;
- legal 14:20-14:50 slot.

The economic-score threshold is the prior calibration-window quantile needed
to target a 25% candidate-day rate. Calibration outcomes train and calibrate
the fixed model, but they are not searched for a profitable threshold.

At most three candidates are released per trade date. Streaming order is
authoritative: once a stock first qualifies, a later signal cannot replace it.
The full A-share evaluation calendar, including no-source and no-signal dates,
is the denominator for opportunity frequency.

## Historical Acceptance Gates

Every gate must pass:

- at least 120 nested out-of-sample candidates;
- at least 80 candidate days;
- candidate-day rate between 12% and 35%;
- win rate at least 55%;
- Wilson win-rate lower bound at least 50%;
- trade-date-clustered win-rate lower bound at least 48%;
- +0.50% margin hit rate at least 40%;
- -2.00% tail-loss rate at most 15%;
- mean net return at least +0.20%;
- trade-date-clustered mean-return lower bound above zero;
- profit factor at least 1.20;
- mean return nonnegative after an additional 50bp cost;
- 10th percentile return no worse than -3.00%;
- at least three active calendar years;
- at least 20 candidates in every active year;
- at least three positive calendar years;
- worst active calendar-year mean no worse than -0.10%;
- temporal, source, feature, outcome, and point-in-time data integrity all pass.

## Decision Rule

- If any historical gate fails, V24 is rejected.
- Passing historical gates never authorizes production.
- A passing model and policy are frozen and must complete at least 150 future
  A-share shadow trading days with at least 60 candidates over at least 40
  candidate days before a production decision.
- Production and shadow decisions must use the same immutable first-signal
  entry contract and fixed T+1 close exit.

