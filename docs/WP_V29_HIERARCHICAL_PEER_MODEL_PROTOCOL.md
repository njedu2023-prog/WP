# WP V29 Hierarchical Peer Model Protocol

## Frozen Question

The V29 outcome-blind data build passed before this protocol was committed:
31,955 of 31,955 immutable V24 candidate rows have finite hierarchical peer
features, exact identities, and no duplicates.

V29 asks one question: does causal leave-one-out industry confirmation,
represented with fixed L3-to-L2 shrinkage, identify a practical-frequency set
of 14:20-14:50 candidates with positive net economics under the established
T+1 close exit?

There is one model, one policy, and one acceptance rule. Any failed gate closes
V29. Features, weights, thresholds, subgroups, and exit rules may not be tuned
after results are read.

## Execution And Candidate Contract

- immutable V9/V24 top-five source candidates per date and slot;
- signal slots: 14:20, 14:25, 14:30, 14:35, 14:40, 14:45, 14:50;
- established executable next-five-minute-close entry and slippage contract;
- established fixed T+1 close-auction exit;
- established round-trip costs and failed-exit treatment;
- one slot leader can qualify at each signal time;
- at most three immutable first signals per trade date;
- multiple candidates may be published and the human chooses;
- `NO_SIGNAL` is valid and no threshold may be relaxed to force a list.

## Information Boundary

Only the preregistered V29 hierarchical peer features may drive alpha.
Existing V9 estimates are limited to hard execution controls:

- round-trip fill lower bound at least 0.95;
- source severe-loss probability at most 0.45;
- market-data age no more than 420 seconds when available;
- complete point-in-time and V29 feature records.

Old model scores, post-signal prices, T-day close, T+1 values, outcomes, truth,
and artificial user choices are not V29 alpha features.

## Frozen Nested Model

- prior training window: 252 A-share trading days;
- purge: 2 trading days;
- prior calibration window: 42 trading days;
- final purge before each outer test fold: 2 trading days;
- minimum training rows: 4,000;
- minimum calibration rows: 800;
- minimum pairwise training rows: 12,000;
- minimum pairwise calibration rows: 2,000;
- outer tests: immutable V9 folds;
- each repeated stock-day has equal total weight;
- each date-slot pairwise group has equal total weight.

Four regularized logistic heads use exactly the active V29 feature columns:

1. probability net return is positive;
2. probability net return exceeds +0.50%;
3. probability net return is at or below -2.00%;
4. probability stock A beats stock B in the same date and signal slot.

All use median imputation, standardization, `C=0.05`, balanced class weights,
and prior-window isotonic calibration. The pairwise sample includes A-B and
B-A differences.

## Frozen Score And Release Policy

```text
1.00 * (P(net positive) - 0.50)
+ 0.75 * (P(net > 0.50%) - 0.35)
+ 1.00 * (same-slot pairwise score - 0.50)
- 1.00 * P(net <= -2.00%)
```

No model probability is a hard gate. After the execution gates, the highest
score in each date-slot proceeds. The score threshold is the prior 42-day
calibration threshold targeting candidate days on 25% of calendar dates. It is
not selected for historical profit. Chronological first signal is immutable;
the first three qualifying stock-days are retained.

## Historical Acceptance Gates

All V24 economic gates must pass:

- at least 120 candidates over at least 80 candidate days;
- candidate-day rate between 12% and 35%;
- win rate at least 55%;
- Wilson lower bound at least 50%;
- trade-date-clustered win-rate lower bound at least 48%;
- +0.50% hit rate at least 40%;
- -2.00% tail-loss rate at most 15%;
- mean net return at least +0.20%;
- clustered mean-return lower bound above zero;
- profit factor at least 1.20;
- nonnegative mean after an additional 50bp cost;
- 10th percentile no worse than -3.00%;
- at least three active and three positive calendar years;
- at least 20 candidates in every active year;
- worst active-year mean at least -0.10%;
- all temporal, identity, source, feature, and outcome checks pass.

The same-slot mechanism must also pass:

- at least 1,000 evaluable date-slot groups;
- mean same-slot Spearman IC at least +0.05;
- mean highest-minus-lowest return spread at least +0.20%;
- trade-date-clustered 95% lower bound of the spread above zero;
- positive spread in at least three calendar years.

## Decision Rule

Any failed gate rejects V29. A full historical pass only freezes a shadow
candidate. It must then complete at least 150 untouched future A-share trading
days, with at least 60 candidates over at least 40 candidate days, before any
production decision.
