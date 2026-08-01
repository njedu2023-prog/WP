# WP V37 Fast-Entry Contract Protocol

## Status

This protocol is frozen before any V37 outcome is calculated. V37 contains
one execution hypothesis, one model architecture, one policy, and one
historical run. Post-result threshold repair, delay selection, subgroup
selection, feature substitution, or a second V37 run is prohibited.

V37 is research only. Passing every historical gate can authorize only a new
150-trading-day untouched shadow period. It cannot authorize production.

## Objective

Maximize the probability that an executable candidate published from 14:20
through 14:50 earns a positive net return under the fixed T+1 close exit after
entry slippage, round-trip costs, and conservative fill rules. A day with no
qualified candidate produces `NO_SIGNAL`.

## Distinct Hypothesis

The existing historical contract observes a completed five-minute signal bar
at time `t` but benchmarks entry at the next five-minute close, `t+5`. That
delay can discard a real short-lived tail edge before the order is assumed to
fill.

V37 tests exactly one new economic contract:

> Publish a completed-bar signal at `t+2`, allow one full minute for a human
> or automated order, and benchmark execution at the exact `t+3` one-minute
> close plus the established 10 bp slippage.

This is not an exit-time search and does not add a future feature. It changes
only the executable entry contract.

## Immutable Sources

- V9 nested out-of-sample source run: `30600193544`;
- V24 outcome-blind point-in-time data run: `30635569735`;
- V34 outcome-blind one-minute data run: `30677075531`;
- exact V9 causal panel cache associated with the immutable source run;
- evaluation end: the immutable source configuration, no later than
  `20260724`.

Source identities, folds, pre-signal features, original scores, model
fingerprints, T+1 dates, adjusted T+1 close truth, and minute partition hashes
are immutable.

## Timing And Entry

Eligible base alerts are `14:20`, `14:25`, `14:30`, `14:35`, `14:40`, and
`14:45`.

For a base alert at time `t`:

1. all model features stop at the completed bar timestamp `t`;
2. publication time is fixed at `t+2`;
3. entry benchmark is the exact one-minute close at `t+3`, plus 10 bp;
4. the last publication is `14:47` and the last entry benchmark is `14:48`;
5. `14:50` base alerts are excluded because this delay contract cannot finish
   inside the legal decision window.

The `t+3` bar is execution truth only. Its price, amount, volume, or path is
forbidden from model features and policy calibration.

## Fill And Return Contract

An entry is fillable only when:

- the exact `t+3` one-minute bar exists;
- the source row was already execution eligible at `t`;
- bar amount is at least RMB 3 million;
- one percent of bar amount covers the RMB 100,000 reference order;
- volume is positive;
- entry price remains at least 0.50% below the up limit;
- entry price remains at least 1.00% above the down limit.

An entry miss leaves cash uninvested and receives zero return. A filled
position that cannot be sold under the fixed T+1 close contract receives the
established `-10%` penalty.

For a completed round trip:

`gross_return = adjusted_T+1_close / (t+3_close * 1.001) - 1`

`net_return = gross_return - 0.25%`

The adjusted T+1 close is read from the immutable V9 causal panel so corporate
actions cannot create false gains or losses.

## Model And Labels

V37 reuses the frozen V34 causal full-session feature set and its fixed
regularized tree-plus-linear architecture. Every model is retrained only on
the newly calculated V37 fast-entry labels:

- positive: V37 all-in net return above zero;
- margin: V37 all-in net return above `+0.50%`;
- severe: V37 all-in net return at or below `-2.00%`;
- regression: V37 all-in net return, clipped to `[-10%, +10%]` only while
  fitting.

The entry bar, T+1 prices, truth fields, targets, and returns are forbidden
features.

## Nested Out-Of-Sample Design

For every outer immutable V9 fold:

- previous 252 A-share trading days for training;
- purge two trading days;
- next 42 trading days for calibration;
- purge two more trading days before the outer test;
- equal total fitting weight for each stock-day;
- fixed probability and return models;
- calibration and policy threshold use only prior calibration dates;
- every outer test date is scored exactly once.

## Fixed Policy

- one policy only;
- target candidate-day rate: `25%`;
- maximum three candidates per day;
- all V34 fixed probability, severe-loss, uncertainty, point-in-time,
  freshness, and path-integrity gates remain unchanged;
- score threshold is the unlabeled prior-calibration daily-max quantile
  corresponding to the fixed 25% rate;
- first qualifying signal for a stock is immutable;
- allow `NO_SIGNAL`.

Threshold calibration cannot read any return, label, fill truth, or T+1 value.

## Historical Acceptance Gates

Every gate must pass:

- at least 150 candidates and 100 candidate days;
- candidate-day rate between 15% and 40%;
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
- entry fill rate at least 98%;
- T+1 exit fill rate conditional on entry at least 98%;
- at least three active calendar years;
- at least 25 candidates in every active year;
- at least three positive calendar years;
- worst calendar-year mean at least `-0.10%`;
- all temporal, source, feature, minute-partition, panel-truth, execution, and
  outcome checks pass.

## Decision Rule

- If any gate fails, V37 is permanently rejected.
- If every gate passes, freeze the exact bundle and run it unchanged for at
  least 150 future A-share trading days.
- Because earlier hypotheses already used this historical window, only the
  untouched future shadow can provide confirmatory profitability evidence.
