# WP V36 Post-Alert Entry Confirmation Protocol

## Status

This protocol is frozen before any V36 outcome is calculated. V36 contains
one feature family, one model family, one policy, and one historical run.
Post-result threshold repair, feature substitution, subgroup selection, or a
second V36 run is prohibited.

V36 is research only. Passing every historical gate can authorize only a new
150-trading-day untouched shadow period. It cannot authorize production.

## Objective

Maximize the probability that every executable candidate published during the
legal tail window earns a positive net return under the fixed T+1 close exit,
after the established entry slippage, round-trip costs, and fill rules. A day
with no qualified candidate produces `NO_SIGNAL`.

## Distinct Hypothesis

V23, V24, V34, and V35 used only information available at the original
five-minute alert timestamp. They did not test whether a stock should remain a
watch item for four more completed one-minute bars before it becomes a public
candidate.

V36 tests one new causal mechanism:

> A short, post-alert price-and-amount confirmation path observed before the
> already-established next-five-minute entry benchmark may separate durable
> tail demand from transient spikes.

The original V9/V24 row is therefore an observation alert, not the V36 public
signal. V36 cannot use the entry bar, T+1 prices, returns, or labels as model
features.

## Immutable Sources

- V9 nested out-of-sample source run: `30600193544`;
- V24 outcome-blind point-in-time data run: `30635569735`;
- V34 outcome-blind one-minute data run: `30677075531`;
- evaluation end: the immutable source configuration, no later than
  `20260724`.

The source identity set, source folds, source scores, base alert timestamps,
base alert prices, entry prices, outcomes, and model fingerprints are
immutable.

## Causal Timing And Execution

Only base alerts at `14:20`, `14:25`, `14:30`, `14:35`, `14:40`, and `14:45`
are eligible.

For a base alert at time `t`:

1. observe exactly four completed one-minute bars `(t, t+4]`;
2. calculate V36 features at `t+4`;
3. publish only if the frozen V36 policy passes at `t+4`;
4. use the existing `t+5` close plus 10 bp entry-slippage benchmark;
5. sell under the unchanged T+1 close contract and deduct the existing
   25 bp round-trip cost.

The `t+5` entry bar is used only for immutable execution truth and parity
auditing. It is forbidden from every V36 feature. The last possible public
signal is `14:49`, with entry at `14:50`. The old `14:50` base alert is
excluded because it cannot be confirmed and entered inside the legal window.

## Feature Family

V36 uses only ten immutable source-prior values and fifteen new post-alert
confirmation values.

Source priors:

- slot minute and return from previous close;
- lower positive-return probability;
- conditional positive-return probability;
- severe-loss probability;
- lower round-trip fill probability;
- probability-model spread;
- expected-return-model spread;
- lower expected utility;
- source selection score.

Post-alert confirmation:

- four-minute return;
- range, maximum extension, and maximum drawdown from the base alert price;
- rebound from the interval low and reversal from the interval high;
- interval close position and interval VWAP gap;
- up-minute share and directional efficiency;
- signed amount imbalance;
- amount ratio versus the preceding twenty completed minutes;
- second-half versus first-half amount acceleration;
- final two-minute return;
- zero-amount share.

Stock identity, entry-bar values, target fields, truth fields, T+1 fields,
gross return, net return, and exit fields are forbidden model features.

## Labels

Labels are read only inside each outer fold's prior train and calibration
dates:

- `positive`: established all-in `net_return_pct > 0`;
- `margin`: established all-in `net_return_pct > +0.50%`;
- `severe`: established all-in `net_return_pct <= -2.00%`;
- regression target: established all-in net return, clipped to
  `[-10%, +10%]` only while fitting.

Final metrics always use original, unclipped returns. Entry misses remain the
existing zero-return cash outcome. Failed T+1 exits retain the established
conservative penalty.

## Nested Out-Of-Sample Design

For every outer V9 fold:

- previous 252 A-share trading days for training;
- purge two trading days;
- next 42 trading days for calibration;
- purge two more trading days before the outer test;
- each stock-day receives equal total fitting weight;
- fixed regularized tree and linear models estimate positive, margin, severe,
  and expected return;
- probability and return calibration use only the prior calibration period.

Every outer test date is scored exactly once. No outer-test outcome can affect
its model, calibrator, or policy threshold.

## Fixed Policy

- target candidate-day rate: `30%`;
- maximum three candidates per day;
- lower round-trip fill probability at least `0.95`;
- source severe-loss probability at most `0.45`;
- lower V36 positive probability at least `0.50`;
- lower V36 margin probability at least `0.20`;
- upper V36 severe-loss probability at most `0.40`;
- lower V36 expected net return at least `-0.25%`;
- each probability-model spread at most `0.35`;
- expected-return-model spread at most `3.00%`;
- data age at most 420 seconds when available;
- score threshold is the prior calibration period's unlabeled daily-max score
  quantile corresponding to the fixed 30% rate;
- the first passing confirmation for a stock is immutable;
- at a shared confirmation time, retain the highest scores while respecting
  the remaining daily capacity;
- allow `NO_SIGNAL`.

The threshold calculation cannot read calibration outcomes or labels.

## Data Integrity Gates

- exactly four feature bars after the base alert and through confirmation;
- latest feature bar equals the confirmation timestamp;
- no entry bar is present in the feature window;
- the entry audit bar is exactly five minutes after the base alert;
- source entry price and one-minute entry close plus established slippage agree
  within 10 bp;
- all candidate identities with legal timing are covered exactly once;
- all model features are finite or explicitly imputed;
- all source partition hashes and row counts match the immutable V34 manifest.

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
- at least three active calendar years;
- at least 25 candidates in every active year;
- at least three positive calendar years;
- worst calendar-year mean at least `-0.10%`;
- all temporal, source, feature, minute-data, signal-time, entry-price, and
  outcome integrity checks pass.

## Decision Rule

- If any historical gate fails, V36 is permanently rejected and must not enter
  shadow or production.
- If all historical gates pass, freeze the resulting bundle and run it without
  modification for at least 150 future A-share trading days.
- The historical window has already hosted earlier hypotheses, so only the
  future untouched shadow period can provide confirmatory evidence of
  profitability.
