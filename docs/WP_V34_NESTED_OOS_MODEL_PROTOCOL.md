# WP V34 Nested Out-of-Sample Model Protocol

## Frozen Status

This protocol is frozen before the V34 full-session feature dataset is joined
to any T+1 outcome. V34 is one model family with one policy. Post-result
threshold search, subgroup rescue, feature redefinition, and adding another
model after seeing V34 results are prohibited.

## Objective And Trading Contract

The primary target is:

`net_return_pct > 0`

for a candidate that was executable at its immutable 14:20-14:50 signal price,
then sold under the existing fixed T+1 close contract, after established
round-trip cost and non-fill penalties.

The model may publish zero to three unique stocks per day. It never assumes a
user bought a candidate. If one stock qualifies at multiple slots, its first
qualifying slot is immutable.

## Information Available At A Signal

The model uses only:

- the 35 frozen V34 path features available at that signal;
- each path feature's simultaneous percentile rank within the immutable V24
  candidate cross-section for that date and slot;
- causal changes since the same stock's previous candidate appearance that
  day for nine preregistered path features;
- current signal time, candidate appearance count, and minutes since its prior
  candidate appearance.

Later slots cannot alter an earlier row. Stock identity, raw future outcome,
T+1 price, exit state, and date-specific outcome aggregates are not features.

## Model

One fixed ensemble jointly estimates:

1. probability of positive net return;
2. probability of net return above `+0.50%`;
3. probability of severe net loss at or below `-2.00%`;
4. robust expected net return.

Each probability blends a heavily regularized histogram gradient-boosting
tree (`70%`) and standardized regularized logistic model (`30%`), then uses
the immediately preceding calibration window for probability calibration.
Expected return uses the same tree/linear blend with absolute-error loss and
ridge calibration. Model disagreement and a calibration residual quantile
form conservative lower or upper estimates.

## Nested Time Splits

For every immutable outer OOS fold:

- training: preceding 252 A-share trading days;
- calibration: following 42 trading days;
- purge: 2 trading days between history and outer test;
- outer test: only that untouched V9 fold's dates;
- minimum complete training rows: 4,000;
- minimum complete calibration rows: 800.

No random row split is permitted. Repeated appearances of the same stock-day
are equalized in sample weights.

## Fixed Release Policy

Before the fold-specific score threshold, every candidate must pass:

- complete V23 point-in-time and V34 path data;
- source round-trip fill lower probability at least `0.95`;
- source severe-loss probability at most `0.45`;
- conservative positive probability at least `0.50`;
- conservative `+0.50%` margin probability at least `0.20`;
- conservative severe-loss probability at most `0.40`;
- conservative expected net return at least `-0.25%`;
- each classifier disagreement at most `0.40`;
- expected-return model disagreement at most `5.00%`;
- data age missing or no more than 420 seconds;
- legal 14:20-14:50 slot.

The score threshold is the calibration-window daily-maximum quantile that
targets candidate days on 25% of trading days. After applying it, candidates
are ordered chronologically, the first qualifying signal per stock is kept,
and at most three unique stocks are published per day. The model may publish
none.

## Historical Pass Gates

All of these must pass on the concatenated untouched outer OOS predictions:

1. at least 120 candidates and 80 candidate days;
2. candidate-day rate between 12% and 35%;
3. win rate at least 55%;
4. Wilson win-rate lower bound at least 50%;
5. date-clustered win-rate lower bound at least 48%;
6. mean net return at least `+0.20%`;
7. date-clustered mean-return lower bound above zero;
8. profit factor at least 1.20;
9. mean return remains nonnegative after an additional 50 basis points;
10. 10th return percentile at least `-3.00%`;
11. at least three active calendar years, at least 20 candidates per active
    year, every active year positive, and the worst year at least `-0.10%`;
12. complete temporal, source, feature, execution-outcome, and artifact
    integrity.

There is no production authorization from historical data alone. A historical
pass freezes the exact data transform, model, probability calibration, score,
and policy for at least 150 future A-share trading days of untouched shadow
operation, with at least 60 candidates on 40 candidate days before a production
decision.
