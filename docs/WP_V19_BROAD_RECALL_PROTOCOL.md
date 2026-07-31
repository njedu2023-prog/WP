# WP V19 Broad-Recall Research Protocol

## Decision objective

V19 has one objective: maximize the probability that every executable stock
first published between 14:20 and 14:50 produces a positive net return under
the fixed T+1 close-auction exit contract after all established transaction
costs and slippage. Multiple qualified stocks may be published and the human
user chooses which one to buy. `NO_SIGNAL` is valid and thresholds cannot be
lowered to force a list.

V19 is research-only. The three-year historical interval has already been
examined by earlier versions, so even a passing result is only a screening
result. It cannot authorize production and cannot replace the required 150
future A-share trading days.

## Why V19 is a new test

V10 through V18 trained and evaluated successor selectors on a narrow frontier
that retained roughly the top 12 rows per V9 score at each signal slot. That
was computationally controlled, but it prevented later models from discovering
stocks that V9 ranked too low. V19 changes the upstream retrieval contract
before fitting the selector:

- retain the top 48 rows at each date and slot by V9 positive probability;
- retain the top 48 by conditional positive probability;
- retain the top 48 by cross-sectional top-outcome probability;
- retain the top 48 by expected utility;
- retain the top 48 by selection score;
- retain the top 48 by current return;
- retain the 48 lowest severe-loss probabilities;
- retain 24 deterministic exploration rows independent of outcomes.

The union is deduplicated before any V19 outcome is inspected. Retrieval
identities must remain unchanged if T+1 labels or returns are modified.

## Causal features and model

V19 uses only values available at the current signal time:

- frozen V9 probabilities, expected return, downside and uncertainty;
- full-universe same-time market context computed before retrieval;
- within-stock observations from the current and earlier tail slots;
- retrieval-source indicators, which contain no future outcomes.

The selector is a regularized nonlinear and linear ensemble with separate
probability, return-location and lower-return models. Probability and return
adjustments are fitted on a later calibration window that is still strictly
earlier than the outer test. Training is bounded by a deterministic,
outcome-independent stock-day sample; every retained stock-day keeps all of
its retrieved chronological slots.

Changing any 14:50 value must not change a 14:20-14:45 feature. T+1 values,
labels, later slots and future daily aggregates are forbidden features.

## Fixed policy family

Exactly 16 policy specifications are evaluated:

- target candidate-day rates: 12%, 18%, 24% or 30%;
- daily capacity: one or two candidates;
- minimum prior/current quality observations: one or two.

Every specification has the same fixed economic floors:

- calibrated lower positive probability at least 50%;
- expected net return at least 0%;
- predicted lower-quartile return at least -1.50%;
- V9 severe-loss probability at most 35%;
- lower round-trip fill probability at least 95%;
- selector probability disagreement at most 20%;
- same-slot selector percentile at least 90%;
- market data age at most seven minutes when present.

The score threshold is calibrated on a separate prior 42-day window to target
the declared candidate-day rate. This controls opportunity frequency without
weakening the profitability or execution floors.

## Chronological contract

Signals are processed in the real order 14:20, 14:25, 14:30, 14:35, 14:40,
14:45 and 14:50. A stock is locked at its first qualifying time. A later slot
cannot replace its signal price or time. Daily capacity is consumed in this
same chronological order, so a later high score cannot replace an earlier
already-qualified candidate.

## Nested time protocol

Each selector outer fold uses:

1. 252 to 504 prior trading days for model fitting;
2. a two-day purge;
3. 42 days for model calibration;
4. a final two-day purge before the untouched outer test fold.

Policy authorization for an outer fold uses only previously scored OOS rows:

1. 42 days for candidate-frequency threshold calibration;
2. a two-day purge;
3. 84 days for policy design;
4. a two-day purge;
5. 84 days for one-time confirmation of the frozen design champion;
6. a two-day purge before the outer test.

The design search applies Benjamini-Hochberg control at `q <= 0.10`.
Confirmation is run only once for the frozen design champion. A failure
produces `NO_SIGNAL`; the runner cannot try the second-place policy.

## Historical gates

The combined nested OOS result must pass every gate:

- at least 80 candidates and 40 candidate days;
- candidate-day rate from 10% through 32%;
- win rate at least 55%;
- Wilson and day-clustered win-rate lower bounds at least 50%;
- mean net return at least 0.20%;
- positive day-clustered mean-return lower bound;
- profit factor at least 1.20;
- nonnegative mean after an additional 50 basis points per executed trade;
- 10th-percentile net return at least -3.00%;
- at least two profitable active calendar years;
- worst active calendar-year mean no worse than -0.20%;
- complete source and temporal integrity.

Design and confirmation use smaller minimum sample gates only to decide whether
an outer test may emit signals. They do not replace the combined historical
gates.

## Future shadow gate

Passing historical screening only freezes a research bundle. Production remains
prohibited until that exact model, feature contract and policy complete at
least:

- 150 future A-share trading days;
- 60 immutable verified candidates;
- 30 candidate days;
- the same profitability, tail-risk, cost and integrity gates.

Historical reconstruction and backfill never count toward future shadow days.
