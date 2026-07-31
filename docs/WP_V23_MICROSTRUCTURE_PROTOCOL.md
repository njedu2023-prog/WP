# WP V23 Microstructure Gate Protocol

## Frozen Question

Can genuinely new, point-in-time information improve the immutable V9
market-slot leader enough to produce a useful-frequency candidate set with
positive net economics under the existing execution contract?

This protocol is frozen before reading any V23 model result.

## Objective And Execution Contract

- Signal window: T-day 14:20-14:50 at five-minute intervals.
- Source opportunity: one immutable, outcome-blind V9 leader per slot.
- Entry: next five-minute close plus established slippage.
- Exit: fixed T+1 close-auction contract.
- Outcome: net return after the established round-trip cost.
- Failed execution: existing failure penalty.
- Multiple qualified stocks may be published; the user chooses whether and
  which one to trade.
- `NO_SIGNAL` is valid and thresholds may never be relaxed to force a list.

## New Information

V23 uses only:

1. one-minute bars from 13:55 through the signal timestamp;
2. the completed same-day opening auction;
3. L2 money flow from the immediately preceding A-share trading date;
4. a small frozen set of V9 prior outputs needed to contextualize the leader.

The data manifest must prove at least 98% complete point-in-time coverage.
Rows that fail the fixed 90% minute coverage requirement are marked unavailable
before outcomes are joined.

The outcome-blind leader identity is authoritative. Research may attach a
historical outcome to that identity, but it may not replace an unlabeled leader
with another stock. Candidate-day rates use every A-share open date in the
window, including dates with no source leader or no released candidate.

## Frozen Model

One model family is permitted. There is no model grid.

- Train window: 252 prior A-share trading days.
- Purge: 2 trading days.
- Calibration window: 42 prior A-share trading days.
- Final purge before outer test: 2 trading days.
- Outer test: the existing immutable V9 fold.
- Features: the registered V23 microstructure, auction, previous-day
  money-flow, and frozen V9 prior columns only.

The model has four economic heads:

- probability that net return is above zero;
- probability that net return exceeds +0.50%;
- probability that net return is at or below -2.00%;
- expected clipped net return.

Each probability head is a fixed 70% shallow regularized histogram tree and
30% regularized logistic model, calibrated only on prior data. The return head
is a fixed 70% shallow regularized tree and 30% ridge model, calibrated only on
prior data. Model disagreement and a calibration downside residual are
subtracted from the release score.

## Frozen Policy

Only one policy is evaluated:

- target candidate-day rate: 20%;
- maximum candidates per day: 3;
- positive-probability lower bound: at least 0.50;
- +0.50% margin-probability lower bound: at least 0.25;
- severe-loss probability upper bound: at most 0.35;
- expected-net-return lower bound: at least -0.10%;
- existing source fill and severe-risk gates remain active;
- fixed model-disagreement gates remain active.

The economic-score threshold is the prior calibration-window quantile needed
to target the fixed candidate-day rate. Calibration outcomes are used to
calibrate model outputs but are not searched for a profitable threshold.

The first qualifying signal for each stock and day is immutable. Later
appearances cannot replace its time or price.

## Historical Acceptance Gates

Every gate must pass:

- at least 100 nested OOS candidates;
- at least 70 candidate days;
- candidate-day rate between 12% and 30%;
- win rate at least 55%;
- Wilson win-rate lower bound at least 50%;
- day-clustered win-rate lower bound at least 48%;
- +0.50% margin hit rate at least 40%;
- -2.00% tail-loss rate at most 15%;
- mean net return at least +0.20%;
- day-clustered mean-return lower bound above zero;
- profit factor at least 1.20;
- mean return nonnegative after an additional 50bp cost;
- 10th percentile return no worse than -3.00%;
- at least three active calendar years;
- at least 15 candidates in every active year;
- at least three positive calendar years;
- worst calendar-year mean no worse than -0.10%;
- temporal, source, feature, and point-in-time data integrity all pass.

## Decision Rule

- If any historical gate fails, V23 is rejected.
- If all historical gates pass, production remains forbidden.
- A passing model is frozen and must complete at least 150 future A-share
  trading days of untouched shadow operation before any production decision.
- No V23 threshold, feature, model weight, or subgroup may be changed after the
  result is read and still be called V23.
