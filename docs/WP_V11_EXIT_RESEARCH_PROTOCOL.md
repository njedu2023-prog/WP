# WP V11 Executable Exit Research Protocol

## Research Question

Determine whether the rejected fixed T+1 close contract can be improved by a
predeclared and executable T+1 exit without changing the immutable V9
walk-forward predictions. The study must distinguish overnight losses from
T+1 intraday deterioration before any new production model is considered.

## Contract Family

The fixed family is declared before outcomes are inspected:

- T+1 opening call-auction exit;
- T+1 closing call-auction exit;
- net take-profit targets of 10, 25, 50, 100, or 200 basis points with a
  T+1 close-auction fallback.

The take-profit order is credited only at its target price and only when the
T+1 daily high trades at least one price tick through the target. A target
that merely equals the daily high is treated as unfilled. An unfilled target
falls back to the existing close-auction contract. A failed final exit keeps
the existing conservative penalty.

## Causal Selection

V11 consumes immutable V9 out-of-sample predictions and the same fixed causal
candidate frontier used by V10. Candidate qualification uses only V9 values
known at the signal slot. Slots are processed chronologically, a stock's first
qualifying signal is immutable, and later slots cannot displace an earlier
qualified signal.

The pruned causal frontier, attached exit truth, and its SHA-256 digest are
retained as an immutable Parquet artifact so later exit research does not
depend on short-lived fold artifacts.

## Nested Evaluation

For every untouched V9 outer fold:

- use 84 prior trading days for policy design;
- purge two trading days;
- use 42 trading days for independent confirmation;
- purge two trading days before the outer-fold test;
- authorize a policy only if both design and confirmation gates pass.

The study reports every fixed exit contract separately and an adaptive
challenger that may choose among the predeclared contracts using prior
design/confirmation evidence. No outer-fold outcome participates in its own
contract or threshold selection.

## Promotion Boundary

V11 is research-only. Positive historical results cannot alter production.
An exact frozen challenger must still pass all profitability, fill, cost,
tail, and clustered-confidence gates and then complete a new 150-trading-day
shadow period.
