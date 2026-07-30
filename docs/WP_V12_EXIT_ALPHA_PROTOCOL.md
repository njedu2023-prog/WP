# WP V12 Exit-Specific Alpha Protocol

## Research Question

Test whether the failure of V11 came from reusing T+1-close model scores for a
different exit. V12 retrains the second-layer positive-return, severe-loss, and
expected-return models separately for each predeclared executable exit.

## Fixed Contracts

Five contracts run independently:

- T+1 opening call-auction exit;
- net 25, 50, 100, or 200 basis-point take-profit with a T+1 close-auction
  fallback.

All contracts retain V11's conservative fill and failed-exit treatment.

## Causal Evaluation

Each untouched V9 outer test fold uses only earlier V9 out-of-sample
predictions:

- 126 trading days for exit-alpha training;
- a two-day purge;
- 21 days for probability and return calibration;
- a two-day purge;
- 42 days for policy design;
- a two-day purge;
- 21 days for independent confirmation;
- a final two-day purge before the outer test fold.

The candidate frontier is fixed before exit outcomes by retaining the union of
the top 30 candidates per causal V9 score at each slot. Policies process slots
chronologically and keep a stock's first qualifying signal.

## Promotion Boundary

V12 is research-only. A positive historical contract still cannot alter
production until the exact frozen challenger passes all profitability, fill,
50-basis-point stress, tail, and clustered-confidence gates, followed by a new
150-trading-day shadow period.
