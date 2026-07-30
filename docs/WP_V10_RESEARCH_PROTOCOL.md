# WP V10 Market-State Meta-Alpha Research Protocol

## Objective

Maximize the probability that a candidate first shown between 14:20 and
14:50 can be bought under the fixed entry contract and produces a positive
net return under the predefined T+1 exit contract. Transaction costs,
slippage, entry misses, exit failures, and price-limit constraints are part
of the target. `NO_TRADE` is valid.

## Immutable Source

V10 does not retrain or rewrite V9 predictions. It consumes only immutable
walk-forward out-of-sample V9 prediction shards and their already observed
execution outcomes. A V10 fold may train only on V9 OOS rows strictly earlier
than its test start.

## Candidate Frontier

For memory-bounded research, each date and signal slot retains the fixed union
of the top 12 rows by:

- V9 all-in positive probability;
- V9 expected utility;
- V9 selection score;
- V9 conditional positive probability;
- lowest V9 severe-loss probability.

The frontier is fixed before V10 outcomes are inspected. It is not an
outcome-selected universe.

## Causal Meta Features

V10 uses V9 probabilities, expected-return estimates, uncertainty spreads,
current return, signal time, and same-time cross-sectional market context.
No T+1 value, target, label, future bar, later signal slot, or later daily
aggregate is a feature.

## Nested Time Protocol

Each eligible test fold uses:

- 126 prior trading days for meta-model fitting;
- a two-day purge;
- 21 days for probability and return calibration;
- a two-day purge;
- 42 days for fixed policy design;
- a two-day purge;
- 21 days for independent policy confirmation;
- a final two-day purge before the test fold;
- the next untouched V9 fold for testing.

The policy grid is fixed in source. A policy must pass design and confirmation
gates before it can emit candidates in the test fold. Otherwise that fold is
`NO_TRADE`.

## Execution Rule

Qualified events are processed chronologically. The first qualifying signal
for a stock is immutable. A policy may select at most one, two, or three
candidates per day according to the fixed grid. It cannot rank a 14:20 signal
using information from a later slot.

## Evidence and Promotion

The research workflow writes fold decisions, all selected OOS candidates,
yearly metrics, clustered confidence intervals, and 35bp/50bp cost results.
V10 never modifies the production registry or live dashboard.

Even if every historical OOS gate passes, production remains unauthorized
until the exact frozen model completes at least 150 trading days of shadow
operation and passes the separately frozen shadow gates.
