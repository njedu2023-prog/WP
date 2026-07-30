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

## Executed Three-Year Result

The frozen protocol was executed by workflow run `30516136872` from immutable
V9 source run `30466227350`. Evidence artifact `8749112795` has digest
`sha256:2df1f8a41973d1a8d96d57fcbbb1d8bfdec659594830bb74f9df4d433a93baa1`.

- Pruned causal candidate frontier: 207,666 rows.
- Outer folds with sufficient history: 12.
- Folds authorized by both design and confirmation: 1.
- Untouched test candidates: 28 across 13 trading days.
- Positive-net-return rate: 57.14%; Wilson lower bound: 39.07%.
- Mean / median net return at 35 bps: -0.0677% / +0.1936%.
- Profit factor: 0.9586.
- Entry fill / T+1 close exit fill: 100.00% / 92.86%.
- 50 bps stress mean net return: -0.2177%.
- Net-return 10% quantile: -8.0967%.
- Maximum day-equal-weight drawdown: -9.0937%.

The minimum sample, clustered win-rate lower bound, positive mean return,
clustered mean-return lower bound, profit factor, exit-fill, and 50 bps stress
gates all failed. V10 is rejected for production and must not replace the
current `NO_SIGNAL` posture. Adding more threshold combinations to the fixed
T+1 close contract is not a defensible next step. A subsequent study must
change a predeclared executable economic contract or add genuinely new
point-in-time information, then restart nested OOS and shadow validation.
