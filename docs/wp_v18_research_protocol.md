# WP V18 Causal Frequency Research Protocol

## Objective

V18 is a research-only successor candidate. It maximizes the probability that
each executable signal first observed from 14:20 through 14:50 earns a positive
net return under the immutable T+1 close-auction exit contract after established
round-trip costs. Zero candidates remain valid.

V18 does not modify V15, production outputs, the public dashboard, or the live
candidate ledger. It cannot authorize production.

## Frozen evidence

- Candidate source: immutable V9 causal out-of-sample scores.
- Exit truth: immutable V11 T+1 close-auction executable labels.
- Base model fingerprint: `16695b7fab38a5428f70`.
- V15 is retained unchanged as a positive but unconfirmed benchmark.
- V16 and V17 remain failed research layers and are not promoted.

## Causal signal construction

For each stock and day, V18 processes 14:20, 14:25, 14:30, 14:35, 14:40,
14:45, and 14:50 in chronological order. Added persistence features use only
the current slot and earlier slots:

- probability, utility, score, and intraday-return changes from the prior slot;
- changes from 14:20;
- expanding means and worst values;
- score stability;
- number of prior high-quality observations.

Future slots cannot affect an earlier row. A stock is locked at its first
qualified slot. A later candidate cannot replace a candidate already admitted
under the daily capacity.

## Predeclared policy family

The policy family contains exactly 16 specifications:

- target candidate-day rates: 8%, 12%, 16%, or 20%;
- daily capacity: one or two candidates;
- persistence requirement: one or two high-quality observations.

Each specification uses the same broad execution and risk floor. A score
threshold is estimated only from a separate 42-day threshold-calibration
segment. The threshold is then frozen before design, confirmation, and outer
test evaluation.

## Time splits

Every outer fold observes this order:

1. selector model training, 252 to 504 prior trading days;
2. two-day purge;
3. selector probability and return calibration, 42 days;
4. outer test fold.

Policy selection uses previously scored out-of-sample rows only:

1. threshold calibration, 42 days;
2. two-day purge;
3. policy design, 42 days;
4. two-day purge;
5. one-time confirmation, 42 days;
6. two-day purge;
7. outer test.

The 16-policy family uses Benjamini-Hochberg control at `q <= 0.10` in the
design segment. Confirmation is run once for the frozen design champion.

## Historical readiness

Historical shadow eligibility requires all gates:

- at least 100 nested out-of-sample candidates across at least 40 days;
- candidate-day rate from 8% through 30%;
- win rate at least 55%;
- Wilson and day-clustered win-rate lower bounds at least 50%;
- mean net return at least 0.20%;
- positive day-clustered 95% mean-return lower bound;
- profit factor at least 1.20;
- nonnegative mean after an additional 50 basis points per trade;
- 10th-percentile return at least -3.00%;
- at least two profitable calendar years;
- worst active calendar year no worse than -0.20%;
- complete temporal integrity.

Passing these gates only creates a research shadow bundle.

## Future shadow requirement

Promotion remains prohibited until the exact frozen bundle completes at least:

- 150 future A-share trading days;
- 60 immutable candidates;
- 30 candidate days;
- the same return, risk, cost, and temporal-integrity gates.

The future shadow period cannot be backfilled with historical data.
