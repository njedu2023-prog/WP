# WP V15 Strict Forward Risk Validation Protocol

## Research question

V14 found a positive historical direction on only 12 candidates in outer fold
15. V15 asks one narrower question:

> Does the exact V10 entry policy plus the exact V14 safe-half exit-risk gate
> remain profitable on every later causal outer fold?

V15 does not search for another policy, threshold, time window, feature set, or
exit contract.

## Frozen discovery contract

The discovery period ends with fold 15 on 2025-05-27. The following rules are
immutable before V15 reads any later outcome:

- `meta_p_positive >= 0.54`
- `meta_expected_net_return_pct >= 0.00`
- `meta_p_severe_loss <= 0.35`
- `p_round_trip_fill_lower >= 0.95`
- `meta_rank_pct >= 0.95`
- early slots only, defined by the original V10 policy
- at most three unique stocks per trade date
- exit-failure risk percentile `<= 0.50` within the same trade date and slot
- risk filtering occurs before the daily top-three choice
- entry and exit remain the immutable V10/V11 execution contracts

The workflow verifies that the V10 fold-15 policy ID is
`p0.54-e0.00-s0.35-f0.95-r0.95-k3-early` and that V14's selected research
direction is `risk_rank_safest_50pct`.

## Forward holdout

Only folds 16 through 22 are evaluated:

- fold 16 begins 2025-05-28
- fold 22 ends 2026-07-24

No observation dated on or before 2025-05-27 may enter a V15 test result. Each
fold's exit-risk model is retrained using only prior outer-OOS dates:

- 126 risk-training dates
- 2 purge dates
- 42 probability-calibration dates
- 2 final purge dates
- the untouched current outer test fold

The model and all thresholds are fixed. Test outcomes never select or tune a
rule.

## Comparison

V15 reports both portfolios:

1. Baseline: the frozen V10 policy alone.
2. Challenger: the V14 safe-half risk gate first, followed by the same frozen
   V10 policy.

The pre-filter order matters. A safe candidate may replace an unsafe candidate
inside the daily top three. Applying the gate after top-three selection would
incorrectly suppress that executable alternative.

All reported candidates retain their exact signal identity and immutable
T+1-close-auction outcome. Candidate-level CSVs and the full scored frontier
are uploaded as immutable evidence.

## Confirmation gates

V15 is considered positive forward evidence only when all of the following
predeclared gates pass:

- all seven forward folds are scored
- at least 250 candidates
- at least 60 trading days
- positive mean net return after baseline costs
- positive lower bound of the day-clustered mean
- day-clustered win-rate lower bound at least 52%
- profit factor at least 1.20
- T+1 exit fill rate at least 98%
- positive total return under the 50bp cost stress

Passing these gates still does not authorize production. A new minimum
150-trading-day shadow run is mandatory. Failure is recorded as evidence and
must not be repaired by changing a threshold after reading the holdout.
