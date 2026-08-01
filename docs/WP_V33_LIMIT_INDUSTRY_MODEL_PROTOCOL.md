# WP V33 Limit-Industry Ecology Model Protocol

## Status

This protocol becomes frozen only after the V33 three-year outcome-blind data
manifest passes. It must be committed before the V33 research workflow can read
any candidate return. A historical pass cannot authorize production. It can
authorize only an unchanged 150-future-trading-day shadow run.

## Objective

Select executable T-day candidates first observed from 14:20 through 14:50
whose fixed T+1 close exit has the highest probability of positive net return
after the existing round-trip cost contract. Multiple candidates may qualify;
the human user decides which, if any, to buy. `NO_SIGNAL` is valid.

## Immutable Sources

- V9 nested-OOS outcome shards: run `30600193544`.
- V24 point-in-time candidate and execution-quality data:
  run `30635569735`.
- V33 three-year outcome-blind limit-industry ecology data:
  run `30671468932`.
- V33 data must contain exactly the immutable V24 candidate identities and
  pass all source SHA, coverage, probe-parity, and no-future-information gates.

Raw L2/L3 industry codes are audit identity only. They may not be one-hot
encoded, target encoded, or otherwise used as model inputs.

## Fixed Label And Execution Contract

- Entry: immutable first executable signal price at the 14:20-14:50 slot.
- Exit: fixed T+1 close contract already defined by the V9 source.
- Target: `net_return_pct > 0` after baseline round-trip costs.
- Severe loss target: `net_return_pct <= -2.00%`.
- Fill gate: prior point-in-time lower-bound round-trip fill probability is at
  least `0.95`.
- Existing severe-loss gate: prior point-in-time severe-loss probability is no
  greater than `0.45`.
- Failed-exit penalty and 50bp stress use the unchanged V9 configuration.

## Frozen Model

The only admitted alpha inputs are the numeric V33 ecology features:

- same-L2 and same-L3 member, touch, open-board, down-limit, recent-event,
  net-sealed, rate, and market-share values available by the signal slot;
- immediately previous trading-day market, L2, and L3 event context;
- market-wide causal event context.

The model family is fixed:

1. positive-return head: 50/50 blend of a strongly regularized shallow
   histogram gradient tree and a regularized logistic model;
2. severe-loss head: strongly regularized shallow histogram gradient tree;
3. expected-return head: absolute-error histogram gradient regressor, with
   returns clipped to `[-10%, +15%]` only for fitting and ridge calibration;
4. same-day, same-slot pairwise rank head: regularized logistic comparison
   model.

Hyperparameters in `src/wp/v3/v33_ecology_ranker.py` are frozen. No model grid,
feature subset search, interaction search, or post-result adjustment is
allowed.

## Temporal Design

For every immutable outer test fold:

- training: most recent 252 eligible trading days strictly before test;
- first purge: 2 trading days;
- calibration: following 42 trading days;
- final purge: 2 trading days before the outer test;
- model fitting, active-feature detection, probability calibration, return
  calibration, and score threshold calibration use only those prior segments;
- outer test dates are scored once and never reused for tuning.

Minimum fit evidence:

- 4,000 training candidate rows;
- 800 calibration candidate rows;
- 12,000 training pair rows;
- 2,000 calibration pair rows;
- at least 12 nonconstant admitted features.

## Fixed Candidate Policy

The composite score is:

`expected_return + 1.00*(p_positive-0.50)`
`+ 0.75*(pairwise_score-0.50) - 1.25*p_severe_loss`.

Before the calibration-derived score threshold, every row must pass:

- current same-L2 or same-L3 ecology is active by the signal;
- `p_positive >= 0.50`;
- `expected_net_return_pct >= 0.00`;
- `p_severe_loss <= 0.35`;
- `pairwise_score >= 0.50`;
- the fixed fill and source severe-loss gates;
- a present, nonnegative data age no greater than 420 seconds;
- the fixed point-in-time and legal-slot gates.

The score threshold is the single threshold implied by a fixed 25% target
candidate-day rate in the prior 42-day calibration segment. It is not chosen by
calibration returns. Absolute gates remain binding, so the actual rate may be
lower or zero.

Every candidate that passes the frozen absolute gates and score threshold is
published and written to the immutable candidate ledger. There is no arbitrary
daily quota. Multiple candidates in one slot are allowed. A stock's first
qualifying slot is immutable; later scores cannot replace its signal time or
entry price.

## Historical Pass Gates

All existing V24 economic gates remain binding:

- at least 120 candidates and 80 candidate days;
- candidate-day rate from 12% through 35%;
- win rate at least 55%;
- Wilson win-rate lower bound at least 50%;
- clustered win-rate lower bound at least 48%;
- margin hit rate at least 40%;
- tail-loss rate no greater than 15%;
- mean net return at least `+0.20%`;
- clustered mean-return lower bound above zero;
- Profit Factor at least 1.20;
- mean return under 50bp stress nonnegative;
- 10th return percentile at least `-3.00%`;
- at least three active calendar years, at least 20 candidates per active
  year, at least three positive years, and worst-year mean at least `-0.10%`.

The same-slot rank mechanism must also pass:

- at least 1,000 evaluable same-slot groups;
- mean within-slot rank IC at least 0.05;
- mean top-minus-bottom spread at least `+0.20%`;
- clustered spread lower bound above zero;
- at least three positive rank-spread calendar years.

Every temporal, source, digest, identity, and selected-outcome audit must pass.

## Decision

- Any failed gate rejects V33 for production and forbids threshold repair.
- A complete historical pass still sets `production_authorized=false`.
- The next permitted step after a pass is exactly 150 untouched future A-share
  trading days with the frozen bundle and policy.
- Shadow evidence must include at least 60 candidates on at least 40 candidate
  days before production can be reconsidered.
