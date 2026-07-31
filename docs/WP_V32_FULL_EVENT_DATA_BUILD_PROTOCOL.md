# WP V32 Full Public-Event Data Build Protocol

## Status

This protocol is frozen before any V32 candidate profit outcome is read.

The build may produce only point-in-time event features for the immutable V24
candidate index. A pass authorizes a separately preregistered model study. It
does not authorize production.

## Immutable Sources

- V24 outcome-blind candidate run: `30635569735`;
- V32 feasibility probe run: `30664863503`;
- V32 feasibility artifact ID: `8806551194`;
- admitted sources:
  - `forecast`;
  - `repurchase`;
  - `share_float`;
  - `block_trade`.

The rejected `express` and `stk_holdertrade` sources are excluded. They cannot
be restored after viewing model outcomes.

## Causal Query Contract

For each V24 target date, each admitted source may use only records from the
five immediately preceding A-share trading days.

- Announcement sources are queried by `ann_date`.
- `block_trade` is queried by `trade_date`.
- Every required source-date query must complete.
- Raw schemas and requested dates must pass.
- Non-A-share rows are excluded using the frozen V32 namespace rule.
- Exact retained-row duplicates are forbidden.
- Same-day events, later revisions outside the five-day window, and all
  candidate profit outcomes are forbidden.

## Frozen Feature Set

Every source contributes:

- event count over the prior five trading days;
- active-event flag;
- trading-day age of the latest event.

`forecast` additionally contributes:

- mean, minimum, and maximum midpoint forecast change;
- share of numeric forecast changes above zero.

`repurchase` additionally contributes:

- log summed amount;
- log summed volume;
- maximum announced high price limit relative to signal price;
- mean announced low price limit relative to signal price.

`share_float` additionally contributes:

- log summed float shares;
- summed and maximum float ratio;
- minimum calendar days from the target date to the announced float date.

`block_trade` additionally contributes:

- log summed amount;
- log summed volume;
- amount-weighted block price relative to signal price;
- latest-date amount-weighted block price relative to signal price.

No text-derived category, unconstrained feature search, outcome-conditioned
transformation, or feature added after viewing returns is allowed.

## Build Gates

The full dataset passes only if:

1. the V24 candidate index digest and row count match its immutable manifest;
2. all required source-date queries pass their frozen contracts;
3. output identities match every V24
   `(trade_date, signal_slot, ts_code)` exactly once;
4. all event count and active-flag fields are complete and internally
   consistent;
5. for each active source, at least 80% of active candidate rows contain at
   least one source-specific numeric detail;
6. the union of admitted events covers at least 5% of candidate rows and at
   least 50% of V24 target dates;
7. all eight V32 probe dates reproduce every admitted-source candidate
   presence flag exactly;
8. no profit outcome is read or written.

## Decision Rule

```text
if every build gate passes:
    model_research_authorized = true
    next_gate = freeze_v32_nested_oos_model_protocol
else:
    model_research_authorized = false
    next_gate = stop_and_diagnose_v32_data_contract
```

Any model study must be frozen in a later commit. It must retain the
executable next-five-minute-close entry, fixed T+1 close exit, transaction
costs, same-day same-slot ranking, nested out-of-sample evaluation, and at
least 150 untouched future A-share shadow trading days.
